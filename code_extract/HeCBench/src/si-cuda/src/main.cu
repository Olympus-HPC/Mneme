#include <cxxopts.hpp>
#include <fmt/core.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/scan.h>
#include "io.hpp"
#include "helpers.cuh"
#include "util.hpp"
#include "host_timer.hpp"

void populateBinaryMatrix(float* input, Dataset* dataset, unsigned int start, unsigned int end);
void populateBinaryTransposeMatrix(float* input, Dataset* dataset, unsigned int partition, unsigned int start, unsigned int end);

int main(int argc, char** argv) {
    try {
        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{2: ^{1}}|\n"
                "└{0:─^{1}}┘\n", "", 51, "GPU Matrix multiplication set intersection"
        );

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        size_t freeDeviceMemory, totalDeviceMemory;

        cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);

        // arguments
        std::string input;
        std::string output;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()
                ("input", "Input dataset path", cxxopts::value<std::string>(input))
                ("output", "Output result path", cxxopts::value<std::string>(output))
                ("partition", "Number of sets to be processed per GPU invocation", cxxopts::value<unsigned int>(partition))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("{}\n", "No input dataset given! Exiting...");
            return 1;
        }

        Dataset* d = readDataset(input);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Dataset characteristics", 51, 25,
                "Cardinality", d->cardinality,
                "Universe", d->universe,
                "Total elements", d->totalElements, ""
        );

        d->universe++;
        partition = std::min(d->cardinality, partition);

        d->offsets = new unsigned int[d->cardinality];

        // calculate offsets
        thrust::exclusive_scan(d->sizes, d->sizes + d->cardinality, d->offsets, 0);

        std::vector<tile> tiles = splitToTiles(d->cardinality, partition);
        std::vector<tile_pair> runs = findTilePairs(d->cardinality, partition);

        size_t combinations = partition * partition;

        unsigned long long outputMemory = runs.size() * combinations * sizeof(float);

        unsigned long long deviceMemory = (sizeof(float) * d->universe * partition * 2)
                                          + (sizeof(float) * combinations);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Launch info", 51, 25,
                "Partition", partition,
                "GPU invocations", runs.size(),
                "Required memory (Output)", formatBytes(outputMemory),
                "Required memory (GPU)", formatBytes(deviceMemory),
                ""
        );

        if (deviceMemory > freeDeviceMemory) {
            fmt::print("Error not enough GPU memory ({})!\nExiting...", formatBytes(freeDeviceMemory));
            return 1;
        }

        std::vector<float> counts(runs.size() * combinations);
        float* hostInput = new float[d->universe * partition];
        float* hostInvInput = new float[d->universe * partition];

        memset(hostInput, 0, d->universe * partition * sizeof(float));
        memset(hostInvInput, 0, d->universe * partition * sizeof(float));

        HostTimer hostTimer;
        Interval* device_offload = hostTimer.add("Device offload");

        float* devInput;
        float* devInvInput;
        float* devOutput;

        errorCheck(cudaMalloc((void**) &devInput, d->universe * partition * sizeof(float)))
        errorCheck(cudaMalloc((void**) &devInvInput, d->universe * partition * sizeof(float)))
        errorCheck(cudaMalloc((void**) &devOutput, combinations * sizeof(float)))

        cublasHandle_t handle;
        cublasCreate_v2(&handle);

        float alpha = 1.f;
        float beta = 0.f;

        unsigned int iter = 0;
        for (unsigned int i = 0; i < tiles.size(); ++i) {
            tile& A = tiles[i];

            populateBinaryMatrix(hostInput, d, A.start, A.end);

            errorCheck(cudaMemcpy(devInput, hostInput, d->universe * partition * sizeof(float), cudaMemcpyHostToDevice))

            for (unsigned int j = i; j < tiles.size(); ++j) {
                tile& B = tiles[j];
                populateBinaryTransposeMatrix(hostInvInput, d, partition, B.start, B.end);

                errorCheck(cudaMemcpy(devInvInput, hostInvInput, d->universe * partition * sizeof(float), cudaMemcpyHostToDevice))

                auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                          B.length, A.length, d->universe,
                                          &alpha, devInvInput, B.length,
                                          devInput, d->universe,
                                          &beta, devOutput, B.length);

                // transfer result back to host
                errorCheck(cudaMemcpy(&counts[0] + (iter * combinations), devOutput, sizeof(float) * combinations,
                                      cudaMemcpyDeviceToHost))

                // clear binary matrix to ensure correctness
                memset(hostInvInput, 0, d->universe * partition * sizeof(float));

                iter++;
            }
            // clear binary matrix to ensure correctness
            memset(hostInput, 0, d->universe * partition * sizeof(float));
        }

        errorCheck(cudaFree(devInput))
        errorCheck(cudaFree(devInvInput))
        errorCheck(cudaFree(devOutput))
        cublasDestroy_v2(handle);

        cudaDeviceSynchronize();

        HostTimer::finish(device_offload);
        hostTimer.print();

        if (!output.empty()) {
            fmt::print("Writing result to file {}\n", output);
            writeResult<float, true>(runs, partition, counts, output);
            fmt::print("Finished\n");
        }

        delete [] hostInput; 
        delete [] hostInvInput; 
        delete [] d->offsets; 

    } catch (const cxxopts::OptionException& e) {
        fmt::print("{}\n", e.what());
        return 1;
    }
    return 0;
}

void populateBinaryMatrix(float* input, Dataset* d, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[idx * d->universe + d->elements[j]] = 1.0f;
        }
        idx++;
    }
}

void populateBinaryTransposeMatrix(float* input, Dataset* d, unsigned int partition, unsigned int start, unsigned int end)
{
    unsigned int idx = 0;
    for (unsigned int i = start; i < end; ++i) {
        for (size_t j = d->offsets[i]; j < d->offsets[i] + d->sizes[i]; ++j) {
            input[d->elements[j] * partition + idx] = 1.0f;
        }
        idx++;
    }
}
