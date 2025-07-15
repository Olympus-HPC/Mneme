#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>

#define threadsPerBlock  256

static inline __device__
void gpuAtomicAddNoReturn(int *address, int val) {
  atomicAdd(address, val);
}

#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
                 i += gridDim.x * blockDim.x)

// Memory types used for the bincount implementations
enum class DeviceMemoryType { SHARED, GLOBAL };

template <typename input_t, typename IndexType>
__device__ static IndexType
getBin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins)
{
  IndexType bin = (v - minvalue) * nbins / (maxvalue - minvalue);
  // (only applicable for histc)
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end) the last bin is inclusive at both, i.e. [start, end], in
  // order to include maxvalue if exists therefore when bin == nbins, adjust bin
  // to the last bin
  if (bin == nbins) bin--;
  return bin;
}

// Kernel for computing the histogram of the input
template <typename output_t,
          typename input_t,
          typename IndexType,
          DeviceMemoryType MemoryType>
__global__ void bincount (
       output_t *output,
  const input_t *input,
  IndexType nbins,
  input_t minvalue,
  input_t maxvalue,
  IndexType input_size,
  IndexType output_size)
{
  extern __shared__ unsigned char my_smem[];
  output_t* smem = nullptr;

  if (MemoryType == DeviceMemoryType::SHARED) {
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<output_t*>(my_smem);
    for (IndexType i = threadIdx.x; i < nbins; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();

    FOR_KERNEL_LOOP(linearIndex, input_size) {
      const auto v = input[linearIndex];

      if (v >= minvalue && v <= maxvalue) {
        const IndexType bin = getBin<input_t, IndexType>(
                              v, minvalue, maxvalue, nbins);
        gpuAtomicAddNoReturn(&smem[bin], 1);
      }
    }
    __syncthreads();

    // Atomically update output bin count.
    for (IndexType i = threadIdx.x; i < nbins; i += blockDim.x) {
      gpuAtomicAddNoReturn(&output[i], smem[i]);
    }

  } else {
    ////////////////////////// Global memory //////////////////////////
    // atomically add to the output tensor
    // compute histogram for the block
    FOR_KERNEL_LOOP(linearIndex, input_size) {
      const auto v = input[linearIndex];
      if (v >= minvalue && v <= maxvalue) {
        const IndexType bin = getBin<input_t, IndexType>(
                              v, minvalue, maxvalue, nbins);
        gpuAtomicAddNoReturn(&output[bin], 1);
      }
    }
  }
}

#define HANDLE_CASE(MEMORY_TYPE, SHARED_MEM)                 \
  auto start = std::chrono::steady_clock::now();             \
  for (int i = 0; i < repeat; i++)                           \
  bincount<                                                  \
      output_t,                                              \
      input_t,                                               \
      IndexType,                                             \
      MEMORY_TYPE><<<grid, block, SHARED_MEM>>>(             \
      d_output,                                              \
      d_input,                                               \
      nbins,                                                 \
      input_minvalue,                                        \
      input_maxvalue,                                        \
      input_size,                                            \
      output_size);                                          \
  cudaDeviceSynchronize();                                   \
  auto end = std::chrono::steady_clock::now();               \
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(); \
  printf("Average execution time of bincount kernel: %f (us)\n", \
         (time * 1e-3f) / repeat)

#define HANDLE_SWITCH_CASE(mType)                           \
  switch (mType) {                                          \
    case DeviceMemoryType::SHARED: {                        \
      HANDLE_CASE(DeviceMemoryType::SHARED, sharedMem);     \
      break;                                                \
    }                                                       \
    default: {                                              \
      HANDLE_CASE(DeviceMemoryType::GLOBAL, 0);             \
    }                                                       \
  }

/*
  Calculate the frequency of the input values.
  3 implementations based of input size and memory usage:
    case: sufficient shared mem
        SHARED: Each block atomically adds to it's own **shared** hist copy,
        then atomically updates the global tensor.
    case: insufficient shared memory
        GLOBAL: all threads atomically update to a single **global** hist copy.
 */
template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat)
{
  size_t input_size_bytes = sizeof(input_t) * input_size;

  input_t *input = (input_t*) malloc (input_size_bytes); 

  // https://cplusplus.com/reference/random/normal_distribution/
  std::default_random_engine generator (123);
  std::normal_distribution<input_t> distribution(5.0,2.0);
  for (int i = 0; i < input_size; i++) {
    input[i] = distribution(generator);
  }

  auto min_iter = std::min_element(input, input+input_size);
  auto max_iter = std::max_element(input, input+input_size);
  
  input_t input_minvalue = *min_iter;
  input_t input_maxvalue = *max_iter;
  printf("Input min, max values: (%f %f)\n", (float)input_minvalue, (float)input_maxvalue);

  input_t *d_input;
  cudaMalloc((void**)&d_input, input_size_bytes);
  cudaMemcpy(d_input, input, input_size_bytes, cudaMemcpyHostToDevice);

  int maxSharedMemory;
  cudaDeviceGetAttribute(&maxSharedMemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  printf("Maximum shared local memory size per block in bytes: %d\n", maxSharedMemory);

  for (IndexType nbins = 768; nbins <= 768 * 32; nbins = nbins * 2) {

    printf("\nNumber of bins: %d\n", nbins);
    IndexType sharedMem = nbins * sizeof(output_t);

    IndexType output_size = nbins;
    size_t output_size_bytes = sizeof(output_t) * output_size;
    output_t *output = (output_t*) malloc (output_size_bytes); 

    output_t *d_output;
    cudaMalloc((void**)&d_output, output_size_bytes);

    dim3 grid ((input_size + threadsPerBlock - 1) / threadsPerBlock);
    dim3 block (threadsPerBlock);

    // determine memory type to use in the kernel
    printf("bincount using global atomics\n");

    DeviceMemoryType memType = DeviceMemoryType::GLOBAL;
    cudaMemset(d_output, 0, output_size_bytes);
    HANDLE_SWITCH_CASE(memType)
    cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

    auto min_iter = std::min_element(output, output+output_size);
    auto max_iter = std::max_element(output, output+output_size);
    output_t minvalue = *min_iter;
    output_t maxvalue = *max_iter;
    printf("Output min, median, max values: (%ld %ld %ld)\n",
           (int64_t)minvalue / repeat,
           (int64_t)output[output_size/2] / repeat,
           (int64_t)maxvalue / repeat);

    if (sharedMem <= maxSharedMemory) {
      printf("\n");
      printf("bincount using global and local atomics\n");

      cudaMemset(d_output, 0, output_size_bytes);
      memType = DeviceMemoryType::SHARED;
      HANDLE_SWITCH_CASE(memType)
      cudaMemcpy(output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);

      auto min_iter = std::min_element(output, output+output_size);
      auto max_iter = std::max_element(output, output+output_size);
      output_t minvalue = *min_iter;
      output_t maxvalue = *max_iter;
      printf("Output min, median, max values: (%ld %ld %ld)\n\n",
             (int64_t)minvalue / repeat,
             (int64_t)output[output_size/2] / repeat,
             (int64_t)maxvalue / repeat);
    }

    cudaFree(d_output);
    free(output);
  }

  cudaFree(d_input);
  free(input);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval<int, float, int>(n, repeat);

  return 0; 
}
