/*
 Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

#define BufSize 0x1000
#define Iterations 0x100
#define TotalStreams 4
#define TotalBufs 4


class PerfStreamCreateCopyDestroy {
  private:
    unsigned int numBuffers_;
    unsigned int numStreams_;
    const size_t totalStreams_[TotalStreams];
    const size_t totalBuffers_[TotalBufs];
  public:
    PerfStreamCreateCopyDestroy() : numBuffers_(0), numStreams_(0),
                                       totalStreams_{1, 2, 4, 8},
                                       totalBuffers_{1, 100, 1000, 5000} {};
    ~PerfStreamCreateCopyDestroy() {};
    void open(int deviceID);
    void run_baseline(unsigned int testNumber);
    void run_stream(unsigned int testNumber);
};

void PerfStreamCreateCopyDestroy::run_stream(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter = Iterations / (numStreams_ * ((size_t)1 << (testNumber / TotalBufs + 1)));
  cudaStream_t streams[numStreams_];

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  float* dSrc[numBuffers_];
  size_t nBytes = BufSize * sizeof(float);

  for (size_t b = 0; b < numBuffers_; ++b) {
    cudaMalloc(&dSrc[b], nBytes);
  }

  float* hSrc;
  hSrc = new float[nBytes];
  for (size_t i = 0; i < BufSize; i++) {
    hSrc[i] = 1.618f + i;
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      cudaStreamCreate(&streams[s]);
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        cudaMemcpyAsync(dSrc[b], hSrc, nBytes, cudaMemcpyHostToDevice, streams[s]);
        //cudaMemcpyWithStream(dSrc[b], hSrc, nBytes, cudaMemcpyHostToDevice, streams[s]);
      }
    }

    for (size_t s = 0; s < numStreams_; ++s) {
      cudaStreamSynchronize(streams[s]);
      cudaStreamDestroy(streams[s]);
    }
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  std::cout << "[Stream] Create+Copy+Synchronize+Destroy time for " << numStreams_ << " streams and "
       << std::setw(4) << numBuffers_ << " buffers " << " and " << std::setw(4)
       << iter << " iterations " << time << " (ms) " << std::endl;

  delete [] hSrc;
  for (size_t b = 0; b < numBuffers_; ++b) {
    cudaFree(dSrc[b]);
  }
}

// no streams will be created
void PerfStreamCreateCopyDestroy::run_baseline(unsigned int testNumber) {
  numStreams_ = totalStreams_[testNumber % TotalStreams];
  size_t iter = Iterations / (numStreams_ * ((size_t)1 << (testNumber / TotalBufs + 1)));

  numBuffers_ = totalBuffers_[testNumber / TotalBufs];
  float* dSrc[numBuffers_];
  size_t nBytes = BufSize * sizeof(float);

  for (size_t b = 0; b < numBuffers_; ++b) {
    cudaMalloc(&dSrc[b], nBytes);
  }

  float* hSrc;
  hSrc = new float[nBytes];
  for (size_t i = 0; i < BufSize; i++) {
    hSrc[i] = 1.618f + i;
  }

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < iter; ++i) {
    for (size_t s = 0; s < numStreams_; ++s) {
      for (size_t b = 0; b < numBuffers_; ++b) {
        cudaMemcpyAsync(dSrc[b], hSrc, nBytes, cudaMemcpyHostToDevice);
      }
    }
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> diff = end - start;

  auto time = static_cast<float>(diff.count() * 1000 / (iter * numStreams_));

  std::cout << "[Baseline] Copy+Synchronize time for the default stream and "
       << std::setw(4) << numBuffers_ << " buffers " << " and " << std::setw(4)
       << iter << " iterations " << time << " (ms) " << std::endl;

  delete [] hSrc;
  for (size_t b = 0; b < numBuffers_; ++b) {
    cudaFree(dSrc[b]);
  }
}

int main(int argc, char* argv[]) {
  PerfStreamCreateCopyDestroy streamCCD;

  streamCCD.run_baseline(0); // warmup
  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    streamCCD.run_baseline(testCase);
  }

  streamCCD.run_stream(0); // warmup
  for (auto testCase = 0; testCase < TotalStreams * TotalBufs; testCase++) {
    streamCCD.run_stream(testCase);
  }
}
