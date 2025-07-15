#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cmath>
#include <chrono>
#include <cub/cub.cuh>
#include "utils.h"

#define BLOCK_SIZE 2048    // in unit of byte, the size of one data block
#define THREAD_SIZE 128     // in unit of datatype, the size of the thread block, so as the size of symbols per iteration
#define WINDOW_SIZE 32     // in unit of datatype, maximum 255, the size of the sliding window, so as the maximum match length
#define INPUT_TYPE uint32_t // define input type
// #define DEBUG

// Define the compress match kernel functions
template <typename T>
__global__ void compressKernelI(const T *input, uint32_t numOfBlocks,
                                uint32_t *__restrict__ flagArrSizeGlobal,
                                uint32_t *__restrict__ compressedDataSizeGlobal,
                                uint8_t *__restrict__ tmpFlagArrGlobal,
                                uint8_t *__restrict__ tmpCompressedDataGlobal,
                                int minEncodeLength)
{
  // Block size in uint of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of datatype
  const uint32_t threadSize = THREAD_SIZE;

  // Allocate shared memory for the lookahead buffer of the whole block, the
  // sliding window is included
  __shared__ T buffer[blockSize];
  __shared__ uint8_t lengthBuffer[blockSize];
  __shared__ uint8_t offsetBuffer[blockSize];
  __shared__ uint32_t prefixBuffer[blockSize + 1];

  // initialize the tid
  int tid = 0;

  // Copy the memeory from global to shared
  for (int i = 0; i < blockSize / threadSize; i++)
  {
    buffer[threadIdx.x + threadSize * i] =
        input[blockIdx.x * blockSize + threadIdx.x + threadSize * i];
  }

  // Synchronize all threads to ensure that the buffer is fully loaded
  __syncthreads();

  // find match for every data point
  for (int iteration = 0; iteration < (int)(blockSize / threadSize);
       iteration++)
  {
    // Initialize the lookahead buffer and the sliding window pointers
    tid = threadIdx.x + iteration * threadSize;
    int bufferStart = tid;
    int bufferPointer = bufferStart;
    int windowStart =
        bufferStart - int(WINDOW_SIZE) < 0 ? 0 : bufferStart - WINDOW_SIZE;
    int windowPointer = windowStart;

    uint8_t maxLen = 0;
    uint8_t maxOffset = 0;
    uint8_t len = 0;
    uint8_t offset = 0;

    while (windowPointer < bufferStart && bufferPointer < blockSize)
    {
      if (buffer[bufferPointer] == buffer[windowPointer])
      {
        if (offset == 0)
        {
          offset = bufferPointer - windowPointer;
        }
        len++;
        bufferPointer++;
      }
      else
      {
        if (len > maxLen)
        {
          maxLen = len;
          maxOffset = offset;
        }
        len = 0;
        offset = 0;
        bufferPointer = bufferStart;
      }
      windowPointer++;
    }
    if (len > maxLen)
    {
      maxLen = len;
      maxOffset = offset;
    }

    lengthBuffer[threadIdx.x + iteration * threadSize] = maxLen;
    offsetBuffer[threadIdx.x + iteration * threadSize] = maxOffset;

    // initialize array as 0
    prefixBuffer[threadIdx.x + iteration * threadSize] = 0;
  }
  __syncthreads();

  // find encode information
  uint32_t flagCount = 0;
  __shared__ uint8_t byteFlagArr[(blockSize / 8)];

  if (threadIdx.x == 0)
  {
    uint8_t flagPosition = 0x01;
    uint8_t byteFlag = 0;

    int encodeIndex = 0;

    while (encodeIndex < blockSize)
    {
      // if length < minEncodeLength, no match is found
      if (lengthBuffer[encodeIndex] < minEncodeLength)
      {
        prefixBuffer[encodeIndex] = sizeof(T);
        encodeIndex++;
      }
      // if length > minEncodeLength, match is found
      else
      {
        prefixBuffer[encodeIndex] = 2;
        encodeIndex += lengthBuffer[encodeIndex];
        byteFlag |= flagPosition;
      }
      // store the flag if there are 8 bits already
      if (flagPosition == 0x80)
      {
        byteFlagArr[flagCount] = byteFlag;
        flagCount++;
        flagPosition = 0x01;
        byteFlag = 0;
        continue;
      }
      flagPosition <<= 1;
    }
    if (flagPosition != 0x01)
    {
      byteFlagArr[flagCount] = byteFlag;
      flagCount++;
    }
  }
  __syncthreads();

  // prefix summation, up-sweep
  int prefixSumOffset = 1;
  for (uint32_t d = blockSize >> 1; d > 0; d = d >> 1)
  {
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = threadIdx.x + iteration * threadSize;
      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;
        prefixBuffer[bi] += prefixBuffer[ai];
      }
      __syncthreads();
    }
    prefixSumOffset *= 2;
  }

  // clear the last element
  if (threadIdx.x == 0)
  {
    // printf("block size: %d flag array size: %d\n", prefixBuffer[blockSize - 1], flagCount);
    compressedDataSizeGlobal[blockIdx.x] = prefixBuffer[blockSize - 1];
    flagArrSizeGlobal[blockIdx.x] = flagCount;
    prefixBuffer[blockSize] = prefixBuffer[blockSize - 1];
    prefixBuffer[blockSize - 1] = 0;
  }
  __syncthreads();

  // prefix summation, down-sweep
  for (int d = 1; d < blockSize; d *= 2)
  {
    prefixSumOffset >>= 1;
    for (int iteration = 0; iteration < (int)(blockSize / threadSize);
         iteration++)
    {
      tid = threadIdx.x + iteration * threadSize;

      if (tid < d)
      {
        int ai = prefixSumOffset * (2 * tid + 1) - 1;
        int bi = prefixSumOffset * (2 * tid + 2) - 1;

        uint32_t t = prefixBuffer[ai];
        prefixBuffer[ai] = prefixBuffer[bi];
        prefixBuffer[bi] += t;
      }
      __syncthreads();
    }
  }

  // encoding phase one
  int tmpCompressedDataGlobalOffset;
  tmpCompressedDataGlobalOffset = blockSize * blockIdx.x * sizeof(T);
  for (int iteration = 0; iteration < (int)(blockSize / threadSize); iteration++)
  {
    tid = threadIdx.x + iteration * threadSize;
    if (prefixBuffer[tid + 1] != prefixBuffer[tid])
    {
      if (lengthBuffer[tid] < minEncodeLength)
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        uint8_t *bytePtr = (uint8_t *)&buffer[tid];
        for (int tmpIndex = 0; tmpIndex < sizeof(T); tmpIndex++)
        {
          tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + tmpIndex] = *(bytePtr + tmpIndex);
        }
      }
      else
      {
        uint32_t tmpOffset = prefixBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset] = lengthBuffer[tid];
        tmpCompressedDataGlobal[tmpCompressedDataGlobalOffset + tmpOffset + 1] = offsetBuffer[tid];
      }
    }
  }

  // Copy the memeory back
  if (threadIdx.x == 0)
  {
    for (int flagArrIndex = 0; flagArrIndex < flagCount; flagArrIndex++)
    {
      tmpFlagArrGlobal[blockSize / 8 * blockIdx.x + flagArrIndex] = byteFlagArr[flagArrIndex];
    }
  }
}

// Define the compress Encode kernel functions
template <typename T>
__global__ void compressKernelIII(uint32_t numOfBlocks,
                                  const uint32_t *__restrict__ flagArrOffsetGlobal,
                                  const uint32_t *__restrict__ compressedDataOffsetGlobal,
                                  const uint8_t *__restrict__ tmpFlagArrGlobal,
                                  const uint8_t *__restrict__ tmpCompressedDataGlobal,
                                  uint8_t *__restrict__ flagArrGlobal,
                                  uint8_t *__restrict__ compressedDataGlobal)
{
  // Block size in uint of bytes
  const int blockSize = BLOCK_SIZE / sizeof(T);

  // Window size in uint of bytes
  const int threadSize = THREAD_SIZE;

  // find block index
  int blockIndex = blockIdx.x;

  int flagArrOffset = flagArrOffsetGlobal[blockIndex];
  int flagArrSize = flagArrOffsetGlobal[blockIndex + 1] - flagArrOffsetGlobal[blockIndex];

  int compressedDataOffset = compressedDataOffsetGlobal[blockIndex];
  int compressedDataSize = compressedDataOffsetGlobal[blockIndex + 1] - compressedDataOffsetGlobal[blockIndex];

  int tid = threadIdx.x;

  while (tid < flagArrSize)
  {
    flagArrGlobal[flagArrOffset + tid] = tmpFlagArrGlobal[blockSize / 8 * blockIndex + tid];
    tid += threadSize;
  }

  tid = threadIdx.x;

  while (tid < compressedDataSize)
  {
    compressedDataGlobal[compressedDataOffset + tid] = tmpCompressedDataGlobal[blockSize * sizeof(T) * blockIndex + tid];
    tid += threadSize;
  }
}

// Define the decompress kernel functions
template <typename T>
__global__ void decompressKernel(T *output, uint32_t numOfBlocks,
                                 const uint32_t *__restrict__ flagArrOffsetGlobal,
                                 const uint32_t *__restrict__ compressedDataOffsetGlobal,
                                 const uint8_t *__restrict__ flagArrGlobal,
                                 const uint8_t *__restrict__ compressedDataGlobal)
{
  // Block size in unit of datatype
  const uint32_t blockSize = BLOCK_SIZE / sizeof(T);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < numOfBlocks)
  {
    int flagArrOffset = flagArrOffsetGlobal[tid];
    int flagArrSize = flagArrOffsetGlobal[tid + 1] - flagArrOffsetGlobal[tid];

    int compressedDataOffset = compressedDataOffsetGlobal[tid];

    uint32_t dataPointsIndex = 0;
    uint32_t compressedDataIndex = 0;

    uint8_t byteFlag;

    for (int flagArrayIndex = 0; flagArrayIndex < flagArrSize; flagArrayIndex++)
    {
      byteFlag = flagArrGlobal[flagArrOffset + flagArrayIndex];

      for (int bitCount = 0; bitCount < 8; bitCount++)
      {
        int matchFlag = (byteFlag >> bitCount) & 0x1;
        if (matchFlag == 1)
        {
          int length = compressedDataGlobal[compressedDataOffset + compressedDataIndex];
          int offset = compressedDataGlobal[compressedDataOffset + compressedDataIndex + 1];
          compressedDataIndex += 2;
          int dataPointsStart = dataPointsIndex;
          for (int tmpDecompIndex = 0; tmpDecompIndex < length; tmpDecompIndex++)
          {
            output[tid * blockSize + dataPointsIndex] = output[tid * blockSize + dataPointsStart - offset + tmpDecompIndex];
            dataPointsIndex++;
          }
        }
        else
        {
          uint8_t *tmpPtr = (uint8_t *)&output[tid * blockSize + dataPointsIndex];
          for (int tmpDecompIndex = 0; tmpDecompIndex < sizeof(T); tmpDecompIndex++)
          {
            *(tmpPtr + tmpDecompIndex) = compressedDataGlobal[compressedDataOffset + compressedDataIndex + tmpDecompIndex];
          }

          compressedDataIndex += sizeof(T);
          dataPointsIndex++;
        }
        if (dataPointsIndex >= blockSize)
        {
          return;
        }
      }
    }
  }
}

int main(int argc, char *argv[])
{
  std::string inputFileName;
  int opt;
  int repeat = 1;

  /* parse command line */
  while ((opt = getopt(argc, argv, "i:n:h")) != -1)
  {
    switch (opt)
    {
    case 'i': /* input file name */
      inputFileName = optarg;
      break;

    case 'n':
      repeat = atoi(optarg);
      break;

    case 'h': /* help */
      printf(" Usage for compression and decompression: ./main -i {inputfile} -n {repeat}\n");
      return 0;
    }
  }

  INPUT_TYPE *hostArray = io::read_binary_to_new_array<INPUT_TYPE>(inputFileName);

#ifdef DEBUG
  int debugOffset = 0;

  printf("print the first 1024 elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    std::cout << hostArray[tmpIndex + debugOffset] << "\t";
  }
  printf("\n");
#endif

  INPUT_TYPE *deviceArray;
  INPUT_TYPE *deviceOutput;
  uint32_t fileSize = io::FileSize(inputFileName);

  uint32_t *flagArrSizeGlobal;
  uint32_t *flagArrOffsetGlobal;
  uint32_t *compressedDataSizeGlobal;
  uint32_t *compressedDataOffsetGlobal;
  uint8_t *tmpFlagArrGlobal;
  uint8_t *tmpCompressedDataGlobal;
  uint8_t *flagArrGlobal;
  uint8_t *compressedDataGlobal;

  // calculate the padding size, unit in bytes
  uint32_t paddingSize = fileSize % BLOCK_SIZE == 0 ? 0 : BLOCK_SIZE - fileSize % BLOCK_SIZE;

  // calculate the datatype size, unit in datatype
  uint32_t datatypeSize =
      static_cast<uint32_t>((fileSize + paddingSize) / sizeof(INPUT_TYPE));

  uint32_t numOfBlocks = datatypeSize * sizeof(INPUT_TYPE) / BLOCK_SIZE;

  INPUT_TYPE *hostOutput = (INPUT_TYPE *)malloc(sizeof(INPUT_TYPE) * datatypeSize);

  // malloc the device buffer and set it as 0
  cudaMalloc((void **)&deviceArray, fileSize + paddingSize);
  cudaMalloc((void **)&deviceOutput, fileSize + paddingSize);

  cudaMalloc((void **)&flagArrSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMalloc((void **)&flagArrOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMalloc((void **)&compressedDataSizeGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMalloc((void **)&compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMalloc((void **)&tmpFlagArrGlobal, sizeof(uint8_t) * datatypeSize / 8);
  cudaMalloc((void **)&tmpCompressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize);
  cudaMalloc((void **)&flagArrGlobal, sizeof(uint8_t) * datatypeSize / 8);
  cudaMalloc((void **)&compressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize);

  // initialize the mem as 0
  cudaMemset(deviceArray, 0, fileSize + paddingSize);
  cudaMemset(deviceOutput, 0, fileSize + paddingSize);
  cudaMemset(flagArrSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMemset(flagArrOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMemset(compressedDataSizeGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMemset(compressedDataOffsetGlobal, 0, sizeof(uint32_t) * (numOfBlocks + 1));
  cudaMemset(tmpFlagArrGlobal, 0, sizeof(uint8_t) * datatypeSize / 8);
  cudaMemset(tmpCompressedDataGlobal, 0, sizeof(INPUT_TYPE) * datatypeSize);

  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  // copy host memory to device
  cudaMemcpy(deviceArray, hostArray, fileSize, cudaMemcpyHostToDevice);

  // printf("num of blocks: %d\nfile size: %d\npadding size: %d\n data type length: %d\n", numOfBlocks, fileSize, paddingSize, datatypeSize);

  dim3 gridDim(numOfBlocks);
  dim3 blockDim(THREAD_SIZE);

  dim3 deGridDim(ceil(float(numOfBlocks) / 32));
  dim3 deBlockDim(32);

  uint32_t *flagArrOffsetGlobalHost;
  uint32_t *compressedDataOffsetGlobalHost;
  uint8_t *tmpFlagArrGlobalHost;
  uint8_t *tmpCompressedDataGlobalHost;
  uint8_t *flagArrGlobalHost;
  uint8_t *compressedDataGlobalHost;

  flagArrOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
  compressedDataOffsetGlobalHost = (uint32_t *)malloc(sizeof(uint32_t) * (numOfBlocks + 1));
  tmpFlagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * datatypeSize / 8);
  tmpCompressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * datatypeSize);
  flagArrGlobalHost = (uint8_t *)malloc(sizeof(uint8_t) * datatypeSize / 8);
  compressedDataGlobalHost = (uint8_t *)malloc(sizeof(INPUT_TYPE) * datatypeSize);

  int minEncodeLength = sizeof(INPUT_TYPE) == 1 ? 2 : 1;

  cudaDeviceSynchronize();
  auto compStart = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // launch kernels
    compressKernelI<INPUT_TYPE><<<gridDim, blockDim>>>(deviceArray, numOfBlocks, flagArrSizeGlobal, compressedDataSizeGlobal, tmpFlagArrGlobal, tmpCompressedDataGlobal, minEncodeLength);

    // Determine temporary device storage requirements
    void *flag_d_temp_storage = NULL;
    size_t flag_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal, numOfBlocks + 1);

    // Allocate temporary storage
    cudaMalloc(&flag_d_temp_storage, flag_temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(flag_d_temp_storage, flag_temp_storage_bytes, flagArrSizeGlobal, flagArrOffsetGlobal, numOfBlocks + 1);

    // Determine temporary device storage requirements
    void *data_d_temp_storage = NULL;
    size_t data_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes, compressedDataSizeGlobal, compressedDataOffsetGlobal, numOfBlocks + 1);

    // Allocate temporary storage
    cudaMalloc(&data_d_temp_storage, data_temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(data_d_temp_storage, data_temp_storage_bytes, compressedDataSizeGlobal, compressedDataOffsetGlobal, numOfBlocks + 1);

    compressKernelIII<INPUT_TYPE><<<gridDim, blockDim>>>(numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal, tmpFlagArrGlobal, tmpCompressedDataGlobal, flagArrGlobal, compressedDataGlobal);

    cudaDeviceSynchronize();

    cudaFree(flag_d_temp_storage);
    cudaFree(data_d_temp_storage);
  }
  auto compStop = std::chrono::steady_clock::now();

  auto decompStart = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    decompressKernel<INPUT_TYPE><<<deGridDim, deBlockDim>>>(deviceOutput, numOfBlocks, flagArrOffsetGlobal, compressedDataOffsetGlobal, flagArrGlobal, compressedDataGlobal);
    cudaDeviceSynchronize();
  }
  auto decompStop = std::chrono::steady_clock::now();

  // copy the memory back to global
  cudaMemcpy(flagArrOffsetGlobalHost, flagArrOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1), cudaMemcpyDeviceToHost);
  cudaMemcpy(compressedDataOffsetGlobalHost, compressedDataOffsetGlobal, sizeof(uint32_t) * (numOfBlocks + 1), cudaMemcpyDeviceToHost);
  cudaMemcpy(tmpFlagArrGlobalHost, tmpFlagArrGlobal, sizeof(uint8_t) * datatypeSize / 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(tmpCompressedDataGlobalHost, tmpCompressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(flagArrGlobalHost, flagArrGlobal, sizeof(uint8_t) * datatypeSize / 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(compressedDataGlobalHost, compressedDataGlobal, sizeof(INPUT_TYPE) * datatypeSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(hostOutput, deviceOutput, fileSize, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  printf("print the first 1024 flag array offset elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", flagArrOffsetGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 compressed data offset elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", compressedDataOffsetGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 flag array elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", flagArrGlobalHost[tmpIndex]);
  }
  printf("\n");

  printf("print the first 1024 compressed data elements:\n");
  for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  {
    printf("%d\t", compressedDataGlobalHost[tmpIndex]);
  }
  printf("\n");

  // printf("print the first 1024 tmp flag array elements:\n");
  // for (int tmpIndex = 0; tmpIndex < 1024; tmpIndex++)
  // {
  //   printf("%d\t", tmpFlagArrGlobalHost[tmpIndex]);
  // }
  // printf("\n");
#endif

  // verify the final output
  for (uint32_t verifyIndex = 0; verifyIndex < fileSize / sizeof(INPUT_TYPE); verifyIndex++)
  {
    if (hostArray[verifyIndex] != hostOutput[verifyIndex])
    {
      printf("verification failed!!! Index %d is wrong\n", verifyIndex);
      std::cout << "hostArray: " << hostArray[verifyIndex] << ", hostOutput: " << hostOutput[verifyIndex] << std::endl;
      break;
    }
  }

  float originalSize = fileSize;
  float compressedSize = sizeof(uint32_t) * (numOfBlocks + 1) * 2 + flagArrOffsetGlobalHost[numOfBlocks] + compressedDataOffsetGlobalHost[numOfBlocks];
  float compressionRatio = originalSize / compressedSize;
  std::cout << "compression ratio: " << compressionRatio << std::endl;

  float compTime = std::chrono::duration<float, std::milli>(compStop - compStart).count();
  float decompTime = std::chrono::duration<float, std::milli>(decompStop - decompStart).count();
  float compTp = float(fileSize) / 1024 / 1024 / (compTime / repeat);
  float decompTp = float(fileSize) / 1024 / 1024 / (decompTime / repeat);
  std::cout << "compression e2e throughput: " << compTp << " GB/s" << std::endl;
  std::cout << "decompression e2e throughput: " << decompTp << " GB/s" << std::endl;

  // free dynamic arrays
  free(flagArrOffsetGlobalHost);
  free(compressedDataOffsetGlobalHost);
  free(tmpFlagArrGlobalHost);
  free(tmpCompressedDataGlobalHost);
  free(flagArrGlobalHost);
  free(compressedDataGlobalHost);

  cudaFree(deviceArray);
  cudaFree(deviceOutput);

  cudaFree(flagArrSizeGlobal);
  cudaFree(flagArrOffsetGlobal);
  cudaFree(compressedDataSizeGlobal);
  cudaFree(compressedDataOffsetGlobal);
  cudaFree(tmpFlagArrGlobal);
  cudaFree(tmpCompressedDataGlobal);
  cudaFree(flagArrGlobal);
  cudaFree(compressedDataGlobal);

  free(hostOutput);
  delete hostArray;

  return 0;
}
