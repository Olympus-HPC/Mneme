#include "utils.h"

// DEVICE FUNCTIONS

__device__ int clamp(size_t const val, int const max)
{
  return static_cast<int>(min(static_cast<size_t>(max), val));
}

template <typename T>
__device__ void readMinAndMax(
    T const* const inMin,
    T const* const inMax,
    T* const minBuffer,
    T* const maxBuffer,
    int const blockOffset,
    int const blockEnd)
{
  static_assert(
      BLOCK_SIZE <= BLOCK_WIDTH,
      "BLOCK_SIZE must be less than or equal to BLOCK_WIDTH");

  if (threadIdx.x < blockEnd) {
    T localMin = inMin[blockOffset + threadIdx.x];
    T localMax = inMax[blockOffset + threadIdx.x];
    for (int i = threadIdx.x + BLOCK_SIZE; i < BLOCK_WIDTH && i < blockEnd;
         i += BLOCK_SIZE) {
      int const readIdx = blockOffset + i;
      localMin = min(inMin[readIdx], localMin);
      localMax = max(inMax[readIdx], localMax);
    }
    minBuffer[threadIdx.x] = localMin;
    maxBuffer[threadIdx.x] = localMax;
  }
}

template <typename T>
__device__ void
reduceMinAndMax(T* const minBuffer, T* const maxBuffer, int const blockEnd)
{
  // cooperatively compute min and max
  for (int d = BLOCK_SIZE / 2; d > 0; d >>= 1) {
    if (threadIdx.x < BLOCK_SIZE / 2) {
      int const idx = threadIdx.x;
      if (idx < d && idx + d < blockEnd) {
        minBuffer[idx] = min(minBuffer[idx], minBuffer[d + idx]);
      }
    } else {
      int const idx = threadIdx.x - (BLOCK_SIZE / 2);
      if (idx < d && idx + d < blockEnd) {
        maxBuffer[idx] = max(maxBuffer[idx], maxBuffer[d + idx]);
      }
    }
    __syncthreads();
  }
}


// KERNELS

template <typename LIMIT, typename INPUT>
__global__ void bitPackConfigScanKernel(
    LIMIT* const minValue,
    LIMIT* const maxValue,
    INPUT const* const in,
    const size_t* const numDevice)
{
  static_assert(BLOCK_SIZE % 64 == 0, "BLOCK_SIZE must a multiple of 64");

  //assert(BLOCK_SIZE == blockDim.x);

  const size_t num = *numDevice;
  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  //assert(num > 0);
  //assert(threadIdx.x < BLOCK_SIZE);

  if (blockIdx.x < numBlocks) {
    // each block processes it's chunks, updates min/max
    __shared__ LIMIT minBuffer[BLOCK_SIZE];
    __shared__ LIMIT maxBuffer[BLOCK_SIZE];

    LIMIT localMin = 0;
    LIMIT localMax = 0;

    int lastThread = 0;
    for (int block = blockIdx.x; block < numBlocks; block += gridDim.x) {

      int const blockOffset = BLOCK_SIZE * block;
      int const blockEnd = min(static_cast<int>(num) - blockOffset, BLOCK_SIZE);

      lastThread = max(lastThread, blockEnd);

      if (threadIdx.x < blockEnd) {
        LIMIT const val = in[blockOffset + threadIdx.x];
        if (block == blockIdx.x) {
          // first iteration just set values
          localMax = val;
          localMin = val;
        } else {
          localMin = min(val, localMin);
          localMax = max(val, localMax);
        }
      }
    }

    minBuffer[threadIdx.x] = localMin;
    maxBuffer[threadIdx.x] = localMax;

    __syncthreads();

    // cooperatively compute min and max
    reduceMinAndMax(minBuffer, maxBuffer, lastThread);

    if (threadIdx.x == 0) {
      minValue[blockIdx.x] = minBuffer[0];
      maxValue[blockIdx.x] = maxBuffer[0];
    }
  }
}

template <typename LIMIT, typename INPUT>
__global__ void bitPackConfigFinalizeKernel(
    LIMIT const* const inMin,
    LIMIT const* const inMax,
    unsigned char* const numBitsPtr,
    INPUT* const outMinValPtr,
    const size_t* const numDevice)
{
  static_assert(
      BLOCK_SIZE <= BLOCK_WIDTH,
      "BLOCK_SIZE must be less than or equal to BLOCK_WIDTH");
  static_assert(
      BLOCK_WIDTH % BLOCK_SIZE == 0,
      "BLOCK_WIDTH must be a multiple of BLOCK_SIZE");
  static_assert(BLOCK_SIZE % 64 == 0, "BLOCK_SIZE must a multiple of 64");

  //assert(blockIdx.x == 0);

  const size_t num = min(
      roundUpDiv(*numDevice, BLOCK_SIZE), static_cast<size_t>(BLOCK_WIDTH));

  //assert(num > 0);

  // each block processes it's chunk, updates min/max, and the calculates
  // the bitwidth based on the last update
  __shared__ LIMIT minBuffer[BLOCK_SIZE];
  __shared__ LIMIT maxBuffer[BLOCK_SIZE];

  // load data
  readMinAndMax(inMin, inMax, minBuffer, maxBuffer, 0, num);

  __syncthreads();

  // cooperatively compute min and max
  reduceMinAndMax(minBuffer, maxBuffer, min(BLOCK_SIZE, (int)num));

  if (threadIdx.x == 0) {
    *outMinValPtr = static_cast<INPUT>(minBuffer[0]);
    // we need to update the number of bits
    if (sizeof(LIMIT) > sizeof(int)) {
      const uint64_t range = static_cast<uint64_t>(maxBuffer[0]) - static_cast<uint64_t>(minBuffer[0]);
      // need 64 bit clz
      *numBitsPtr = sizeof(uint64_t) * 8 - __clzll(range);
    } else {
      const uint32_t range = static_cast<uint32_t>(maxBuffer[0]) - static_cast<uint32_t>(minBuffer[0]);
      // can use 32 bit clz
      *numBitsPtr = sizeof(uint32_t) * 8 - __clz(range);
    }
  }
}

template <typename INPUT, typename OUTPUT>
__global__ void bitPackKernel(
    unsigned char const* const numBitsPtr,
    INPUT const* const valueOffsetPtr,
    OUTPUT* const outPtr,
    INPUT const* const in,
    const size_t* const numDevice)
{
  using UINPUT = typename std::make_unsigned<INPUT>::type;

  const size_t num = *numDevice;

  const int numBlocks = roundUpDiv(num, BLOCK_SIZE);

  OUTPUT* const out = outPtr;
  int const numBits = *numBitsPtr;
  INPUT const valueOffset = *valueOffsetPtr;

  __shared__ UINPUT inBuffer[BLOCK_SIZE];

  for (int blockId = blockIdx.x; blockId < numBlocks; blockId += gridDim.x) {
    // The kernel works by assigning an output index to each thread.
    // The kernel then iterates over chunks of input, filling the bits
    // for each thread.
    // And then writing the stored bits to the output.
    int const outputIdx = threadIdx.x + blockId * BLOCK_SIZE;
    //assert(outputIdx >= 0);
    //assert(*numBitsPtr <= sizeof(INPUT) * 8U);

    size_t const bitStart = outputIdx * sizeof(*out) * 8U;
    size_t const bitEnd = bitStart + (sizeof(*out) * 8U);

    int const startIdx = clamp(bitStart / static_cast<size_t>(numBits), num);
    int const endIdx = clamp(roundUpDiv(bitEnd, numBits), num);
    //assert(startIdx >= 0);

    size_t const blockStartBit = blockId * BLOCK_SIZE * sizeof(*out) * 8U;
    size_t const blockEndBit = (blockId + 1) * BLOCK_SIZE * sizeof(*out) * 8U;
    //assert(blockStartBit < blockEndBit);

    int const blockStartIdx = clamp(
        roundDownTo(blockStartBit / static_cast<size_t>(numBits), BLOCK_SIZE),
        num);
    int const blockEndIdx
        = clamp(roundUpTo(roundUpDiv(blockEndBit, numBits), BLOCK_SIZE), num);
    //assert(blockStartIdx >= 0);
    //assert(blockStartIdx <= blockEndIdx);

    OUTPUT val = 0;
    for (int bufferStart = blockStartIdx; bufferStart < blockEndIdx;
         bufferStart += BLOCK_SIZE) {
      __syncthreads();

      // fill input buffer
      int const inputIdx = bufferStart + threadIdx.x;
      if (inputIdx < num) {
        inBuffer[threadIdx.x] = in[inputIdx] - valueOffset;
      }

      __syncthreads();

      int const currentStartIdx = max(startIdx, bufferStart);
      int const currentEndIdx = min(endIdx, bufferStart + BLOCK_SIZE);

      for (int idx = currentStartIdx; idx < currentEndIdx; ++idx) {
        int const localIdx = idx - bufferStart;

        // keep only bits we're interested in
        OUTPUT bits = static_cast<OUTPUT>(inBuffer[localIdx]);
        int const offset = static_cast<int>(
            static_cast<ssize_t>(idx * numBits)
            - static_cast<ssize_t>(bitStart));
        //assert(std::abs(offset) < sizeof(bits) * 8U);

        if (offset > 0) {
          bits <<= offset;
        } else {
          bits >>= -offset;
        }

        // update b
        val |= bits;
      }
    }

    if (startIdx < num) {
      out[outputIdx] = val;
    }
  }
}


/**
 * @brief Launch of all of the kernels necessary for the configuration step of
 * bit packing.
 *
 * @tparam LIMIT The type used for min/max values.
 * @tparam INPUT The type being reduced.
 * @param minValueScratch Space used by the kernels to reduce the minimum
 * values. Must be at least the size returned by `getReduceScratchSpaceSize()`.
 * @param maxValueScratch Space used by the kernels to reduce the maximum
 * values. Must be at least the size returned by `getReduceScratchSpaceSize()`.
 * @param minValOutPtr The place to put the actual minimum value of the entire
 * series (output).
 * @param numBitsPtr The number of bits to compact to (output).
 * @param in The input to be compressed.
 * @param numDevice The number of elements on the device.
 * @param maxNum The maximum number of elements in the input.
 */
template <typename LIMIT, typename INPUT>
void bitPackConfigLaunch(
    LIMIT* const minValueScratch,
    LIMIT* const maxValueScratch,
    INPUT* const minValOutPtr,
    unsigned char* const numBitsPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    size_t const maxNum)
{
  const dim3 grid(
      min(BLOCK_WIDTH, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))));
  const dim3 block(BLOCK_SIZE);

  // make sure the result will fit in a single block for the finalize kernel
  bitPackConfigScanKernel<<<grid, block>>>(
      minValueScratch, maxValueScratch, in, numDevice);
#ifdef DEBUG
  cudaError_t err;
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch bitPackConfigScanKernel "
        "kernel: "
        + std::to_string(err));
  }
#endif

  // determine numBits and convert min value
  bitPackConfigFinalizeKernel<<<dim3(1), block>>>(
      minValueScratch, maxValueScratch, numBitsPtr, minValOutPtr, numDevice);
#ifdef DEBUG
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch bitPackConfigFinalizeKernel "
        "kernel: "
        + std::to_string(err));
  }
#endif
}

template <typename INPUT, typename OUTPUT>
void bitPackLaunch(
    const INPUT * const minValueDevicePtr,
    unsigned char const* const numBitsDevicePtr,
    OUTPUT* const outPtr,
    INPUT const* const in,
    const size_t* const numDevice,
    const size_t maxNum)
{
  static_assert(
      BLOCK_SIZE % (sizeof(OUTPUT) * 8U) == 0,
      "Block size must be a multiple of output word size.");

  dim3 const grid(
      std::min(4096, static_cast<int>(roundUpDiv(maxNum, BLOCK_SIZE))));
  dim3 const block(BLOCK_SIZE);

  bitPackKernel<<<grid, block>>>(
      numBitsDevicePtr, minValueDevicePtr, outPtr, in, numDevice);

#ifdef DEBUG
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to launch bitPackKernel kernel: " + std::to_string(err));
  }
#endif
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackFixedBitAndMinInternal(
    void const* const minValueDevicePtr,
    unsigned char const* const numBitsDevicePtr,
    void* const /* workspace */,
    void* const outPtr,
    void const* const in,
    const size_t* const numDevice,
    size_t const maxNum)
{
  // type qualifier is meaningless on cast type
  OUT* const outputTypedPtr = reinterpret_cast<OUT*>(outPtr);
  IN const* const inputTyped = static_cast<IN const*>(in);

  bitPackLaunch(
      reinterpret_cast<const IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      outputTypedPtr,
      inputTyped,
      numDevice,
      maxNum);
}

template <typename IN, typename OUT, typename LIMIT>
void bitPackInternal(
    void* const workspace,
    void* const outPtr,
    void const* const in,
    const size_t* const numDevice,
    size_t const maxNum,
    void* const minValueDevicePtr,
    unsigned char* const numBitsDevicePtr)
{
  // cast voids to known types
  LIMIT* const maxValueTyped = static_cast<LIMIT*>(workspace);
  LIMIT* const minValueTyped = maxValueTyped + getReduceScratchSpaceSize(maxNum);
  IN const* const inputTyped = static_cast<IN const*>(in);

  // determine min, and bit width
  bitPackConfigLaunch(
      minValueTyped,
      maxValueTyped,
      reinterpret_cast<IN*>(minValueDevicePtr),
      numBitsDevicePtr,
      inputTyped,
      numDevice,
      maxNum);

  bitPackFixedBitAndMinInternal<IN, OUT, LIMIT>(
      minValueDevicePtr,
      numBitsDevicePtr,
      workspace,
      outPtr,
      in,
      numDevice,
      maxNum);
}


void compress(
    void* const workspace,
    const size_t workspaceSize,
    const nvcompType_t inType,
    void* const outPtr,
    const void* const in,
    const size_t* const numPtr,
    const size_t maxNum,
    void* const minValueDevicePtr,
    unsigned char* const numBitsDevicePtr)
{
  const size_t reqWorkSize = requiredWorkspaceSize(maxNum, inType);
  if (workspaceSize < reqWorkSize) {
    throw std::runtime_error(
        "Insufficient workspace size: " + std::to_string(workspaceSize)
        + ", need " + std::to_string(reqWorkSize));
  }

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < 1000; n++)
    NVCOMP_TYPE_SWITCH(
      inType,
      bitPackInternal,
      workspace,
      outPtr,
      in,
      numPtr,
      maxNum,
      minValueDevicePtr,
      numBitsDevicePtr);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (1000 iterations) = %f (s)\n", time * 1e-9f);
}

