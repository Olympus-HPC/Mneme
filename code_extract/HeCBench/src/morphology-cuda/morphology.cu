#include "morphology.h"

enum class MorphOpType {
  ERODE,
  DILATE,
};

template <MorphOpType opType>
inline __device__ unsigned char elementOp(unsigned char lhs, unsigned char rhs)
{
}

template <>
inline __device__ unsigned char elementOp<MorphOpType::ERODE>(unsigned char lhs, unsigned char rhs)
{
  return min(lhs, rhs);
}

template <>
inline __device__ unsigned char elementOp<MorphOpType::DILATE>(unsigned char lhs, unsigned char rhs)
{
  return max(lhs, rhs);
}

template <MorphOpType opType>
inline __device__ unsigned char borderValue()
{
}

template <>
inline __device__ unsigned char borderValue<MorphOpType::ERODE>()
{
  return BLACK;
}

template <>
inline __device__ unsigned char borderValue<MorphOpType::DILATE>()
{
  return WHITE;
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void reversedScan(
    const unsigned char* __restrict__ buffer,
          unsigned char* __restrict__ opArray,
    const int selSize,
    const int tid)
{
  opArray[tid] = buffer[tid];
  __syncthreads();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (tid <= selSize - 1 - offset) {
      opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
    }
    __syncthreads();
  }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void scan(
    const unsigned char* __restrict__ buffer,
          unsigned char* __restrict__ opArray,
    const int selSize,
    const int tid)
{
  opArray[tid] = buffer[tid];
  __syncthreads();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (tid >= offset) {
      opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid - offset]);
    }
    __syncthreads();
  }
}

// NOTE: step-efficient parallel scan
template <MorphOpType opType>
__device__ void twoWayScan(
    const unsigned char* __restrict__ buffer,
          unsigned char* __restrict__ opArray,
    const int selSize,
    const int tid)
{
  opArray[tid] = buffer[tid];
  opArray[tid + selSize] = buffer[tid + selSize];
  __syncthreads();

  for (int offset = 1; offset < selSize; offset *= 2) {
    if (tid >= offset) {
      opArray[tid + selSize - 1] = 
        elementOp<opType>(opArray[tid + selSize - 1], opArray[tid + selSize - 1 - offset]);
    }
    if (tid <= selSize - 1 - offset) {
      opArray[tid] = elementOp<opType>(opArray[tid], opArray[tid + offset]);
    }
    __syncthreads();
  }
}

template <MorphOpType opType>
__global__ void vhgw_horiz(
          unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const int width,
    const int height,
    const int selSize
    )
{
  extern __shared__ unsigned char sMem[];
  unsigned char* buffer = sMem;
  unsigned char* opArray = buffer + 2 * selSize;

  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

  if (tidx >= width || tidy >= height) return;

  buffer[threadIdx.x] = src[tidy * width + tidx];
  if (tidx + selSize < width) {
    buffer[threadIdx.x + selSize] = src[tidy * width + tidx + selSize];
  }
  __syncthreads();

  twoWayScan<opType>(buffer, opArray, selSize, threadIdx.x);

  if (tidx + selSize/2 < width - selSize/2) {
    dst[tidy * width + tidx + selSize/2] = 
      elementOp<opType>(opArray[threadIdx.x], opArray[threadIdx.x + selSize - 1]);
  }
}

template <MorphOpType opType>
__global__ void vhgw_vert(
          unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const int width,
    const int height,
    const int selSize)
{
  extern __shared__ unsigned char sMem[];
  unsigned char* buffer = sMem;
  unsigned char* opArray = buffer + 2 * selSize;

  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  if (tidy >= height || tidx >= width) {
    return;
  }

  buffer[threadIdx.y] = src[tidy * width + tidx];
  if (tidy + selSize < height) {
    buffer[threadIdx.y + selSize] = src[(tidy + selSize) * width + tidx];
  }
  __syncthreads();

  twoWayScan<opType>(buffer, opArray, selSize, threadIdx.y);

  if (tidy + selSize/2 < height - selSize/2) {
    dst[(tidy + selSize/2) * width + tidx] = 
      elementOp<opType>(opArray[threadIdx.y], opArray[threadIdx.y + selSize - 1]);
  }

  if (tidy < selSize/2 || tidy >= height - selSize/2) {
    dst[tidy * width + tidx] = borderValue<opType>();
  }
}

template <MorphOpType opType>
double morphology(
    unsigned char* img_d,
    unsigned char* tmp_d,
    const int width,
    const int height,
    const int hsize,
    const int vsize)
{
  unsigned int memSize = width * height * sizeof(unsigned char);
  dim3 blockSize_h;
  dim3 gridSize_h;
  dim3 blockSize_v;
  dim3 gridSize_v;

  cudaMemset(tmp_d, 0, memSize);

  blockSize_h.x = hsize;
  blockSize_h.y = 1;
  gridSize_h.x = roundUp(width, blockSize_h.x);
  gridSize_h.y = roundUp(height, blockSize_h.y);
  size_t sMemSize_h = 4 * hsize * sizeof(unsigned char);

  blockSize_v.x = 1;
  blockSize_v.y = vsize;
  gridSize_v.x = roundUp(width, blockSize_v.x);
  gridSize_v.y = roundUp(height, blockSize_v.y);
  size_t sMemSize_v = 4 * vsize * sizeof(unsigned char);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  vhgw_horiz<opType><<<gridSize_h, blockSize_h, sMemSize_h>>>(tmp_d, img_d, width, height, hsize);

  vhgw_vert<opType><<<gridSize_v, blockSize_v, sMemSize_v>>>(img_d, tmp_d, width, height, vsize);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return time;
}

extern "C"
double erode(unsigned char* img_d,
             unsigned char* tmp_d,
             const int width,
             const int height,
             const int hsize,
             const int vsize)
{
  return morphology<MorphOpType::ERODE>(img_d, tmp_d, width, height, hsize, vsize);
}

extern "C"
double dilate(unsigned char* img_d,
              unsigned char* tmp_d,
              const int width,
              const int height,
              const int hsize,
              const int vsize)
{
  return morphology<MorphOpType::DILATE>(img_d, tmp_d, width, height, hsize, vsize);
}
