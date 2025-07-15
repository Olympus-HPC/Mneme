#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

__device__
float sigmoid (float x) {
  return (1.f / (1.f + expf(-x)));
}

__global__
void parallelPitched2DAccess (float* devPtr, size_t pitch, int width, int height)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < height && c < width) {
    float* row = (float*)((char*)devPtr + r * pitch);
    row[c] = sigmoid(row[c]);
  }
}

__global__
void parallelSimple2DAccess (float* elem, int width, int height)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < height && c < width) {
    elem[r * width + c] = sigmoid(elem[r * width + c]);
  }
}

__global__
void parallelPitched3DAccess (cudaPitchedPtr devPitchedPtr, int width, int height, int depth)
{
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (z < depth && y < height && x < width) {
    char* devPtr = (char*)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;
    char* slice = devPtr + z * slicePitch;
    float* row = (float*)(slice + y * pitch);
    row[x] = sigmoid(row[x]);
  }
}

__global__
void parallelSimple3DAccess (float* elem, int width, int height, int depth)
{
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (z < depth && y < height && x < width) {
    float element = elem[z * height * width + y * width + x];
    elem[z * height * width + y * width + x] = sigmoid(element);
  }
}

// Host code
void malloc2D (int repeat, int width, int height) {
  printf("Dimension: (%d %d)\n", width, height);

  dim3 grid ((width + 15)/16, (height + 15)/16);
  dim3 block (16, 16);

  float* devPtr;
  size_t pitch;
  cudaMallocPitch((void**)&devPtr, &pitch, width * sizeof(float), height);

  parallelPitched2DAccess<<<grid, block>>>(devPtr, pitch, width, height);
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    parallelPitched2DAccess<<<grid, block>>>(devPtr, pitch, width, height);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaFree(devPtr);

  cudaMalloc((void**)&devPtr, width * height * sizeof(float));

  parallelSimple2DAccess<<<grid, block>>>(devPtr, width, height);
  cudaDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    parallelSimple2DAccess<<<grid, block>>>(devPtr, width, height);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time (pitched vs simple): %f %f (us)\n",
          (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);

  cudaFree(devPtr);
}


// Host code
void malloc3D (int repeat, int width, int height, int depth) {
  printf("Dimension: (%d %d %d)\n", width, height, depth);
  dim3 grid ((width + 15)/16, (height + 7)/8, (depth + 3)/4);
  dim3 block (16, 8, 4);

  cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
  cudaPitchedPtr devPitchedPtr;
  cudaMalloc3D(&devPitchedPtr, extent);

  parallelPitched3DAccess<<<grid, block>>>(devPitchedPtr, width, height, depth);
  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    parallelPitched3DAccess<<<grid, block>>>(devPitchedPtr, width, height, depth);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaFree(devPitchedPtr.ptr);

  float* devPtr;
  cudaMalloc(&devPtr, width * height * depth * sizeof(float));

  parallelSimple3DAccess<<<grid, block>>>(devPtr, width, height, depth);
  cudaDeviceSynchronize();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    parallelSimple3DAccess<<<grid, block>>>(devPtr, width, height, depth);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time (pitched vs simple): %f %f (us)\n",
          (time * 1e-3f) / repeat, (time2 * 1e-3f) / repeat);

  cudaFree(devPtr);
}


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

  // width, height and depth
  const int w[] = {227, 256, 720, 768, 854, 1280, 1440, 1920, 2048, 3840, 4096};
  const int h[] = {227, 256, 480, 576, 480, 720, 1080, 1080, 1080, 2160, 2160};
  const int d[] = {1, 3};

  for (int i = 0; i < 11; i++)
    malloc2D(repeat, w[i], h[i]);

  for (int i = 0; i < 11; i++)
    for (int j = 0; j < 2; j++)
      malloc3D(repeat, w[i], h[i], d[j]);

  return 0;
}
