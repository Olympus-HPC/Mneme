#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ 
void rotate (const int n, const float angle, const float3 w, float3 *d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float s, c;
  sincosf(angle, &s,&c);
  
  const float3 p = d[i];
  const float mc = 1.f - c;

  // Rodrigues' formula:
  float m1 = c+(w.x)*(w.x)*(mc);
  float m2 = (w.z)*s+(w.x)*(w.y)*(mc);
  float m3 =-(w.y)*s+(w.x)*(w.z)*(mc);
  
  float m4 =-(w.z)*s+(w.x)*(w.y)*(mc);
  float m5 = c+(w.y)*(w.y)*(mc);
  float m6 = (w.x)*s+(w.y)*(w.z)*(mc);
  
  float m7 = (w.y)*s+(w.x)*(w.z)*(mc);
  float m8 =-(w.x)*s+(w.y)*(w.z)*(mc);
  float m9 = c+(w.z)*(w.z)*(mc);

  float ox = p.x*m1 + p.y*m2 + p.z*m3;
  float oy = p.x*m4 + p.y*m5 + p.z*m6;
  float oz = p.x*m7 + p.y*m8 + p.z*m9;
  d[i] = {ox, oy, oz};
}

__global__ 
void rotate2 (const int n, const float angle, const float3 w, float4 *d)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float s, c;
  sincosf(angle, &s,&c);
  
  const float4 p = d[i];
  const float mc = 1.f - c;

  // Rodrigues' formula:
  float m1 = c+(w.x)*(w.x)*(mc);
  float m2 = (w.z)*s+(w.x)*(w.y)*(mc);
  float m3 =-(w.y)*s+(w.x)*(w.z)*(mc);
  
  float m4 =-(w.z)*s+(w.x)*(w.y)*(mc);
  float m5 = c+(w.y)*(w.y)*(mc);
  float m6 = (w.x)*s+(w.y)*(w.z)*(mc);
  
  float m7 = (w.y)*s+(w.x)*(w.z)*(mc);
  float m8 =-(w.x)*s+(w.y)*(w.z)*(mc);
  float m9 = c+(w.z)*(w.z)*(mc);

  float ox = p.x*m1 + p.y*m2 + p.z*m3;
  float oy = p.x*m4 + p.y*m5 + p.z*m6;
  float oz = p.x*m7 + p.y*m8 + p.z*m9;
  d[i] = {ox, oy, oz, 0.f};
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
    
  // axis of rotation
  const float wx = -0.3, wy = -0.6, wz = 0.15;
  const float norm = 1.f / sqrtf(wx*wx + wy*wy + wz*wz);
  const float3 w = make_float3(wx*norm, wy*norm, wz*norm);

  const float angle = 0.5f;

  float3 *h = (float3*) malloc (sizeof(float3) * n);
  float4 *h2 = (float4*) malloc (sizeof(float4) * n);

  srand(123);
  for (int i = 0; i < n; i++) {
    float a = rand();
    float b = rand();
    float c = rand();
    float d = sqrtf(a*a + b*b + c*c);
    h[i] = make_float3(a/d, b/d, c/d);
    h2[i] = make_float4(a/d, b/d, c/d, 0.f);
  }

  dim3 grids ((n + 255) / 256);
  dim3 blocks (256);
 
  float3 *d;
  cudaMalloc((void**)&d, sizeof(float3) * n);
  cudaMemcpy(d, h, sizeof(float3) * n, cudaMemcpyHostToDevice);

  float4 *d2;
  cudaMalloc((void**)&d2, sizeof(float4) * n);
  cudaMemcpy(d2, h2, sizeof(float4) * n, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    rotate <<<grids, blocks>>> (n, angle, w, d);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (float3): %f (us)\n", (time * 1e-3f) / repeat);

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    rotate2 <<<grids, blocks>>> (n, angle, w, d2);
  }

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (float4): %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(d);
  cudaFree(d2);
  free(h);
  free(h2);
  return 0;
}
