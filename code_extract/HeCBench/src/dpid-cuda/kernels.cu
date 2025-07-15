// Copyright (c) 2016 Nicolas Weber and Sandra C. Amend / GCC / TU-Darmstadt. All rights reserved. 
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include "shared.h"

#define THREADS 128
#define WSIZE 32
#define TSIZE (THREADS / WSIZE)

#define TX threadIdx.x
#define PX (blockIdx.x * TSIZE + (TX / WSIZE))
#define PY blockIdx.y
#define WTHREAD  (TX % WSIZE)

__device__ __forceinline__
void normalize(float4& var) {
  var.x /= var.w;
  var.y /= var.w;
  var.z /= var.w;
  var.w = 1.f;
}

__device__ __forceinline__
void add(float4& output, const uchar3& color, const float factor) {
  output.x += color.x * factor;  
  output.y += color.y * factor;  
  output.z += color.z * factor;  
  output.w += factor;
}

__device__ __forceinline__
void add(float4& output, const float4& color) {
  output.x += color.x;
  output.y += color.y;
  output.z += color.z;
  output.w += color.w;
}

__device__ __forceinline__
float lambda(const Params p, const float dist) {
  if(p.lambda == 0.f)
    return 1.f;
  else if(p.lambda == 1.f)
    return dist;
  return powf(dist, p.lambda);
}

__device__ __forceinline__
void operator+=(float4& output, const float4 value) {
  output.x += value.x;
  output.y += value.y;
  output.z += value.z;
  output.w += value.w;
}

struct Local {
  float sx, ex, sy, ey;
  uint32_t sxr, syr, exr, eyr, xCount, yCount, pixelCount;

  __device__ __forceinline__ Local(const Params& p) {
    sx      = fmaxf( PX    * p.pWidth, 0.f);
    ex      = fminf((PX+1) * p.pWidth, (float)p.iWidth);
    sy      = fmaxf( PY    * p.pHeight, 0.f);
    ey      = fminf((PY+1) * p.pHeight, (float)p.iHeight);

    sxr      = (uint32_t)floorf(sx);
    syr      = (uint32_t)floorf(sy);
    exr      = (uint32_t)ceilf(ex);
    eyr      = (uint32_t)ceilf(ey);
    xCount    = exr - sxr;
    yCount    = eyr - syr;
    pixelCount  = xCount * yCount;
  }
};

__device__ __forceinline__
float contribution(const Local& l, float f, const uint32_t x, const uint32_t y) {
  if(x < l.sx)    f *= 1.f - (l.sx - x);
  if((x+1.f) > l.ex)  f *= 1.f - ((x+1.f) - l.ex);
  if(y < l.sy)    f *= 1.f - (l.sy - y);
  if((y+1.f) > l.ey)  f *= 1.f - ((y+1.f) - l.ey);
  return f;
}

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ __forceinline__
float4 __shfl_down(const float4 var, const uint32_t srcLane, const uint32_t width = 32) {
  float4 output;
  output.x = __shfl_down_sync(0xFFFFFFFF, var.x, srcLane, width);
  output.y = __shfl_down_sync(0xFFFFFFFF, var.y, srcLane, width);
  output.z = __shfl_down_sync(0xFFFFFFFF, var.z, srcLane, width);
  output.w = __shfl_down_sync(0xFFFFFFFF, var.w, srcLane, width);
  return output;
}

__device__ __forceinline__
void reduce(float4& value) {
  value += __shfl_down(value, 16);
  value += __shfl_down(value, 8);
  value += __shfl_down(value, 4);
  value += __shfl_down(value, 2);
  value += __shfl_down(value, 1);
}

__device__ __forceinline__
float distance(const float4& avg, const uchar3& color) {
  const float x = avg.x - color.x;
  const float y = avg.y - color.y;
  const float z = avg.z - color.z;
  return sqrtf(x * x + y * y + z * z) / 441.6729559f; // L2-Norm / sqrt(255^2 * 3)
}

__global__
void kernelGuidance(const uchar3* __restrict__ input,
                          uchar3* __restrict__ patches, const Params p)
{
  if(PX >= p.oWidth || PY >= p.oHeight) return;

  // init
  const Local l(p);
  float4 color = make_float4(0.f, 0.f, 0.f, 0.f);

  // iterate pixels
  for(uint32_t i = WTHREAD; i < l.pixelCount; i += WSIZE) {
    const uint32_t x = l.sxr + (i % l.xCount);
    const uint32_t y = l.syr + (i / l.xCount);

    float f = contribution(l, 1.f, x, y);  

    const uchar3& pixel = input[x + y * p.iWidth];
    add(color, make_float4(pixel.x * f, pixel.y * f, pixel.z * f, f));
  }

  // reduce warps
  reduce(color);

  // store results
  if((TX % 32) == 0) {
    normalize(color);
    patches[PX + PY * p.oWidth] = make_uchar3(color.x, color.y, color.z);
  }
}

__device__ __forceinline__
float4 calcAverage(const Params& p, const uchar3* __restrict__ patches) {
  const float corner = 1.0;
  const float edge   = 2.0;
  const float center = 4.0;

  // calculate average color
  float4 avg = make_float4(0.f, 0.f, 0.f, 0.f);

  // TOP
  if(PY > 0) {
    if(PX > 0) 
      add(avg, patches[(PX - 1) + (PY - 1) * p.oWidth], corner);

    add(avg, patches[(PX) + (PY - 1) * p.oWidth], edge);

    if((PX+1) < p.oWidth)
      add(avg, patches[(PX + 1) + (PY - 1) * p.oWidth], corner);
  }

  // LEFT
  if(PX > 0) 
    add(avg, patches[(PX - 1) + (PY) * p.oWidth], edge);

  // CENTER
  add(avg, patches[(PX) + (PY) * p.oWidth], center);

  // RIGHT
  if((PX+1) < p.oWidth)
    add(avg, patches[(PX + 1) + (PY) * p.oWidth], edge);

  // BOTTOM
  if((PY+1) < p.oHeight) {
    if(PX > 0) 
      add(avg, patches[(PX - 1) + (PY + 1) * p.oWidth], corner);

    add(avg, patches[(PX) + (PY + 1) * p.oWidth], edge);

    if((PX+1) < p.oWidth)
      add(avg, patches[(PX + 1) + (PY + 1) * p.oWidth], corner);
  }

  normalize(avg);

  return avg;
}

__global__
void kernelDownsampling(const uchar3* __restrict__ input,
                        const uchar3* __restrict__ patches,
                        const Params p,
                              uchar3* __restrict__ output)
{
  if(PX >= p.oWidth || PY >= p.oHeight) return;

  // init
  const Local l(p);
  const float4 avg = calcAverage(p, patches);

  float4 color = make_float4(0.f, 0.f, 0.f, 0.f);

  // iterate pixels
  for(uint32_t i = WTHREAD; i < l.pixelCount; i += WSIZE) {
    const uint32_t x = l.sxr + (i % l.xCount);
    const uint32_t y = l.syr + (i / l.xCount);

    const uchar3& pixel = input[x + y * p.iWidth];
    float f = distance(avg, pixel);

    f = lambda(p, f);
    f = contribution(l, f, x, y);

    add(color, pixel, f);
  }

  // reduce warp
  reduce(color);

  if(WTHREAD == 0) {
    uchar3& ref = output[PX + PY * p.oWidth];

    if(color.w == 0.0f)
      ref = make_uchar3((unsigned char)avg.x, (unsigned char)avg.y, (unsigned char)avg.z);
    else {
      normalize(color);
      ref = make_uchar3((unsigned char)color.x, (unsigned char)color.y, (unsigned char)color.z);
    }
  }
}

void run(const Params& p, const void* hInput, void* hOutput) {
  const size_t sInput    = sizeof(uchar3) * p.iWidth * p.iHeight;
  const size_t sOutput  = sizeof(uchar3) * p.oWidth * p.oHeight;
  const size_t sGuidance  = sizeof(uchar3) * p.oWidth * p.oHeight;

  uchar3* dInput = 0, *dOutput = 0, *dGuidance = 0;

  cudaMalloc(&dInput, sInput);
  cudaMalloc(&dOutput, sOutput);
  cudaMalloc(&dGuidance, sGuidance);

  cudaMemcpy(dInput, hInput, sInput, cudaMemcpyHostToDevice);

  const dim3 threads(THREADS, 1, 1); // 4 warps, 1 warp per patch
  const dim3 blocks((uint32_t)std::ceil(p.oWidth / (float)TSIZE), p.oHeight, 1);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (uint32_t i = 0; i < p.repeat; i++) {
    kernelGuidance <<<blocks, threads>>> (dInput, dGuidance, p);
    kernelDownsampling <<<blocks, threads>>> (dInput, dGuidance, p, dOutput);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / p.repeat);

  cudaMemcpy(hOutput, dOutput, sOutput, cudaMemcpyDeviceToHost);

  cudaFree(dInput);
  cudaFree(dOutput);
  cudaFree(dGuidance);
}
