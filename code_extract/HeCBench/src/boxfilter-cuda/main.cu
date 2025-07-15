/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <chrono>
#include <memory>
#include <iostream>
#include <cuda.h>
#include "shrUtils.h"
#include "helper_math.h"

extern void BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                          unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale );


const unsigned int RADIUS = 10;                   // initial radius of 2D box filter mask
const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(const uint a, const uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
__device__
float4 rgbaUintToFloat4(const unsigned int c)
{
  float4 rgba;
  rgba.x = c & 0xff;
  rgba.y = (c >> 8) & 0xff;
  rgba.z = (c >> 16) & 0xff;
  rgba.w = (c >> 24) & 0xff;
  return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba, const float fScale)
{
  unsigned int uiPackedPix = 0U;
  uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
  uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
  uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
  uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
  return uiPackedPix;
}

__global__ void row_kernel (
    const uchar4* __restrict__ ucSource, 
            uint* __restrict__ uiDest,
    const unsigned int uiWidth,
    const unsigned int uiHeight,
    const int iRadius,
    const int iRadiusAligned, 
    const float fScale, 
    const unsigned int uiNumOutputPix)
{
  extern __shared__ uchar4 uc4LocalData[];

  int lid = threadIdx.x;
  int gidx = blockIdx.x;
  int gidy = blockIdx.y; 

  int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
  int globalPosY = gidy;
  int iGlobalOffset = globalPosY * uiWidth + globalPosX;

  // Read global data into LMEM
  if (globalPosX >= 0 && globalPosX < uiWidth)
  {
    uc4LocalData[lid] = ucSource[iGlobalOffset];
  }
  else
    uc4LocalData[lid] = {0, 0, 0, 0}; 

  __syncthreads();

  if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) && 
      (lid < (iRadiusAligned + (int)uiNumOutputPix)))
  {
    // Init summation registers to zero
    float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

    // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
    int iOffsetX = lid - iRadius;
    int iLimit = iOffsetX + (2 * iRadius) + 1;
    for(; iOffsetX < iLimit; iOffsetX++)
    {
      f4Sum.x += uc4LocalData[iOffsetX].x;
      f4Sum.y += uc4LocalData[iOffsetX].y;
      f4Sum.z += uc4LocalData[iOffsetX].z;
      f4Sum.w += uc4LocalData[iOffsetX].w; 
    }

    // Use inline function to scale and convert registers to packed RGBA values in a uchar4, 
    // and write back out to GMEM
    uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
  }
}

__global__ void col_kernel (
    const uint* __restrict__ uiSource,
          uint* __restrict__ uiDest,
    const unsigned int uiWidth, 
    const unsigned int uiHeight, 
    const int iRadius, 
    const float fScale)
{
  size_t globalPosX = blockIdx.x * blockDim.x + threadIdx.x;
  const uint* uiInputImage = &uiSource[globalPosX];
  uint* uiOutputImage = &uiDest[globalPosX];

  float4 f4Sum;

  float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
  float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

  f4Sum = top_color *
          make_float4((float)iRadius, (float)iRadius, (float)iRadius, (float)iRadius); 
  for (int y = 0; y < iRadius + 1; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
  }
  uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);

  for(int y = 1; y < iRadius + 1; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
    f4Sum -= top_color;
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }

  for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
    f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }

  for (int y = uiHeight - iRadius; y < uiHeight; y++) 
  {
    f4Sum += bot_color;
    f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }
}

void BoxFilterGPU (uchar4* cmBufIn,
    unsigned int* cmBufTmp,
    unsigned int* cmBufOut,
    const unsigned int uiWidth, 
    const unsigned int uiHeight, 
    const int iRadius, 
    const float fScale,
    const int iCycles )
{
  const int szMaxWorkgroupSize = 256;
  const int iRadiusAligned = ((iRadius + 15)/16) * 16;  // 16
  unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup
  if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
    uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

  // Set global and local work sizes for row kernel // Workgroup padded left and right
  dim3 row_grid(DivUp((size_t)uiWidth, (size_t)uiNumOutputPix), uiHeight); 
  dim3 row_block((size_t)(iRadiusAligned + uiNumOutputPix + iRadius), 1);

  // Set global and local work sizes for column kernel
  dim3 col_grid(DivUp((size_t)uiWidth, 64));
  dim3 col_block(64);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iCycles; i++) {
    // Launch row kernel
    row_kernel<<<row_grid, row_block, sizeof(uchar4)*(iRadiusAligned+uiNumOutputPix+iRadius)>>> (
        cmBufIn, cmBufTmp, uiWidth, uiHeight, iRadius, iRadiusAligned, fScale, uiNumOutputPix);


    // Launch column kernel
    col_kernel<<<col_grid, col_block>>>(cmBufTmp, cmBufOut, uiWidth, uiHeight, iRadius, fScale);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / iCycles);
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage %s <PPM image> <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int uiImageWidth = 0;     // Image width
  unsigned int uiImageHeight = 0;    // Image height
  unsigned int* uiInput = NULL;      // Host buffer to hold input image data
  unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiDevOutput = NULL;      
  unsigned int* uiHostOutput = NULL;      

  shrLoadPPM4ub(argv[1], (unsigned char**)&uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, bpp = %u, Mask Radius = %u\n", 
      uiImageWidth, uiImageHeight, unsigned(sizeof(unsigned int) * 8), RADIUS);
  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff= uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)malloc(szBuffBytes);
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);
  uiHostOutput = (unsigned int*)malloc(szBuffBytes);

  uchar4* cmDevBufIn;
  unsigned int* cmDevBufTmp;
  unsigned int* cmDevBufOut;

  cudaMalloc((void**)&cmDevBufIn, szBuffBytes);
  cudaMalloc((void**)&cmDevBufTmp, szBuffBytes);
  cudaMalloc((void**)&cmDevBufOut, szBuffBytes);

  // Copy input data from host to device 
  cudaMemcpy(cmDevBufIn, uiInput, szBuffBytes, cudaMemcpyHostToDevice);

  const int iCycles = atoi(argv[2]);

  printf("Warmup..\n");
  BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut, 
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
  BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut, 
                uiImageWidth, uiImageHeight, RADIUS, SCALE, iCycles);

  // Copy output from device to host
  cudaMemcpy(uiDevOutput, cmDevBufOut, szBuffBytes, cudaMemcpyDeviceToHost);

  cudaFree(cmDevBufIn);
  cudaFree(cmDevBufTmp);
  cudaFree(cmDevBufOut);

  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

  // Verification 
  // The entire images do not match due to the difference between BoxFilterHostY and the column kernel )
  int error = 0;
  for (unsigned i = RADIUS * uiImageWidth; i < (uiImageHeight-RADIUS)*uiImageWidth; i++)
  {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%d %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  return 0;
}
