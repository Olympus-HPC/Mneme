
#define CLAMP_TO_EDGE 
#define MAC

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <chrono>
#include <cuda.h>
#include "helper_math.h"
#include "main.h"
#include "shrUtils.h"

// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
__device__
float4 rgbaUintToFloat4(const unsigned int uiPackedRGBA)
{
    float4 rgba;
    rgba.x = uiPackedRGBA & 0xff;
    rgba.y = (uiPackedRGBA >> 8) & 0xff;
    rgba.z = (uiPackedRGBA >> 16) & 0xff;
    rgba.w = (uiPackedRGBA >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba)
{
    unsigned int uiPackedRGBA = 0U;
    uiPackedRGBA |= 0x000000FF & (unsigned int)rgba.x;
    uiPackedRGBA |= 0x0000FF00 & (((unsigned int)rgba.y) << 8);
    uiPackedRGBA |= 0x00FF0000 & (((unsigned int)rgba.z) << 16);
    uiPackedRGBA |= 0xFF000000 & (((unsigned int)rgba.w) << 24);
    return uiPackedRGBA;
}

// Transpose kernel (see transpose SDK sample for details)
//*****************************************************************
__global__ 
void Transpose(const unsigned int* uiDataIn, 
                     unsigned int* uiDataOut, 
               const int iWidth, const int iHeight)
{
    // read the matrix tile into LMEM
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ unsigned int uiLocalBuff[16*17];

    if((xIndex < iWidth) && (yIndex < iHeight))
    {
        //uiLocalBuff[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = uiDataIn[(yIndex * iWidth) + xIndex];
        uiLocalBuff[threadIdx.y * (blockDim.x + 1) + threadIdx.x] = uiDataIn[(yIndex * iWidth) + xIndex];
    }

    // Synchronize the read into LMEM
    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = __mul24(blockIdx.y, blockDim.y) + threadIdx.x;
    yIndex = __mul24(blockIdx.x, blockDim.x) + threadIdx.y;
    if((xIndex < iHeight) && (yIndex < iWidth))
    {
        uiDataOut[(yIndex * iHeight) + xIndex] = 
         //uiLocalBuff[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)];
         uiLocalBuff[threadIdx.x * (blockDim.y + 1) + threadIdx.y];
    }
}

// 	simple 1st order recursive filter kernel 
//*****************************************************************
//    - processes one image column per thread
//      parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a  - blur parameter
//*****************************************************************
__global__ void SimpleRecursiveRGBA(
  const unsigned int* uiDataIn,
        unsigned int* uiDataOut,
  const int iWidth, const int iHeight, const float a)
{
    // compute X pixel location and check in-bounds
  unsigned int X = blockIdx.x * blockDim.x + threadIdx.x;
  if (X >= iWidth) return;
    
  // advance global pointers to correct column for this work item and x position
  uiDataIn += X;    
  uiDataOut += X;

  // start forward filter pass
  float4 yp = rgbaUintToFloat4(*uiDataIn);  // previous output
  for (int Y = 0; Y < iHeight; Y++) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);   
    *uiDataOut = rgbaFloat4ToUint(yc);
    yp = yc;
    uiDataIn += iWidth;     // move to next row
    uiDataOut += iWidth;    // move to next row
  }

  // reset global pointers to point to last element in column for this work item and x position
  uiDataIn -= iWidth;
  uiDataOut -= iWidth;

  // start reverse filter pass: ensures response is symmetrical
  yp = rgbaUintToFloat4(*uiDataIn);
  for (int Y = iHeight - 1; Y > -1; Y--) 
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * make_float4(a, a, a, a);
    *uiDataOut = rgbaFloat4ToUint((rgbaUintToFloat4(*uiDataOut) + yc) * 0.5f);
    yp = yc;
    uiDataIn -= iWidth;   // move to previous row
    uiDataOut -= iWidth;  // move to previous row
  }
}

// Recursive Gaussian filter 
//*****************************************************************
//  parameters:	
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data 
//      iWidth  - image width
//      iHeight  - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//
//      If used, CLAMP_TO_EDGE is passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
__global__ void RecursiveRGBA(
  const unsigned int* uiDataIn, 
  unsigned int* uiDataOut, 
  const int iWidth, const int iHeight, 
  const float a0, const float a1, 
  const float a2, const float a3, 
  const float b1, const float b2, 
  const float coefp, const float coefn)
{
    // compute X pixel location and check in-bounds
    //unsigned int X = mul24(get_group_id(0), get_local_size(0)) + get_local_id(0);
    unsigned int X = blockIdx.x * blockDim.x + threadIdx.x;
    if (X >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    uiDataIn += X;    
    uiDataOut += X;

    // start forward filter pass
    float4 xp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous input
    float4 yp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output
    float4 yb = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // previous output by 2

#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*uiDataIn); 
    yb = xp * make_float4(coefp,coefp,coefp,coefp); 
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
		*uiDataOut = rgbaFloat4ToUint(yc);
        xp = xc; 
        yb = yp; 
        yp = yc; 
        uiDataIn += iWidth;     // move to next row
        uiDataOut += iWidth;    // move to next row
    }

    // reset global pointers to point to last element in column for this work item and x position
    uiDataIn -= iWidth;
    uiDataOut -= iWidth;

    // start reverse filter pass: ensures response is symmetrical
    float4 xn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 xa = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 yn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 ya = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*uiDataIn);
    xa = xn; 
    yn = xn * make_float4(coefn,coefn,coefn,coefn); 
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y > -1; Y--) 
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
        *uiDataOut = rgbaFloat4ToUint(rgbaUintToFloat4(*uiDataOut) + yc);
        uiDataIn -= iWidth;   // move to previous row
        uiDataOut -= iWidth;  // move to previous row
    }
}

double GPUGaussianFilterRGBA(const unsigned int* uiInput,
                             unsigned int* uiOutput,
                             unsigned int* d_BufIn,
                             unsigned int* d_BufTmp,
                             unsigned int* d_BufOut,
                             const unsigned int uiImageWidth,
                             const unsigned int uiImageHeight, 
                             const GaussParms* pGP)
{
#if USE_SIMPLE_FILTER
  float ema = pGP->ema;
#else
  float a0 = pGP->a0;
  float a1 = pGP->a1;
  float a2 = pGP->a2;
  float a3 = pGP->a3;
  float b1 = pGP->b1;
  float b2 = pGP->b2;
  float coefp = pGP->coefp;
  float coefn = pGP->coefn;
#endif

  unsigned int szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
  cudaMemcpy(d_BufIn, uiInput, szBuffBytes, cudaMemcpyHostToDevice); 

  auto start = std::chrono::steady_clock::now();
  
  // const int iTransposeBlockDim = 16; // initial height and width dimension of 2D transpose workgroup 
  size_t szGaussLocalWork = 256;
  size_t szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth);
  dim3 g_grid (szGaussGlobalWork / szGaussLocalWork);
  dim3 g_block (szGaussLocalWork);
 
#if USE_SIMPLE_FILTER
  SimpleRecursiveRGBA<<<g_grid, g_block>>>(d_BufIn, d_BufTmp, uiImageWidth, uiImageHeight, ema);
#else
        RecursiveRGBA<<<g_grid, g_block>>>(d_BufIn, d_BufTmp, uiImageWidth, uiImageHeight, 
                    a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif

  size_t szTransposeGlobalWork[2];
  size_t szTransposeLocalWork[2] = {16, 16};
  // Launch transpose kernel in 1st direction
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight); 
  dim3 t1_grid (szTransposeGlobalWork[0] / szTransposeLocalWork[0], 
                szTransposeGlobalWork[1] / szTransposeLocalWork[1]);
  dim3 t1_block (szTransposeLocalWork[0], szTransposeLocalWork[1]);

  Transpose<<<t1_grid, t1_block>>>(d_BufTmp, d_BufOut, uiImageWidth, uiImageHeight);
    
  // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
  // note width and height parameters flipped due to transpose
  szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight); 

  dim3 g2_grid (szGaussGlobalWork / szGaussLocalWork);

#if USE_SIMPLE_FILTER
  SimpleRecursiveRGBA<<<g2_grid, g_block>>>(d_BufOut, d_BufTmp, uiImageHeight, uiImageWidth, ema);
#else
        RecursiveRGBA<<<g2_grid, g_block>>>(d_BufOut, d_BufTmp, uiImageHeight, uiImageWidth, 
                    a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif

  // Reset transpose global work dimensions and variable args 
  // note width and height parameters flipped due to 1st transpose
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth); 

  dim3 t2_grid (szTransposeGlobalWork[0] / szTransposeLocalWork[0] , 
                szTransposeGlobalWork[1] / szTransposeLocalWork[1]);
  //range<1> t2_lws (szTransposeLocalWork[1], szTransposeLobalWork[0]);
  // Launch transpose kernel in 2nd direction
  Transpose<<<t2_grid, t1_block>>>(d_BufTmp, d_BufOut, uiImageHeight, uiImageWidth);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaMemcpy(uiOutput, d_BufOut, szBuffBytes, cudaMemcpyDeviceToHost); 
  return time;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <path to image> <repeat>\n", argv[0]);
    return 1;
  }

  const float fSigma = 10.0f;         // filter sigma (blur factor)
  const int iOrder = 0;               // filter order
  unsigned int uiImageWidth = 1920;   // Image width
  unsigned int uiImageHeight = 1080;  // Image height
  unsigned int* uiInput = NULL;       // Host buffer to hold input image data
  unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

  shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
  const int iCycles = atoi(argv[2]);

  printf("Image Width = %i, Height = %i, bpp = %lu\n\n",
         uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

  // Allocate intermediate and output host image buffers
  unsigned int szBuff = uiImageWidth * uiImageHeight;
  unsigned int szBuffBytes = szBuff * sizeof (unsigned int);
  uiTemp = (unsigned int*)malloc(szBuffBytes);
  uiOutput = (unsigned int*)malloc(szBuffBytes);
  printf("Allocate Host Image Buffers...\n"); 

  // Allocate the source, intermediate and result buffer memory objects on the device GMEM
  unsigned int* d_BufIn;
  unsigned int* d_BufTmp;
  unsigned int* d_BufOut;
  cudaMalloc((void**)&d_BufIn, szBuffBytes);
  cudaMalloc((void**)&d_BufTmp, szBuffBytes);
  cudaMalloc((void**)&d_BufOut, szBuffBytes);

  // init filter coefficients
  PreProcessGaussParms (fSigma, iOrder, &GP);

  // Warmup call to assure OpenCL driver is awake
  GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp,
                        d_BufOut, uiImageWidth, uiImageHeight, &GP);

  // Start round-trip timer and process iCycles loops on the GPU
  printf("\nRunning GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
  double time = 0.0;

  for (int i = 0; i < iCycles; i++)
  {
     time += GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp,
                                   d_BufOut, uiImageWidth, uiImageHeight, &GP);
  }

  printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / iCycles);

  // Compute on host 
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  HostRecursiveGaussianRGBA(uiInput, uiTemp, uiGolden, uiImageWidth, uiImageHeight, &GP);

  printf("Comparing GPU Result to CPU Result...\n"); 
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.01f);
  printf("\nGPU Result %s CPU Result within tolerance...\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

  free(uiGolden);
  free(uiInput);
  free(uiTemp);
  free(uiOutput);
  cudaFree(d_BufIn);
  cudaFree(d_BufTmp);
  cudaFree(d_BufOut);
  return 0;
}
