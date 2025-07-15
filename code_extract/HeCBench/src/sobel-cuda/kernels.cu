/**********************************************************************
  Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ********************************************************************/

/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */


inline __device__ float4 convert_float4(uchar4 data) 
{
   float4 r = make_float4(data.x, data.y, data.z, data.w);
   return r;
}

inline __device__ uchar4 convert_uchar4(float4 v) {
  uchar4 res;
  res.x = (uchar) ((v.x > 255.f) ? 255.f : (v.x < 0.f ? 0.f : v.x));
  res.y = (uchar) ((v.y > 255.f) ? 255.f : (v.y < 0.f ? 0.f : v.y));
  res.z = (uchar) ((v.z > 255.f) ? 255.f : (v.z < 0.f ? 0.f : v.z));
  res.w = (uchar) ((v.w > 255.f) ? 255.f : (v.w < 0.f ? 0.f : v.w));
  return res;
}

inline __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__global__
void sobel_filter(const uchar4*__restrict__ inputImage, 
                        uchar4*__restrict__ outputImage, 
                  const uint width,
                  const uint height)
{
  uint x = blockDim.x * blockIdx.x + threadIdx.x;
  uint y = blockDim.y * blockIdx.y + threadIdx.y;

  /* Read each texel component and calculate the filtered value using neighbouring texel components */
  if( x >= 1 && x < (width-1) && y >= 1 && y < height - 1)
  {
    int c = x + y * width;
    float4 i00 = convert_float4(inputImage[c - 1 - width]);
    float4 i01 = convert_float4(inputImage[c - width]);
    float4 i02 = convert_float4(inputImage[c + 1 - width]);

    float4 i10 = convert_float4(inputImage[c - 1]);
    float4 i12 = convert_float4(inputImage[c + 1]);

    float4 i20 = convert_float4(inputImage[c - 1 + width]);
    float4 i21 = convert_float4(inputImage[c + width]);
    float4 i22 = convert_float4(inputImage[c + 1 + width]);

    const float4 two = make_float4(2.f, 2.f, 2.f, 2.f);

    float4 Gx = i00 + two * i10 + i20 - i02 - two * i12 - i22;

    float4 Gy = i00 - i20  + two * i01 - two * i21 + i02 - i22;

    /* taking root of sums of squares of Gx and Gy */
    outputImage[c] = convert_uchar4(make_float4(sqrtf(Gx.x*Gx.x + Gy.x*Gy.x)/2.f,
                                                sqrtf(Gx.y*Gx.y + Gy.y*Gy.y)/2.f,
                                                sqrtf(Gx.z*Gx.z + Gy.z*Gy.z)/2.f,
                                                sqrtf(Gx.w*Gx.w + Gy.w*Gy.w)/2.f));
  }
}
