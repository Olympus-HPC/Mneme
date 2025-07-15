#include <cmath>
#include "distort.h"

float calc_shift(float x1, float x2, float cx, float k, float thresh)
{
  float x3 = x1 + (x2 - x1) * 0.5f;
  float result1 = x1 + ((x1 - cx) * k * ((x1 - cx) * (x1 - cx)));
  float result3 = x3 + ((x3 - cx) * k * ((x3 - cx) * (x3 - cx)));

  if(result1 > -thresh and result1 < thresh)
    return x1;
  if(result3 < 0)
  {
    return calc_shift(x3, x2, cx, k, thresh);
  }
  else
  {
    return calc_shift(x1, x3, cx, k, thresh);
  }
}

__host__ __device__
inline float getRadialX(float x, float y, const struct Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  float result = x + ((x - prop->centerX) * prop->K *
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
  return result;
}

__host__ __device__
inline float getRadialY(float x, float y, const struct Properties* prop)
{
  x = (x * prop->xscale + prop->xshift);
  y = (y * prop->yscale + prop->yshift);
  float result = y + ((y - prop->centerY) * prop->K * 
    ((x - prop->centerX) * (x - prop->centerX) + (y - prop->centerY) * (y - prop->centerY)));
  return result;
}

__host__ __device__
inline void sampleImageTest(const uchar3* src, float idx0, float idx1,
                            uchar3& result, const struct Properties* prop)
{
  // out-of-bound check
  if((idx0 < 0) || (idx1 < 0) || (idx0 > prop->height - 1) || (idx1 > prop->width - 1))
  {
    result.x = 0;
    result.y = 0;
    result.z = 0;
    return;
  }

  int idx0_floor = (int)floorf(idx0);
  int idx0_ceil = (int)ceilf(idx0);
  int idx1_floor = (int)floorf(idx1);
  int idx1_ceil = (int)ceilf(idx1);

  uchar3 s1 = src[(idx0_floor * prop->width) + idx1_floor];
  uchar3 s2 = src[(idx0_floor * prop->width) + idx1_ceil];
  uchar3 s3 = src[(idx0_ceil * prop->width) + idx1_ceil];
  uchar3 s4 = src[(idx0_ceil * prop->width) + idx1_floor];

  float x = idx0 - idx0_floor;
  float y = idx1 - idx1_floor;

  result.x = s1.x * (1.f - x) * (1.f - y) + s2.x * (1.f - x) * y + s3.x * x * y + s4.x * x * (1.f - y);
  result.y = s1.y * (1.f - x) * (1.f - y) + s2.y * (1.f - x) * y + s3.y * x * y + s4.y * x * (1.f - y);
  result.z = s1.z * (1.f - x) * (1.f - y) + s2.z * (1.f - x) * y + s3.z * x * y + s4.z * x * (1.f - y);
}

__global__ void barrel_distort (
  const uchar3 *__restrict__ src,
        uchar3 *__restrict__ dst,
  const struct Properties *__restrict__ prop)
{
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w < prop->width && h < prop->height) {
    float x = getRadialX((float)w, (float)h, prop);
    float y = getRadialY((float)w, (float)h, prop);
    uchar3 temp;
    sampleImageTest(src, y, x, temp, prop);
    dst[(h * prop->width) + w] = temp;
  }
}

void reference (
  const uchar3 *src,
        uchar3 *dst,
  const struct Properties *prop)
{
  for (int h = 0; h < prop->height; h++) {
    for (int w = 0; w < prop->width; w++) {
      float x = getRadialX((float)w, (float)h, prop);
      float y = getRadialY((float)w, (float)h, prop);
      uchar3 temp;
      sampleImageTest(src, y, x, temp, prop);
      dst[(h * prop->width) + w] = temp;
    }
  }
}
