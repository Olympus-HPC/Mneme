#ifndef DISTORTION_H
#define DISTORTION_H

#include <cuda.h>

struct Properties{
  float K;
  float centerX;
  float centerY;
  int width;
  int height;
  float thresh;
  float xscale;
  float yscale;
  float xshift;
  float yshift;
};

float calc_shift(float x1, float x2, float cx, float k, float thresh);

void reference (
  const uchar3* src,
        uchar3* dst,
  const struct Properties* prop);

__global__ void barrel_distort (
  const uchar3*__restrict__ src,
        uchar3*__restrict__ dst,
  const struct Properties*__restrict__ prop);

#endif
