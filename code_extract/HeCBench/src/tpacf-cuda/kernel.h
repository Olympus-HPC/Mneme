#pragma once

#include <stdio.h>
#include <math.h>

#define NUMBINS 64

struct cartesian {
  double *x;
  double *y;
  double *z;
  int *jk;
};
