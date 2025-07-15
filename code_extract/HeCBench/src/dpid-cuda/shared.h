#ifndef _SHARED
#define _SHARED

#include <cuda.h>

struct Params {
  uint32_t oWidth;
  uint32_t oHeight;
  uint32_t iWidth;
  uint32_t iHeight;
     float pWidth;
     float pHeight;
     float lambda;
  uint32_t repeat;
};

#endif
