#ifndef __UTIL_H__
#define __UTIL_H__

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

float* get_parameter(const char* filename, int size);

float* transpose(float* weight, int h, int w);

void output_checker(float* A, float* B, int len, int channel, int shift);

#endif
