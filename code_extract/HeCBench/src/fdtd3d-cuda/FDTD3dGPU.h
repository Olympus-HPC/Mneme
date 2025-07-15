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

#ifndef _FDTD3DGPU_H_
#define _FDTD3DGPU_H_

#include <cstddef>
#if defined(_WIN32) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

// default value used for global memory  
#define MEMORY_SIZE 134217728

// default values used for shared memory  
#define k_blockDimMaxY 16
#define k_blockDimMaxX 32


// The values are set to give reasonable performance, they can be changed
// but note that setting an excessively large work group size can result in
// build failure due to insufficient local memory, even though it would be
// clamped before execution. This is because the maximum work group size
// cannot be determined before the build.
#define k_localWorkX    32
#define k_localWorkY    8
#define k_localWorkMin  128
#define k_localWorkMax  1024

// The values are set to give reasonable runtimes, they can
// be changed but note that running very large dimensions can
// take a very long time and you should avoid running on your
// primary display in this case.
#define k_dim_min           96
#define k_dim_max           376
#define k_dim_qa            248

// Note that the maximum radius is defined here as 4 since the
// minimum work group height is 4, if you have a larger work
// group then you can increase the radius accordingly.
#define k_radius_min        2
#define k_radius_max        4
#define k_radius_default    4

// The values are set to give reasonable runtimes, they can
// be changed but note that running a very large number of
// timesteps can take a very long time and you should avoid
// running on your primary display in this case.
#define k_timesteps_min     1
#define k_timesteps_max     10
#define k_timesteps_default 5

bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv);

#endif
