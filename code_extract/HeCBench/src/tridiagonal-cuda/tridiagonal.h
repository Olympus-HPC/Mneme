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

#ifndef COMMON_H
#define COMMON_H

#define NATIVE_DIVIDE   // use native divide in the kernels

#define TRANSPOSE_BLOCK_DIM    16
#define BLOCK_DIM TRANSPOSE_BLOCK_DIM
#define REORDER

//#define OUTPUT_RESULTS

#ifndef BENCH_ITERATIONS
#define BENCH_ITERATIONS  1
#endif


#endif
