/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Simple kernel demonstrating atomic functions in device code. */

#ifndef _SIMPLEATOMICS_KERNEL_H_
#define _SIMPLEATOMICS_KERNEL_H_

template <class T>
__global__ void testKernel(T *g_odata)
{
    // access thread id
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Atomic addition
    atomicAdd(&g_odata[0], (T)10);

    // Atomic subtraction (final should be 0)
    atomicSub(&g_odata[1], (T)10);

    // Atomic maximum
    atomicMax(&g_odata[2], (T)tid);

    // Atomic minimum
    atomicMin(&g_odata[3], (T)tid);

    // Atomic AND
    atomicAnd(&g_odata[4], (T)(2*tid+7));

    // Atomic OR
    atomicOr(&g_odata[5], (T)(1 << tid));

    // Atomic XOR
    atomicXor(&g_odata[6], (T)tid);

    // Atomic increment (modulo 17+1)
    atomicInc((unsigned int*)&g_odata[7], 17);

    // Atomic decrement
    atomicDec((unsigned int*)&g_odata[8], 137);
}

#endif // #ifndef _SIMPLEATOMICS_KERNEL_H_
