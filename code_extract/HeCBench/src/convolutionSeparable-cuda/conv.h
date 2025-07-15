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

#ifndef CONV_H
#define CONV_H


#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);

void convolutionColumnHost(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
);


void convolutionRows(
    float *d_Dst,
    const float *d_Src,
    const float *d_Kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

void convolutionColumns(
    float *d_Dst,
    const float *d_Src,
    const float *d_Kernel,
    const int imageW,
    const int imageH,
    const int pitch
);

#endif
