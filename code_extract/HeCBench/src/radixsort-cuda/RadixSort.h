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
#ifndef _RADIXSORT_H_
#define _RADIXSORT_H_

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "Scan.h"

static const unsigned int WARP_SIZE = 32;
static const unsigned int bitStep = 4;
static const unsigned int CTA_SIZE = 128;

void radixSortKeys(unsigned int* d_keys, 
                   unsigned int* d_tempKeys, 
                   unsigned int* d_counters, 
                   unsigned int* d_blockOffsets, 
                   unsigned int* d_countersSum, 
                   unsigned int* d_buffer, 
                   const unsigned int numElements, 
                   const unsigned int keyBits, 
                   const unsigned int batchSize 
);

#endif
