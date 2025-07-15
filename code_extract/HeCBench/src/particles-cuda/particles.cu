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

//Standard utilities and systems includes
#include <stdio.h>
#include <cuda.h>
#include "particles.h"
#include "particles_kernels.cu"

//Simulation parameters

static const size_t wgSize = 64;

static size_t uSnap(size_t a, size_t b){
  return ((a % b) == 0) ? a : (a - (a % b) + b);
}

void integrateSystem(
    float4* d_Pos,
    float4* d_Vel,
    const simParams_t &params,
    const float deltaTime,
    const unsigned int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);
  dim3 gws (globalWorkSize/wgSize);
  dim3 lws (wgSize);

  integrateSystemK<<<gws, lws>>>(d_Pos, d_Vel, params, deltaTime, numParticles);
}

void calcHash(
    unsigned int *d_Hash,
    unsigned int *d_Index,
    float4 *d_Pos,
    const simParams_t &params,
    const int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);
  dim3 gws (globalWorkSize/wgSize);
  dim3 lws (wgSize);

  calcHashK<<<gws, lws>>>(d_Hash, d_Index, d_Pos, params, numParticles);
}

void memSet(
    unsigned int* d_Data,
    unsigned int val,
    unsigned int N)
{
  size_t globalWorkSize = uSnap(N, wgSize);

  dim3 gws (globalWorkSize/wgSize);
  dim3 lws (wgSize);

  memSetK<<<gws, lws>>>(d_Data, val, N);
}

void findCellBoundsAndReorder(
    unsigned int* d_CellStart,
    unsigned int* d_CellEnd,
    float4 *d_ReorderedPos,
    float4 *d_ReorderedVel,
    unsigned int *d_Hash,
    unsigned int *d_Index,
    float4 *d_Pos,
    float4 *d_Vel,
    const unsigned int numParticles,
    const unsigned int numCells)
{
  memSet(d_CellStart, 0xFFFFFFFFU, numCells);

  size_t globalWorkSize = uSnap(numParticles, wgSize);
  dim3 gws (globalWorkSize/wgSize);
  dim3 lws (wgSize);

  findCellBoundsAndReorderK<<<gws, lws, (wgSize+1)*sizeof(unsigned int)>>>(
      d_CellStart,
      d_CellEnd, 
      d_ReorderedPos,
      d_ReorderedVel, 
      d_Hash, 
      d_Index, 
      d_Pos, 
      d_Vel,
      numParticles);
}

void collide(
    float4 *d_Vel,
    float4 *d_ReorderedPos,
    float4 *d_ReorderedVel,
    unsigned int *d_Index,
    unsigned int *d_CellStart,
    unsigned int *d_CellEnd,
    const simParams_t &params,
    const unsigned int   numParticles,
    const unsigned int   numCells)
{
  size_t globalWorkSize = uSnap(numParticles, wgSize);

  dim3 gws (globalWorkSize/wgSize);
  dim3 lws (wgSize);

  collideK<<<gws, lws>>>(d_Vel, d_ReorderedPos, d_ReorderedVel, 
      d_Index, d_CellStart, d_CellEnd, 
      params, numParticles);
}
