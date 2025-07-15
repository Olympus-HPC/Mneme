//
// CUDA implementation of FIM (Fast Iterative Method) for Eikonal equations
//
// Copyright (c) Won-Ki Jeong (wkjeong@unist.ac.kr)
//
// 2016. 2. 4
//

#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <cstdio>
#include "common_def.h"

#define MEM(index) _mem[index]
#define SOL(i,j,k) _sol[i][j][k]
#define SPD(i,j,k) _spd[i][j][k]

__device__ DOUBLE get_time_eikonal(DOUBLE a, DOUBLE b, DOUBLE c, DOUBLE s);
//
// F : Input speed (positive)
// if F =< 0, skip that pixel (masking out)
//
__global__ void run_solver(
  const double*__restrict__ spd,
  const bool*__restrict__ mask,
  const DOUBLE *__restrict__ sol_in,
  DOUBLE *__restrict__ sol_out,
  bool *__restrict__ con,
  const uint*__restrict__ list,
  int xdim, int ydim, int zdim,
  int nIter, uint nActiveBlock);
//
// run_reduction
//
// con is pixelwise convergence. Do reduction on active tiles and write tile-wise
// convergence to listVol. The implementation assumes that the block size is 4x4x4.
//
__global__ void run_reduction(
  const bool *__restrict__ con,
  bool *__restrict__ listVol,
  const uint *__restrict__ list,
  uint nActiveBlock);
//
// if block is active block, copy values
// if block is neighbor, run solver once
//
__global__ void run_check_neighbor(
  const double*__restrict__ spd,
  const bool*__restrict__ mask,
  const DOUBLE *__restrict__ sol_in,
  DOUBLE *__restrict__ sol_out,
  bool *__restrict__ con,
  const uint*__restrict__ list,
  int xdim, int ydim, int zdim,
  uint nActiveBlock, uint nTotalBlock);

#endif

