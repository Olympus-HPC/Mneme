#include <iostream>
#include "mttkrp_cpu.h"


void MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt){

  int *curMode = new int [X.ndims];
  ITYPE R = Opt.R;

  for (int m = 0; m < X.ndims; ++m)
    curMode[m] = (m + Opt.mode) % X.ndims;

  ITYPE mode0 = curMode[0];
  ITYPE mode1 = curMode[1];
  ITYPE mode2 = curMode[2];

  for(ITYPE x=0; x<X.totNnz; ++x) {

    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];

    for(ITYPE r=0; r<R; ++r) {            
      tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
      U[mode0].vals[idx0 * R + r] += tmp_val;
    }
  }
  delete[] curMode;
}

void MTTKRP_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt){

  int *curMode = new int [X.ndims];
  ITYPE R = Opt.R;

  for (int m = 0; m < X.ndims; ++m)
    curMode[m] = (m + Opt.mode) % X.ndims;


  ITYPE mode0 = curMode[0];
  ITYPE mode1 = curMode[1];
  ITYPE mode2 = curMode[2];
  ITYPE mode3 = curMode[3];

  for(ITYPE x=0; x<X.totNnz; ++x) {

    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];
    ITYPE idx3 = X.inds[mode3][x];

    for(ITYPE r=0; r<R; ++r) {            
      tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r] * 
        U[mode3].vals[idx3 * R + r];
      U[mode0].vals[idx0 * R + r] += tmp_val;
    }
  }
  delete[] curMode;
}


