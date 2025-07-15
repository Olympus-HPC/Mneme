#include "utils.h"

extern "C" void reference (double *flux_in_0, double *flux_in_1, double *flux_in_2, double *flux_in_3, double *flux_in_4, double *cons_in_1, double *cons_in_2, double *cons_in_3, double *cons_in_4, double *q_in_1, double *q_in_2, double *q_in_3, double *q_in_4, double *dxinv, int N) {
  double (*flux_0)[D][D] = (double (*)[D][D])flux_in_0;
  double (*flux_1)[D][D] = (double (*)[D][D])flux_in_1;
  double (*flux_2)[D][D] = (double (*)[D][D])flux_in_2;
  double (*flux_3)[D][D] = (double (*)[D][D])flux_in_3;
  double (*flux_4)[D][D] = (double (*)[D][D])flux_in_4;
  double (*cons_1)[D][D] = (double (*)[D][D])cons_in_1;
  double (*cons_2)[D][D] = (double (*)[D][D])cons_in_2;
  double (*cons_3)[D][D] = (double (*)[D][D])cons_in_3;
  double (*cons_4)[D][D] = (double (*)[D][D])cons_in_4;
  double (*q_1)[D][D] = (double (*)[D][D])q_in_1;
  double (*q_2)[D][D] = (double (*)[D][D])q_in_2;
  double (*q_3)[D][D] = (double (*)[D][D])q_in_3;
  double (*q_4)[D][D] = (double (*)[D][D])q_in_4;
  int i, j, k;

  for (k = 4; k < N-4; k++) {
    for (j = 4; j < N-4; j++) {
      for (i = 4; i < N-4; i++) {
        flux_0[k][j][i] = -((0.8f*(cons_1[k][j][i+1]-cons_1[k][j][i-1])-0.2f*(cons_1[k][j][i+2]-cons_1[k][j][i-2])+0.038f*(cons_1[k][j][i+3]-cons_1[k][j][i-3])-0.0035f*(cons_1[k][j][i+4]-cons_1[k][j][i-4]))*dxinv[0]);
        flux_0[k][j][i] -= (0.8f*(cons_2[k][j+1][i]-cons_2[k][j-1][i])-0.2f*(cons_2[k][j+2][i]-cons_2[k][j-2][i])+0.038f*(cons_2[k][j+3][i]-cons_2[k][j-3][i])-0.0035f*(cons_2[k][j+4][i]-cons_2[k][j-4][i]))*dxinv[1];
        flux_0[k][j][i] -= (0.8f*(cons_3[k+1][j][i]-cons_3[k-1][j][i])-0.2f*(cons_3[k+2][j][i]-cons_3[k-2][j][i])+0.038f*(cons_3[k+3][j][i]-cons_3[k-3][j][i])-0.0035f*(cons_3[k+4][j][i]-cons_3[k-4][j][i]))*dxinv[2];
      }
    } 
  }
  for (k = 4; k < N-4; k++) {
    for (j = 4; j < N-4; j++) {
      for (i = 4; i < N-4; i++) {
        flux_1[k][j][i] = -((0.8f*(cons_1[k][j][i+1]*q_1[k][j][i+1]-cons_1[k][j][i-1]*q_1[k][j][i-1]+(q_4[k][j][i+1]-q_4[k][j][i-1]))-0.2f*(cons_1[k][j][i+2]*q_1[k][j][i+2]-cons_1[k][j][i-2]*q_1[k][j][i-2]+(q_4[k][j][i+2]-q_4[k][j][i-2]))+0.038f*(cons_1[k][j][i+3]*q_1[k][j][i+3]-cons_1[k][j][i-3]*q_1[k][j][i-3]+(q_4[k][j][i+3]-q_4[k][j][i-3]))-0.0035f*(cons_1[k][j][i+4]*q_1[k][j][i+4]-cons_1[k][j][i-4]*q_1[k][j][i-4]+(q_4[k][j][i+4]-q_4[k][j][i-4])))*dxinv[0]);
  	flux_1[k][j][i] -= (0.8f*(cons_1[k][j+1][i]*q_2[k][j+1][i]-cons_1[k][j-1][i]*q_2[k][j-1][i])-0.2f*(cons_1[k][j+2][i]*q_2[k][j+2][i]-cons_1[k][j-2][i]*q_2[k][j-2][i])+0.038f*(cons_1[k][j+3][i]*q_2[k][j+3][i]-cons_1[k][j-3][i]*q_2[k][j-3][i])-0.0035f*(cons_1[k][j+4][i]*q_2[k][j+4][i]-cons_1[k][j-4][i]*q_2[k][j-4][i]))*dxinv[1];
        flux_1[k][j][i] -= (0.8f*(cons_1[k+1][j][i]*q_3[k+1][j][i]-cons_1[k-1][j][i]*q_3[k-1][j][i])-0.2f*(cons_1[k+2][j][i]*q_3[k+2][j][i]-cons_1[k-2][j][i]*q_3[k-2][j][i])+0.038f*(cons_1[k+3][j][i]*q_3[k+3][j][i]-cons_1[k-3][j][i]*q_3[k-3][j][i])-0.0035f*(cons_1[k+4][j][i]*q_3[k+4][j][i]-cons_1[k-4][j][i]*q_3[k-4][j][i]))*dxinv[2];
  	  }
    }
  } 
  for (k = 4; k < N-4; k++) {
    for (j = 4; j < N-4; j++) { 
      for (i = 4; i < N-4; i++) {
         flux_2[k][j][i] = -((0.8f*(cons_2[k][j][i+1]*q_1[k][j][i+1]-cons_2[k][j][i-1]*q_1[k][j][i-1])-0.2f*(cons_2[k][j][i+2]*q_1[k][j][i+2]-cons_2[k][j][i-2]*q_1[k][j][i-2])+0.038f*(cons_2[k][j][i+3]*q_1[k][j][i+3]-cons_2[k][j][i-3]*q_1[k][j][i-3])-0.0035f*(cons_2[k][j][i+4]*q_1[k][j][i+4]-cons_2[k][j][i-4]*q_1[k][j][i-4]))*dxinv[0]);
        flux_2[k][j][i] -= (0.8f*(cons_2[k][j+1][i]*q_2[k][j+1][i]-cons_2[k][j-1][i]*q_2[k][j-1][i]+(q_4[k][j+1][i]-q_4[k][j-1][i]))-0.2f*(cons_2[k][j+2][i]*q_2[k][j+2][i]-cons_2[k][j-2][i]*q_2[k][j-2][i]+(q_4[k][j+2][i]-q_4[k][j-2][i]))+0.038f*(cons_2[k][j+3][i]*q_2[k][j+3][i]-cons_2[k][j-3][i]*q_2[k][j-3][i]+(q_4[k][j+3][i]-q_4[k][j-3][i]))-0.0035f*(cons_2[k][j+4][i]*q_2[k][j+4][i]-cons_2[k][j-4][i]*q_2[k][j-4][i]+(q_4[k][j+4][i]-q_4[k][j-4][i])))*dxinv[1];
        flux_2[k][j][i] -= (0.8f*(cons_2[k+1][j][i]*q_3[k+1][j][i]-cons_2[k-1][j][i]*q_3[k-1][j][i])-0.2f*(cons_2[k+2][j][i]*q_3[k+2][j][i]-cons_2[k-2][j][i]*q_3[k-2][j][i])+0.038f*(cons_2[k+3][j][i]*q_3[k+3][j][i]-cons_2[k-3][j][i]*q_3[k-3][j][i])-0.0035f*(cons_2[k+4][j][i]*q_3[k+4][j][i]-cons_2[k-4][j][i]*q_3[k-4][j][i]))*dxinv[2];
      }
    }
  } 

  for (k = 4; k < N-4; k++) {
    for (j = 4; j < N-4; j++) {
      for (i = 4; i < N-4; i++) {
        flux_3[k][j][i] = -((0.8f*(cons_3[k][j][i+1]*q_1[k][j][i+1]-cons_3[k][j][i-1]*q_1[k][j][i-1])-0.2f*(cons_3[k][j][i+2]*q_1[k][j][i+2]-cons_3[k][j][i-2]*q_1[k][j][i-2])+0.038f*(cons_3[k][j][i+3]*q_1[k][j][i+3]-cons_3[k][j][i-3]*q_1[k][j][i-3])-0.0035f*(cons_3[k][j][i+4]*q_1[k][j][i+4]-cons_3[k][j][i-4]*q_1[k][j][i-4]))*dxinv[0]);
        flux_3[k][j][i] -= (0.8f*(cons_3[k][j+1][i]*q_2[k][j+1][i]-cons_3[k][j-1][i]*q_2[k][j-1][i])-0.2f*(cons_3[k][j+2][i]*q_2[k][j+2][i]-cons_3[k][j-2][i]*q_2[k][j-2][i])+0.038f*(cons_3[k][j+3][i]*q_2[k][j+3][i]-cons_3[k][j-3][i]*q_2[k][j-3][i])-0.0035f*(cons_3[k][j+4][i]*q_2[k][j+4][i]-cons_3[k][j-4][i]*q_2[k][j-4][i]))*dxinv[1];
        flux_3[k][j][i] -= (0.8f*(cons_3[k+1][j][i]*q_3[k+1][j][i]-cons_3[k-1][j][i]*q_3[k-1][j][i]+(q_4[k+1][j][i]-q_4[k-1][j][i]))-0.2f*(cons_3[k+2][j][i]*q_3[k+2][j][i]-cons_3[k-2][j][i]*q_3[k-2][j][i]+(q_4[k+2][j][i]-q_4[k-2][j][i]))+0.038f*(cons_3[k+3][j][i]*q_3[k+3][j][i]-cons_3[k-3][j][i]*q_3[k-3][j][i]+(q_4[k+3][j][i]-q_4[k-3][j][i]))-0.0035f*(cons_3[k+4][j][i]*q_3[k+4][j][i]-cons_3[k-4][j][i]*q_3[k-4][j][i]+(q_4[k+4][j][i]-q_4[k-4][j][i])))*dxinv[2];
      }
    }
  } 
  for (k = 4; k < N-4; k++) {
    for (j = 4; j < N-4; j++) { 
      for (i = 4; i < N-4; i++) {
        flux_4[k][j][i] = -((0.8f*(cons_4[k][j][i+1]*q_1[k][j][i+1]-cons_4[k][j][i-1]*q_1[k][j][i-1]+(q_4[k][j][i+1]*q_1[k][j][i+1]-q_4[k][j][i-1]*q_1[k][j][i-1]))-0.2f*(cons_4[k][j][i+2]*q_1[k][j][i+2]-cons_4[k][j][i-2]*q_1[k][j][i-2]+(q_4[k][j][i+2]*q_1[k][j][i+2]-q_4[k][j][i-2]*q_1[k][j][i-2]))+0.038f*(cons_4[k][j][i+3]*q_1[k][j][i+3]-cons_4[k][j][i-3]*q_1[k][j][i-3]+(q_4[k][j][i+3]*q_1[k][j][i+3]-q_4[k][j][i-3]*q_1[k][j][i-3]))-0.0035f*(cons_4[k][j][i+4]*q_1[k][j][i+4]-cons_4[k][j][i-4]*q_1[k][j][i-4]+(q_4[k][j][i+4]*q_1[k][j][i+4]-q_4[k][j][i-4]*q_1[k][j][i-4])))*dxinv[0]);
        flux_4[k][j][i] -= (0.8f*(cons_4[k+1][j][i]*q_3[k+1][j][i]-cons_4[k-1][j][i]*q_3[k-1][j][i]+(q_4[k+1][j][i]*q_3[k+1][j][i]-q_4[k-1][j][i]*q_3[k-1][j][i]))-0.2f*(cons_4[k+2][j][i]*q_3[k+2][j][i]-cons_4[k-2][j][i]*q_3[k-2][j][i]+(q_4[k+2][j][i]*q_3[k+2][j][i]-q_4[k-2][j][i]*q_3[k-2][j][i]))+0.038f*(cons_4[k+3][j][i]*q_3[k+3][j][i]-cons_4[k-3][j][i]*q_3[k-3][j][i]+(q_4[k+3][j][i]*q_3[k+3][j][i]-q_4[k-3][j][i]*q_3[k-3][j][i]))-0.0035f*(cons_4[k+4][j][i]*q_3[k+4][j][i]-cons_4[k-4][j][i]*q_3[k-4][j][i]+(q_4[k+4][j][i]*q_3[k+4][j][i]-q_4[k-4][j][i]*q_3[k-4][j][i])))*dxinv[2];
        flux_4[k][j][i] -= (0.8f*(cons_4[k][j+1][i]*q_2[k][j+1][i]-cons_4[k][j-1][i]*q_2[k][j-1][i]+(q_4[k][j+1][i]*q_2[k][j+1][i]-q_4[k][j-1][i]*q_2[k][j-1][i]))-0.2f*(cons_4[k][j+2][i]*q_2[k][j+2][i]-cons_4[k][j-2][i]*q_2[k][j-2][i]+(q_4[k][j+2][i]*q_2[k][j+2][i]-q_4[k][j-2][i]*q_2[k][j-2][i]))+0.038f*(cons_4[k][j+3][i]*q_2[k][j+3][i]-cons_4[k][j-3][i]*q_2[k][j-3][i]+(q_4[k][j+3][i]*q_2[k][j+3][i]-q_4[k][j-3][i]*q_2[k][j-3][i]))-0.0035f*(cons_4[k][j+4][i]*q_2[k][j+4][i]-cons_4[k][j-4][i]*q_2[k][j-4][i]+(q_4[k][j+4][i]*q_2[k][j+4][i]-q_4[k][j-4][i]*q_2[k][j-4][i])))*dxinv[1];
      }
    }
  }
}
