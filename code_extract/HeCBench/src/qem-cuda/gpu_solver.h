#ifndef GPU_SOLVER_CUH
#define GPU_SOLVER_CUH

#include <cuda_runtime.h>

cudaError_t checkCuda(cudaError_t result);

__global__ void cubicSolver(int n, float *A, float *B, float *C, float *D,
                            float *Q, float *R, float *del, float *theta,
                            float *sqrtQ, float *x1, float *x2, float *x3,
                            float *x1_img, float *x2_img, float *x3_img);

__global__ void QRdel(int n, float *A, float *B, float *C, float *D, float *b,
                      float *c, float *d, float *Q, float *R, float *Qint,
                      float *Rint, float *del);

__global__ void QuarticSolver(int n, float *A, float *B, float *C, float *D,
                              float *b, float *Q, float *R, float *del,
                              float *theta, float *sqrtQ, float *x1, float *x2,
                              float *x3, float *temp, float *min);

__global__ void QuarticSolver_full(int n, float *A, float *B, float *C,
                                   float *D, float *b, float *c, float *d,
                                   float *Q, float *R, float *del, float *theta,
                                   float *sqrtQ, float *x1, float *x2,
                                   float *x3, float *temp, float *min);

void QuarticMinimumGPU(int N, float *A, float *B, float *C, float *D, float *E,
                       float *min);

void QuarticMinimumGPUStreams(int N, float *A, float *B, float *C, float *D,
                              float *E, float *min);

#endif
