#include <stdio.h>
#include <chrono>
#include <cuda.h>

#define max(x,y)  ((x) > (y)? (x) : (y))
#define min(x,y)  ((x) < (y)? (x) : (y))
#define ceil(a,b) ((a) % (b) == 0 ? (a) / (b) : ((a) / (b)) + 1)

void check_error (const char* message) {
  cudaError_t error = cudaGetLastError ();
  if (error != cudaSuccess) {
    printf ("Error : %s, %s\n", message, cudaGetErrorString (error));
  }
}

__global__
void hypterm_1 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1])-0.2f*(cons_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2])+0.038f*(cons_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3])-0.0035f*(cons_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_1[k*M*N+j*N+i] = -((0.8f*(cons_1[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_1[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]))-0.2f*(cons_1[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_1[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]))+0.038f*(cons_1[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_1[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]))-0.0035f*(cons_1[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_1[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4])))*dxinv0);
  	 flux_2[k*M*N+j*N+i] = -((0.8f*(cons_2[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_2[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_2[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_2[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_2[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_2[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_2[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_2[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_3[k*M*N+j*N+i] = -((0.8f*(cons_3[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_3[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1])-0.2f*(cons_3[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_3[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2])+0.038f*(cons_3[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_3[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3])-0.0035f*(cons_3[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_3[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]))*dxinv0);
  	flux_4[k*M*N+j*N+i] = -((0.8f*(cons_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-cons_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]+(q_4[k*M*N+j*N+i+1]*q_1[k*M*N+j*N+i+1]-q_4[k*M*N+j*N+i-1]*q_1[k*M*N+j*N+i-1]))-0.2f*(cons_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-cons_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]+(q_4[k*M*N+j*N+i+2]*q_1[k*M*N+j*N+i+2]-q_4[k*M*N+j*N+i-2]*q_1[k*M*N+j*N+i-2]))+0.038f*(cons_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-cons_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]+(q_4[k*M*N+j*N+i+3]*q_1[k*M*N+j*N+i+3]-q_4[k*M*N+j*N+i-3]*q_1[k*M*N+j*N+i-3]))-0.0035f*(cons_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-cons_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4]+(q_4[k*M*N+j*N+i+4]*q_1[k*M*N+j*N+i+4]-q_4[k*M*N+j*N+i-4]*q_1[k*M*N+j*N+i-4])))*dxinv0);
  } 
}

__global__
void hypterm_2 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i])-0.2f*(cons_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i])+0.038f*(cons_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_1[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_1[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_1[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_1[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_1[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_1[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_1[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_2[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]))-0.2f*(cons_2[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_2[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]))+0.038f*(cons_2[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_2[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]))-0.0035f*(cons_2[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_2[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i])))*dxinv1;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_3[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i])-0.2f*(cons_3[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_3[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i])+0.038f*(cons_3[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_3[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i])-0.0035f*(cons_3[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_3[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]))*dxinv1;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]))-0.2f*(cons_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]))+0.038f*(cons_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]))-0.0035f*(cons_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i])))*dxinv2;
  } 
}


__global__
void hypterm_3 (double * __restrict__ flux_0,
                double * __restrict__ flux_1,
                double * __restrict__ flux_2,
                double * __restrict__ flux_3,
                double * __restrict__ flux_4,
                const double * __restrict__ cons_1,
                const double * __restrict__ cons_2,
                const double * __restrict__ cons_3,
                const double * __restrict__ cons_4, 
                const double * __restrict__ q_1,
                const double * __restrict__ q_2,
                const double * __restrict__ q_3,
                const double * __restrict__ q_4,
                double dxinv0, double dxinv1, double dxinv2,
                int L, int M, int N)
{
  //Determing the block's indices
  int blockdim_i= (int)(blockDim.x);
  int i0 = (int)(blockIdx.x)*(blockdim_i);
  int i = max (i0, 0) + (int)(threadIdx.x);
  int blockdim_j= (int)(blockDim.y);
  int j0 = (int)(blockIdx.y)*(blockdim_j);
  int j = max (j0, 0) + (int)(threadIdx.y);
  int blockdim_k= (int)(blockDim.z);
  int k0 = (int)(blockIdx.z)*(blockdim_k);
  int k = max (k0, 0) + (int)(threadIdx.z);

  if (i>=4 & j>=4 & k>=4 & i<=N-5 & j<=N-5 & k<=N-5) {
  	flux_0[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i])-0.2f*(cons_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i])+0.038f*(cons_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_1[k*M*N+j*N+i] -= (0.8f*(cons_1[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_1[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_1[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_1[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_1[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_1[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_1[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_1[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_2[k*M*N+j*N+i] -= (0.8f*(cons_2[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_2[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i])-0.2f*(cons_2[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_2[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i])+0.038f*(cons_2[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_2[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i])-0.0035f*(cons_2[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_2[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]))*dxinv2;
  	flux_3[k*M*N+j*N+i] -= (0.8f*(cons_3[(k+1)*M*N+j*N+i]*q_3[(k+1)*M*N+j*N+i]-cons_3[(k-1)*M*N+j*N+i]*q_3[(k-1)*M*N+j*N+i]+(q_4[(k+1)*M*N+j*N+i]-q_4[(k-1)*M*N+j*N+i]))-0.2f*(cons_3[(k+2)*M*N+j*N+i]*q_3[(k+2)*M*N+j*N+i]-cons_3[(k-2)*M*N+j*N+i]*q_3[(k-2)*M*N+j*N+i]+(q_4[(k+2)*M*N+j*N+i]-q_4[(k-2)*M*N+j*N+i]))+0.038f*(cons_3[(k+3)*M*N+j*N+i]*q_3[(k+3)*M*N+j*N+i]-cons_3[(k-3)*M*N+j*N+i]*q_3[(k-3)*M*N+j*N+i]+(q_4[(k+3)*M*N+j*N+i]-q_4[(k-3)*M*N+j*N+i]))-0.0035f*(cons_3[(k+4)*M*N+j*N+i]*q_3[(k+4)*M*N+j*N+i]-cons_3[(k-4)*M*N+j*N+i]*q_3[(k-4)*M*N+j*N+i]+(q_4[(k+4)*M*N+j*N+i]-q_4[(k-4)*M*N+j*N+i])))*dxinv2;
  	flux_4[k*M*N+j*N+i] -= (0.8f*(cons_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-cons_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]+(q_4[k*M*N+(j+1)*N+i]*q_2[k*M*N+(j+1)*N+i]-q_4[k*M*N+(j-1)*N+i]*q_2[k*M*N+(j-1)*N+i]))-0.2f*(cons_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-cons_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]+(q_4[k*M*N+(j+2)*N+i]*q_2[k*M*N+(j+2)*N+i]-q_4[k*M*N+(j-2)*N+i]*q_2[k*M*N+(j-2)*N+i]))+0.038f*(cons_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-cons_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]+(q_4[k*M*N+(j+3)*N+i]*q_2[k*M*N+(j+3)*N+i]-q_4[k*M*N+(j-3)*N+i]*q_2[k*M*N+(j-3)*N+i]))-0.0035f*(cons_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-cons_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i]+(q_4[k*M*N+(j+4)*N+i]*q_2[k*M*N+(j+4)*N+i]-q_4[k*M*N+(j-4)*N+i]*q_2[k*M*N+(j-4)*N+i])))*dxinv1;
  } 
}

extern "C" void offload (double *h_flux_0, double *h_flux_1, double *h_flux_2, double *h_flux_3, double *h_flux_4, double *h_cons_1, double *h_cons_2, double *h_cons_3, double *h_cons_4, double *h_q_1, double *h_q_2, double *h_q_3, double *h_q_4, double dxinv0, double dxinv1, double dxinv2, int L, int M, int N, int repeat) {

  size_t vol_size = sizeof(double)*L*M*N;

  double *flux_0;
  cudaMalloc (&flux_0, vol_size);
  check_error ("Failed to allocate device memory for flux_0\n");
  double *flux_1;
  cudaMalloc (&flux_1, vol_size);
  check_error ("Failed to allocate device memory for flux_1\n");
  double *flux_2;
  cudaMalloc (&flux_2, vol_size);
  check_error ("Failed to allocate device memory for flux_2\n");
  double *flux_3;
  cudaMalloc (&flux_3, vol_size);
  check_error ("Failed to allocate device memory for flux_3\n");
  double *flux_4;
  cudaMalloc (&flux_4, vol_size);
  check_error ("Failed to allocate device memory for flux_4\n");
  double *cons_1;
  cudaMalloc (&cons_1, vol_size);
  check_error ("Failed to allocate device memory for cons_1\n");
  cudaMemcpy (cons_1, h_cons_1, vol_size, cudaMemcpyHostToDevice);
  double *cons_2;
  cudaMalloc (&cons_2, vol_size);
  check_error ("Failed to allocate device memory for cons_2\n");
  cudaMemcpy (cons_2, h_cons_2, vol_size, cudaMemcpyHostToDevice);
  double *cons_3;
  cudaMalloc (&cons_3, vol_size);
  check_error ("Failed to allocate device memory for cons_3\n");
  cudaMemcpy (cons_3, h_cons_3, vol_size, cudaMemcpyHostToDevice);
  double *cons_4;
  cudaMalloc (&cons_4, vol_size);
  check_error ("Failed to allocate device memory for cons_4\n");
  cudaMemcpy (cons_4, h_cons_4, vol_size, cudaMemcpyHostToDevice);
  double *q_1;
  cudaMalloc (&q_1, vol_size);
  check_error ("Failed to allocate device memory for q_1\n");
  cudaMemcpy (q_1, h_q_1, vol_size, cudaMemcpyHostToDevice);
  double *q_2;
  cudaMalloc (&q_2, vol_size);
  check_error ("Failed to allocate device memory for q_2\n");
  cudaMemcpy (q_2, h_q_2, vol_size, cudaMemcpyHostToDevice);
  double *q_3;
  cudaMalloc (&q_3, vol_size);
  check_error ("Failed to allocate device memory for q_3\n");
  cudaMemcpy (q_3, h_q_3, vol_size, cudaMemcpyHostToDevice);
  double *q_4;
  cudaMalloc (&q_4, vol_size);
  check_error ("Failed to allocate device memory for q_4\n");
  cudaMemcpy (q_4, h_q_4, vol_size, cudaMemcpyHostToDevice);

  dim3 blockconfig (16, 4, 4);
  dim3 gridconfig (ceil(N, 16), ceil(M, 4), ceil(L, 4));

  long t1 = 0, t2 = 0, t3 = 0;

  for (int i = 0; i < repeat; i++) {
    cudaMemcpy (flux_0, h_flux_0, vol_size, cudaMemcpyHostToDevice);
    cudaMemcpy (flux_1, h_flux_1, vol_size, cudaMemcpyHostToDevice);
    cudaMemcpy (flux_2, h_flux_2, vol_size, cudaMemcpyHostToDevice);
    cudaMemcpy (flux_3, h_flux_3, vol_size, cudaMemcpyHostToDevice);
    cudaMemcpy (flux_4, h_flux_4, vol_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();
    hypterm_1 <<<gridconfig, blockconfig>>> (flux_0, flux_1, flux_2, flux_3, flux_4,
                                             cons_1, cons_2, cons_3, cons_4,
                                             q_1, q_2, q_3, q_4,
                                             dxinv0, dxinv1, dxinv2, L, M, N);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    t1 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    hypterm_2 <<<gridconfig, blockconfig>>> (flux_0, flux_1, flux_2, flux_3, flux_4,
                                             cons_1, cons_2, cons_3, cons_4,
                                             q_1, q_2, q_3, q_4,
                                             dxinv0, dxinv1, dxinv2, L, M, N);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    t2 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    start = std::chrono::steady_clock::now();
    hypterm_3 <<<gridconfig, blockconfig>>> (flux_0, flux_1, flux_2, flux_3, flux_4,
                                             cons_1, cons_2, cons_3, cons_4,
                                             q_1, q_2, q_3, q_4,
                                             dxinv0, dxinv1, dxinv2, L, M, N);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    t3 += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average kernel execution time (k1): %f (ms)\n", t1 * 1e-6 / repeat);
  printf("Average kernel execution time (k2): %f (ms)\n", t2 * 1e-6 / repeat);
  printf("Average kernel execution time (k3): %f (ms)\n", t3 * 1e-6 / repeat);

  cudaMemcpy (h_flux_0, flux_0, vol_size, cudaMemcpyDeviceToHost);
  cudaMemcpy (h_flux_1, flux_1, vol_size, cudaMemcpyDeviceToHost);
  cudaMemcpy (h_flux_2, flux_2, vol_size, cudaMemcpyDeviceToHost);
  cudaMemcpy (h_flux_3, flux_3, vol_size, cudaMemcpyDeviceToHost);
  cudaMemcpy (h_flux_4, flux_4, vol_size, cudaMemcpyDeviceToHost);

  cudaFree(cons_1);
  cudaFree(cons_2);
  cudaFree(cons_3);
  cudaFree(cons_4);
  cudaFree(q_1);
  cudaFree(q_2);
  cudaFree(q_3);
  cudaFree(q_4);
  cudaFree(flux_0);
  cudaFree(flux_1);
  cudaFree(flux_2);
  cudaFree(flux_3);
  cudaFree(flux_4);
}
