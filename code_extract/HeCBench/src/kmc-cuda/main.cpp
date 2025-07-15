#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include "driver.c"

void cal_km(struct svm_problem *pecm)
{
  int i_v, i_el, i_r, i_c, trvei;

  int len_tv = prob.x[0].dim;
  int ntv = prob.l;

  float *tva = (float*) malloc ( len_tv * ntv* sizeof(float) );
  float *vtm = (float*) malloc ( len_tv * sizeof(float) );
  float *dp  = (float*) malloc ( ntv * sizeof(float) );
  float *tr_ar = (float*) malloc ( len_tv * ntv* sizeof(float) );

  double *tv_sq = (double*) malloc ( ntv * sizeof(double) );
  double *v_f_g = (double*) malloc ( ntv * sizeof(double) );

  float gamma = param.gamma;
  double g_val = (double)gamma;

  for ( i_r = 0; i_r < ntv ; i_r++ )
  {
    for ( i_c = 0; i_c < len_tv; i_c++ )
      tva[i_r * len_tv + i_c] = (float)prob.x[i_r].values[i_c];
  }

  for( i_r = 0; i_r < ntv; i_r++ )
    for( i_c = 0; i_c < len_tv; i_c++ )
      tr_ar[i_c * ntv + i_r] = tva[i_r * len_tv + i_c];

  for( i_v = 0; i_v < ntv; i_v++ )
  {
    tv_sq[ i_v ] = 0;
    for( i_el = 0; i_el < len_tv; i_el++ )
      tv_sq[i_v] += pow( tva[i_v*len_tv + i_el], (float)2.0 );
  }

  // offload
  auto start_offload = std::chrono::steady_clock::now();
  const float alpha = 1;
  const float beta = 0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  float *g_tva, *g_vtm, *g_dp;

  cudaMalloc((void**)&g_tva, len_tv * ntv * sizeof(float));
  cudaMalloc((void**)&g_vtm, len_tv * sizeof(float));
  cudaMalloc((void**)&g_dp, ntv * sizeof(float));

  // Copy cpu vector to gpu vector
  //cublasSetVector( len_tv * ntv, sizeof(float), tr_ar, 1, g_tva, 1 );
  cudaMemcpy(g_tva, tr_ar, len_tv * ntv * sizeof(float), cudaMemcpyHostToDevice);

  double time = 0.0;
  for ( trvei = 0; trvei < ntv; trvei++ )
  {
    auto start = std::chrono::steady_clock::now();

    //cublasSetVector( len_tv, sizeof(float), &tva[trvei * len_tv], 1, g_vtm, 1 );
    cudaMemcpy( g_vtm, &tva[trvei * len_tv], len_tv * sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemv( handle, CUBLAS_OP_N, ntv, len_tv, &alpha, g_tva, ntv , g_vtm, 1, &beta, g_dp, 1 );

    //cublasGetVector( ntv, sizeof(float), g_dp, 1, dp, 1 );
    cudaMemcpy(dp, g_dp, ntv * sizeof(float), cudaMemcpyDeviceToHost);

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    for ( i_c = 0; i_c < ntv; i_c++ ) {
      v_f_g[i_c] = exp( -g_val * (tv_sq[trvei] + tv_sq[i_c]-((double)2.0)* (double)dp[i_c] ));
    }

    pecm-> x[trvei].values[0] = trvei + 1;

    for ( i_c = 0; i_c < ntv; i_c++ )
      pecm-> x[trvei].values[i_c + 1] = v_f_g[i_c];
  }

  printf("Average kernel matrix offload time: %lf (us)\n", (time * 1e-3) / ntv);

  cudaFree( g_tva );
  cudaFree( g_vtm );
  cudaFree( g_dp );
  cublasDestroy( handle );
  auto end_offload = std::chrono::steady_clock::now();
  auto time_offload = std::chrono::duration_cast<std::chrono::nanoseconds>(end_offload - start_offload).count();
  printf("Total kernel matrix execution time: %lf (us)\n", time_offload * 1e-3);

  free( tr_ar );
  free( tva );
  free( vtm );
  free( dp  );
  free( v_f_g );
  free( tv_sq );
}
