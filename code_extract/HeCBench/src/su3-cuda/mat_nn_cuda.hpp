// Cuda implementation
#include <cuda_runtime.h>

#define CUCHECK(err, s) \
  if (err != cudaSuccess) { \
        printf("%s (error code %d:%s)!\n", s, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
  }

#define THREADS_PER_SITE 36

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints 
//  C  <-  A*B	
__global__ void k_mat_nn(
  const site*       __restrict__ a,
  const su3_matrix* __restrict__ b,
        site*       __restrict__ c,
  int               total_sites)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int i = id/36;

  if (i < total_sites) {
    int j = (id%36)/9;
    int k = (id%9)/3;
    int l = id%3;
    Complx cc = {0.0, 0.0};
    for (int m=0;m<3;m++)
#ifdef MILC_COMPLEX
      CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], cc);
#else
      cc += a[i].link[j].e[k][m] * b[j].e[m][l];
#endif
    c[i].link[j].e[k][l] = cc;

  }
}


#ifdef MILC_COMPLEX
double su3_mat_nn(const std::vector<site> &a, const std::vector<su3_matrix> &b, std::vector<site> &c, 
#else
double su3_mat_nn(thrust::host_vector<site> &a, thrust::host_vector<su3_matrix> &b, thrust::host_vector<site> &c, 
#endif
              size_t total_sites, int iterations, int threadsPerBlock, int use_device)
{
  int blocksPerGrid;
  int size_a = sizeof(site) * total_sites;
  int size_b = sizeof(su3_matrix) * 4;
  int size_c = sizeof(site) * total_sites;

  if (threadsPerBlock == 0)
    threadsPerBlock = THREADS_PER_SITE;

  // Declare target storage and copy A and B
  cudaError_t cuErr;
  site *d_a, *d_c;
  su3_matrix *d_b;
  cuErr = cudaMalloc((void **)&d_a, size_a);
  CUCHECK(cuErr, "Unable to allocate array d_a");
  cuErr = cudaMalloc((void **)&d_b, size_b);
  CUCHECK(cuErr, "Unable to allocate array d_b");
  cuErr = cudaMalloc((void **)&d_c, size_c);
  CUCHECK(cuErr, "Unable to allocate array d_c");
  cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice);

  blocksPerGrid = total_sites;

  if (verbose >= 1) {
    printf("Number of blocks set to %d\n", blocksPerGrid);
    printf("Threads per block set to %d\n", threadsPerBlock);
  }

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      cudaDeviceSynchronize();
      tstart = Clock::now();
    }
    k_mat_nn<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, total_sites);
  }
  cudaDeviceSynchronize();
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  CUCHECK(cudaGetLastError(), "k_mat_nn kernel Failed");

  // copy data back from device
  cudaMemcpy(c.data(), d_c, size_c, cudaMemcpyDeviceToHost);

  // Deallocate
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return (ttotal /= 1.0e6);
}
