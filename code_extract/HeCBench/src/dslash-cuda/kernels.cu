#include "dslash.h"

__global__ void make_back(
    const su3_matrix*__restrict__ d_fat,
    const su3_matrix*__restrict__ d_lng,
    const size_t*__restrict__ d_bck, 
    const size_t*__restrict__ d_bck3,
          su3_matrix*__restrict__ d_fatbck,
          su3_matrix*__restrict__ d_lngbck,
    const int total_even_sites)
{
  size_t mySite = blockIdx.x * blockDim.x + threadIdx.x;
  if (mySite < total_even_sites) {
    for(int dir = 0; dir < 4; dir++) {
      su3_adjoint( d_fat + 4*d_bck[4*mySite+dir]+dir, 
          d_fatbck + 4*mySite+dir );
      su3_adjoint( d_lng + 4*d_bck3[4*mySite+dir]+dir, 
          d_lngbck + 4*mySite+dir );
    }
  }
}

__global__ void dslash (
    const su3_matrix*__restrict__ d_fat,
    const su3_matrix*__restrict__ d_lng,
    const su3_matrix*__restrict__ d_fatbck,
    const su3_matrix*__restrict__ d_lngbck,
    const su3_vector*__restrict__ d_src,
          su3_vector*__restrict__ d_dst,
    const size_t*__restrict__ d_fwd,
    const size_t*__restrict__ d_bck,
    const size_t*__restrict__ d_fwd3,
    const size_t*__restrict__ d_bck3,
    const int total_even_sites)
{
  size_t mySite = blockIdx.x * blockDim.x + threadIdx.x;
  if (mySite < total_even_sites) {
    su3_vector v;
    for (size_t k=0; k<4; ++k) {
      auto a = d_fat + mySite*4 + k;
      auto b = d_src + d_fwd[4*mySite + k];
      if (k == 0)
        mult_su3_mat_vec(a, b, &d_dst[mySite]);
      else 
        mult_su3_mat_vec_sum(a, b, &d_dst[mySite]);
    }
    for (size_t k=0; k<4; ++k) {
      auto a = d_lng + mySite*4 + k;
      auto b = d_src + d_fwd3[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    add_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
    for (size_t k=0; k<4; ++k) {
      auto a = d_fatbck + mySite*4 + k;
      auto b = d_src + d_bck[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
    for (size_t k=0; k<4; ++k) {
      auto a = d_lngbck + mySite*4 + k;
      auto b = d_src + d_bck3[4*mySite + k];
      if (k == 0) 
        mult_su3_mat_vec(a, b, &v);
      else
        mult_su3_mat_vec_sum(a, b, &v);
    }
    sub_su3_vector(&d_dst[mySite], &v, &d_dst[mySite]);
  } // end of if mySite
}

double dslash_fn(
    const std::vector<su3_vector> &src, 
    std::vector<su3_vector> &dst,
    const std::vector<su3_matrix> &fat,
    const std::vector<su3_matrix> &lng,
    std::vector<su3_matrix> &fatbck,
    std::vector<su3_matrix> &lngbck,
    size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3,    
    const size_t iterations,
    size_t wgsize )
{ 

  // Set the loop and work-group parameters
  size_t total_sites = sites_on_node; 
  size_t total_even_sites = even_sites_on_node;

  auto copy_start = Clock::now();

  // allocate device memory
  su3_vector* d_src;
  cudaMalloc((void**)&d_src, total_sites * 1 * sizeof(su3_vector));
  cudaMemcpy(d_src, src.data(), total_sites * 1 * sizeof(su3_vector), cudaMemcpyHostToDevice);

  su3_matrix* d_fat;
  cudaMalloc((void**)&d_fat, total_sites * 4 * sizeof(su3_matrix));
  cudaMemcpy(d_fat, fat.data(), total_sites * 4 * sizeof(su3_matrix), cudaMemcpyHostToDevice);

  su3_matrix* d_lng;
  cudaMalloc((void**)&d_lng, total_sites * 4 * sizeof(su3_matrix));
  cudaMemcpy(d_lng, lng.data(), total_sites * 4 * sizeof(su3_matrix), cudaMemcpyHostToDevice);

  su3_vector* d_dst;
  cudaMalloc((void**)&d_dst, total_sites * 1 * sizeof(su3_vector));

  su3_matrix* d_fatbck;
  cudaMalloc((void**)&d_fatbck, total_sites * 4 * sizeof(su3_matrix));

  su3_matrix* d_lngbck;
  cudaMalloc((void**)&d_lngbck, total_sites * 4 * sizeof(su3_matrix));

  size_t* d_fwd;
  cudaMalloc((void**)&d_fwd, total_sites * 4 * sizeof(size_t));
  cudaMemcpy(d_fwd, fwd, total_sites * 4 * sizeof(size_t), cudaMemcpyHostToDevice);

  size_t* d_bck;
  cudaMalloc((void**)&d_bck, total_sites * 4 * sizeof(size_t));
  cudaMemcpy(d_bck, bck, total_sites * 4 * sizeof(size_t), cudaMemcpyHostToDevice);

  size_t* d_fwd3;
  cudaMalloc((void**)&d_fwd3, total_sites * 4 * sizeof(size_t));
  cudaMemcpy(d_fwd3, fwd3, total_sites * 4 * sizeof(size_t), cudaMemcpyHostToDevice);

  size_t* d_bck3;
  cudaMalloc((void**)&d_bck3, total_sites * 4 * sizeof(size_t));
  cudaMemcpy(d_bck3, bck3, total_sites * 4 * sizeof(size_t), cudaMemcpyHostToDevice);

  // allocate offsets for device gathers and copy to shared buffers

  double copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload input data = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  // Create backward links on the device
  size_t total_wi = total_even_sites;
  if (verbose > 1) {
    std::cout << "Creating backward links"  << std::endl;
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << 1 << std::endl;
  }
  auto back_start = Clock::now();

  make_back<<<total_wi, 1>>>(d_fat, d_lng, d_bck, d_bck3, d_fatbck, d_lngbck, total_even_sites);

  cudaDeviceSynchronize();
  double back_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-back_start).count();
  if (verbose > 1) {
    std::cout << "Time to create back links = " << back_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  // Dslash benchmark loop
  total_wi = total_even_sites;
  if (verbose > 0) {
    std::cout << "Running dslash loop" << std::endl;
    std::cout << "Setting number of work items to " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << wgsize << std::endl;
  }
  auto tstart = Clock::now();
  for (size_t iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      cudaDeviceSynchronize();
      tstart = Clock::now();
    } 
    // Dslash kernel
    dim3 grid ((total_wi + wgsize - 1) / wgsize);
    dim3 block (wgsize);
    dslash<<<grid, block>>>(d_fat, d_lng, d_fatbck, d_lngbck, d_src, d_dst, 
                            d_fwd, d_bck, d_fwd3, d_bck3, total_even_sites);

    cudaDeviceSynchronize();
  } // end of iteration loop
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(
      Clock::now()-tstart).count();

  // Move the result back to the host
  copy_start = Clock::now();

  cudaMemcpy(dst.data(), d_dst, total_sites * 1 * sizeof(su3_vector), cudaMemcpyDeviceToHost);
  cudaMemcpy(fatbck.data(), d_fatbck, total_sites * 4 * sizeof(su3_matrix), cudaMemcpyDeviceToHost);
  cudaMemcpy(lngbck.data(), d_lngbck, total_sites * 4 * sizeof(su3_matrix), cudaMemcpyDeviceToHost);

  copy_time = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-copy_start).count();
  if (verbose > 1) {
    std::cout << "Time to offload backward links = " << copy_time/1.0e6 << " secs\n";
    std::cout << std::flush;
  }

  cudaFree(d_src);
  cudaFree(d_fat);
  cudaFree(d_lng);
  cudaFree(d_dst);
  cudaFree(d_fatbck);
  cudaFree(d_lngbck);
  cudaFree(d_fwd);
  cudaFree(d_bck);
  cudaFree(d_fwd3);
  cudaFree(d_bck3);

  return (ttotal /= 1.0e6);
}
