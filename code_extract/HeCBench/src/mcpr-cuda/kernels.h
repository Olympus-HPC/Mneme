__global__ void compute_probs(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if(i < n) {
    double maxval;    
    int m, k;
    int maxind;
    double M_d = (double) M; 
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){   // initialize probs (though already done on CPU)
      probs[i*K + k] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){   // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        w[k] = alphas[i*K + k] + rands[m*K + k];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        } 
      }
      probs[i*K + maxind] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      probs[i*K + k] /= M_d;
    }
  }
}

__global__ void compute_probs_unitStrides(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x; 

  if(i < n) {
    double maxval;    
    int m, k;
    int maxind;
    double M_d = (double) M; 
    double w[21]; // w[K]

    for(k = 0; k < K; ++k){  // initialize probs (though already done on CPU)
      probs[k*n + i] = 0.0;
    }

    // core computations
    for(m = 0; m < M; ++m){    // loop over Monte Carlo iterations
      for(k = 0; k < K; ++k){  // generate W ~ N(alpha, 1)
        // with +i we now have unit strides in inner loop
        w[k] = alphas[k*n + i] + rands[k*M + m];
      }

      // determine which category has max W
      maxind = K-1;
      maxval = w[K-1];
      for(k = 0; k < (K-1); ++k){
        if(w[k] > maxval){
          maxind = k;
          maxval = w[k];
        } 
      }
      probs[maxind*n + i] += 1.0;
    }

    // compute final proportions
    for(k = 0; k < K; ++k) {
      // unit strides
      probs[k*n + i] /= M_d;
    }
  }
}

__global__ void compute_probs_unitStrides_sharedMem(
  const double* __restrict__ alphas,
  const double* __restrict__ rands,
        double* __restrict__ probs,
  int n, int K, int M)
{
  // assign overall id/index of the thread = id of row
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  if (i >= n) return;

  int threads_per_block = blockDim.x; 

  // set up shared memory: half for probs and half for w
  extern __shared__ double shared[];
  double* probs_shared = shared;

  // shared mem is one big block, so need to index into latter portion of it to use for w
  double* w = &shared[K*threads_per_block];

  double maxval;    
  int m, k;
  int maxind;
  double M_d = (double) M; 

  // initialize shared memory probs
  for(k = 0; k < K; ++k) {
    probs_shared[k*threads_per_block + threadIdx.x] = 0.0;
  }

  // core computation
  for(m = 0; m < M; ++m){     // loop over Monte Carlo iterations 
    for(k = 0; k < K; ++k){   // generate W ~ N(alpha, 1)
      w[k*threads_per_block + threadIdx.x] = alphas[k*n + i] + rands[k*M + m];
    }
    maxind = K-1;
    maxval = w[(K-1)*threads_per_block + threadIdx.x];
    for(k = 0; k < (K-1); ++k){
      if(w[k*threads_per_block + threadIdx.x] > maxval){
        maxind = k;
        maxval = w[k*threads_per_block + threadIdx.x];
      } 
    }
    probs_shared[maxind*threads_per_block + threadIdx.x] += 1.0;
  }

  for(k = 0; k < K; ++k) {
    probs_shared[k*threads_per_block + threadIdx.x] /= M_d;
  }

  // copy to device memory so can be returned to CPU
  for(k = 0; k < K; ++k) {
    probs[k*n + i] = probs_shared[k*threads_per_block + threadIdx.x];
  }
}
