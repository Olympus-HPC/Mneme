/**
 * @brief      Calculate Gray-Scott reaction rate
 */
__global__ void reaction_gray_scott(
    const float *__restrict__ fx,
    const float *__restrict__ fy,
    float *__restrict__ drx,
    float *__restrict__ dry,
    const unsigned int ncells,
    const float d_c1,
    const float d_c2)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < ncells; i += stride) {
    float r = fx[i] * fy[i] * fy[i];
    drx[i] = -r + d_c1 * (1.f - fx[i]);
    dry[i] = r - (d_c1 + d_c2) * fy[i];
  }
}

/**
 * @brief      Calculate second derivative in x direction with periodic boundary conditions
 */
__global__ void derivative_x2_pbc(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  const int offset = 1;
  extern __shared__ float s_f[]; // 2-wide halo

  int i   = threadIdx.x;
  int j   = blockIdx.x * blockDim.y + threadIdx.y;
  int k   = blockIdx.y;
  int si  = i + offset;  // local i for shared memory access + halo offset
  int sj  = threadIdx.y; // local j for shared memory access

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * (mx + 2 * offset) + si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array
  if (i < offset) {
    s_f[sj * (mx + 2 * offset) + si - offset]  = s_f[sj * (mx + 2 * offset) + si + mx - offset];
    s_f[sj * (mx + 2 * offset) + si + mx] = s_f[sj * (mx + 2 * offset) + si];
  }

  __syncthreads();

  df[globalIdx] = s_f[sj * (mx + 2 * offset) + si + 1] - 2.f * s_f[sj * (mx + 2 * offset) + si] + s_f[sj * (mx + 2 * offset) + si - 1];
}

/**
 * @brief      Calculate second derivative in x direction with zero-flux boundary conditions
 */
__global__ void derivative_x2_zeroflux(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my)
{
  extern __shared__ float s_f[];

  int i   = threadIdx.x;
  int j   = blockIdx.x * blockDim.y + threadIdx.y;
  int k   = blockIdx.y;
  int sj  = threadIdx.y; // local j for shared memory access

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * mx + i] = f[globalIdx];

  __syncthreads();

  if(i == 0) {
    df[globalIdx] = s_f[sj * mx + i + 1] - s_f[sj * mx + i];
  } else if(i == (mx - 1)) {
    df[globalIdx] = s_f[sj * mx + i - 1] - s_f[sj * mx + i];
  } else {
    df[globalIdx] = s_f[sj * mx + i + 1] - 2.f * s_f[sj * mx + i] + s_f[sj * mx + i - 1];
  }
}

/**
 * @brief      Calculate second derivative in y direction with periodic boundary conditions
 */
__global__ void derivative_y2_pbc(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  const int offset = 1;
  extern __shared__ float s_f[]; // 2-wide halo

  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = threadIdx.y;
  int k  = blockIdx.y;
  int si = threadIdx.x;
  int sj = j + offset;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sj * pencils + si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array
  if (j < offset) {
    s_f[(sj - offset) * pencils + si]  = s_f[(sj + my - offset) * pencils + si];
    s_f[(sj + my) * pencils + si] = s_f[sj * pencils + si];
  }

  __syncthreads();

  df[globalIdx] = s_f[(sj+1) * pencils + si] - 2.f * s_f[sj * pencils + si] + s_f[(sj-1) * pencils + si];
}

/**
 * @brief      Calculate second derivative in y direction with zero-flux  boundary conditions
 */
__global__ void derivative_y2_zeroflux(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int pencils)
{
  extern __shared__ float s_f[];

  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = threadIdx.y;
  int k  = blockIdx.y;
  int si = threadIdx.x;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[j * pencils + si] = f[globalIdx];

  __syncthreads();

  if(j == 0) {
    df[globalIdx] = s_f[(j+1) * pencils + si] - s_f[j * pencils + si];
  } else if(j == (my - 1)) {
    df[globalIdx] = s_f[(j-1) * pencils + si] - s_f[j * pencils + si];
  } else {
    df[globalIdx] = s_f[(j+1) * pencils + si] - 2.f * s_f[j * pencils + si] + s_f[(j-1) * pencils + si];
  }
}

/**
 * @brief      Calculate second derivative in z direction with periodic boundary conditions
 */
__global__ void derivative_z2_pbc(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils)
{
  const int offset = 1;
  extern __shared__ float s_f[]; // 2-wide halo

  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int k  = threadIdx.y;
  int si = threadIdx.x;
  int sk = k + offset; // halo offset

  int globalIdx = k * mx * my + j * mx + i;

  s_f[sk * pencils + si] = f[globalIdx];

  __syncthreads();

  // fill in periodic images in shared memory array
  if (k < offset) {
    s_f[(sk - offset) * pencils + si]  = s_f[(sk + mz - offset) * pencils + si];
    s_f[(sk + mz) * pencils + si] = s_f[sk * pencils + si];
  }

  __syncthreads();

  df[globalIdx] = s_f[(sk+1) * pencils + si] - 2.f * s_f[sk * pencils + si] + s_f[(sk-1) * pencils + si];
}

/**
 * @brief      Calculate second derivative in z direction with zero-flux boundary conditions
 */
__global__ void derivative_z2_zeroflux(
    const float *__restrict__ f,
    float *__restrict__ df,
    const unsigned int mx,
    const unsigned int my,
    const unsigned int mz,
    const unsigned int pencils)
{
  extern __shared__ float s_f[]; // 2-wide halo

  int i  = blockIdx.x * blockDim.x + threadIdx.x;
  int j  = blockIdx.y;
  int k  = threadIdx.y;
  int si = threadIdx.x;

  int globalIdx = k * mx * my + j * mx + i;

  s_f[k * pencils + si] = f[globalIdx];

  __syncthreads();

  if(k == 0) {
    df[globalIdx] = s_f[(k+1) * pencils + si] - s_f[k * pencils + si];
  } else if(k == (mz - 1)) {
    df[globalIdx] = s_f[(k-1) * pencils + si] - s_f[k * pencils + si];
  } else {
    df[globalIdx] = s_f[(k+1) * pencils + si] - 2.f * s_f[k * pencils + si] + s_f[(k-1) * pencils + si];
  }
}

/**
 * @brief      Construct the Laplacian for a component 
 */
__global__ void construct_laplacian(
    float *__restrict__ df,
    const float *__restrict__ dfx,
    const float *__restrict__ dfy,
    const float *__restrict__ dfz,
    const unsigned int ncells,
    const float d_diffcon)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < ncells; i += stride) {
    df[i] = d_diffcon * (dfx[i] + dfy[i] + dfz[i]);
  }
}

/**
 * @brief      Perform time-step integration
 *
 * @param      x     pointer to concentration of A
 * @param      y     pointer to concentration of B
 * @param[in]  ddx   diffusion of component A
 * @param[in]  ddy   diffusion of component B
 * @param[in]  drx   reaction of component A
 * @param[in]  dry   reaction of component B
 */
__global__ void update(
    float *__restrict__ x,
    float *__restrict__ y,
    const float *__restrict__ ddx,
    const float *__restrict__ ddy,
    const float *__restrict__ drx,
    const float *__restrict__ dry,
    const unsigned int ncells,
    const float d_dt)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < ncells; i += stride) {
    x[i] += (ddx[i] + drx[i]) * d_dt;
    y[i] += (ddy[i] + dry[i]) * d_dt;
  }
}
