#ifdef __NVCC__
inline __host__ __device__ void operator+=(float2 &a, const float2 &b)
{
    a.x += b.x;
    a.y += b.y;
}

inline __host__ __device__ void operator+=(float4 &a, const float4 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
#endif

__global__
void linear_regression(
  const float2 *__restrict__ dataset,
        float4 *__restrict__ result)
{
  extern __shared__ float4 interns[];

  size_t loc_id   = threadIdx.x;
  size_t loc_size = blockDim.x; 
  size_t glob_id  = blockIdx.x * loc_size + loc_id;

  /* Initialize local buffer */
  interns[loc_id].x = dataset[glob_id].x;
  interns[loc_id].y = dataset[glob_id].y;
  interns[loc_id].z = (dataset[glob_id].x * dataset[glob_id].y);
  interns[loc_id].w = (dataset[glob_id].x * dataset[glob_id].x);
  
  __syncthreads();

  for (size_t i = (loc_size / 2), old_i = loc_size; i > 0; old_i = i, i /= 2)
  {
    if (loc_id < i) {
      // Only first half of workitems on each workgroup
      interns[loc_id] += interns[loc_id + i];
      if (loc_id == (i - 1) && old_i % 2 != 0) {
        // If there is an odd number of data
        interns[loc_id] += interns[old_i - 1];
      }
    }
    __syncthreads();
  }

  if (loc_id == 0) result[blockIdx.x] = interns[0];
}

__global__
void rsquared(
  const float2 *__restrict__ dataset,
  const float mean,
  const float2 equation, // [a0,a1]
  float2 *__restrict__ result)
{
  extern __shared__ float2 dist[];

  size_t loc_id   = threadIdx.x;
  size_t loc_size = blockDim.x; 
  size_t glob_id  = blockIdx.x * loc_size + loc_id;

  dist[loc_id].x = powf((dataset[glob_id].y - mean), 2.f);

  float y_estimated = dataset[glob_id].x * equation.y + equation.x;
  dist[loc_id].y = powf((y_estimated - mean), 2.f);

  __syncthreads();

  for (size_t i = (loc_size / 2), old_i = loc_size; i > 0; old_i = i, i /= 2)
  {
    if (loc_id < i) {
      // Only first half of workitems on each workgroup
      dist[loc_id] += dist[loc_id + i];
      if (loc_id == (i - 1) && old_i % 2 != 0) {
        // If there is an odd number of data
        dist[loc_id] += dist[old_i - 1];
      }
    }
    __syncthreads();
  }

  if (loc_id == 0) result[blockIdx.x] = dist[0];
}
