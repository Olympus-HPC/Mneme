__global__ void mr32_sf(
  const uint32_t *__restrict__ bases,
  const uint32_t *__restrict__ n32,
  int *__restrict__ val,
  int iter)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      atomicAdd(val, straightforward_mr32(bases, cnt, n));
    }
  }
}

__global__ void mr32_eff(
  const uint32_t *__restrict__ bases,
  const uint32_t *__restrict__ n32,
  int *__restrict__ val,
  int iter)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < iter) {
    int n = n32[j];
    for (int cnt = 1; cnt <= BASES_CNT32; cnt++) {
      atomicAdd(val, efficient_mr32(bases, cnt, n));
    }
  }
}
