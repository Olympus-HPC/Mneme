__global__ void kernel_add(int *p, int val){
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
  int gid = threads_per_block + tid;
  
  if (gid > 512)
    return;
  p[gid] += val;
}
