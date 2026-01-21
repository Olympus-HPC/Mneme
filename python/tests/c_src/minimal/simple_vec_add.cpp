
__global__ void kernel_add(int *p, int val){
  int tid = threadIdx.x;
  if (tid > 512)
    return;
  p[tid] += val;
}
