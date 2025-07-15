__global__ void
kernel_sum (float* partial_sums, const int Nparticles)
{
  int x;
  float sum = 0;
  int num_blocks = (Nparticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (x = 0; x < num_blocks; x++) {
    sum += partial_sums[x];
  }
  partial_sums[0] = sum;
}

