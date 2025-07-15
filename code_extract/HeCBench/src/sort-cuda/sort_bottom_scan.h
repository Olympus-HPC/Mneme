__global__ void
bottom_scan (T* out, const T* in, const T* isums, const size_t size, const unsigned int shift)
{

  __shared__ T lmem[256*2];
  __shared__ T l_scanned_seeds[16];
  __shared__ T l_block_counts[16];

  int group_range = gridDim.x;
  int group = blockIdx.x;
  int lid = threadIdx.x;
  int local_range = blockDim.x;


  // Keep a private histogram as well
  int histogram[16];

  // Prepare for reading 4-element vectors
  // Assume: divisible by 4

  int n4 = size / 4; //vector type is 4 wide

  int region_size = n4 / group_range;
  int block_start = group * region_size;
  // Give the last block any extra elements
  int block_stop  = (group == group_range - 1) ?
    n4 : block_start + region_size;

  // Calculate starting index for this thread/work item
  int i = block_start + lid;
  int window = block_start;

  // Set the histogram in local memory to zero
  // and read in the scanned seeds from gmem
  if (lid < 16)
  {
    l_block_counts[lid] = 0;
    l_scanned_seeds[lid] =
      isums[(lid*group_range)+group];
  }
  __syncthreads();

  // Scan multiple elements per thread
  while (window < block_stop)
  {
    // Reset histogram
    for (int q = 0; q < 16; q++) histogram[q] = 0;
    VECTYPE val_4;
    VECTYPE key_4;

    if (i < block_stop) // Make sure we don't read out of bounds
    {
      val_4 = ((VECTYPE*)in)[i];

      // Mask the keys to get the appropriate digit
      key_4.x = (val_4.x >> shift) & 0xFU;
      key_4.y = (val_4.y >> shift) & 0xFU;
      key_4.z = (val_4.z >> shift) & 0xFU;
      key_4.w = (val_4.w >> shift) & 0xFU;

      // Update the histogram
      histogram[key_4.x]++;
      histogram[key_4.y]++;
      histogram[key_4.z]++;
      histogram[key_4.w]++;
    }

    // Scan the digit counts in local memory
    for (int digit = 0; digit < 16; digit++)
    {
      int idx = lid;
      lmem[idx] = 0;
      idx += local_range;
      lmem[idx] = histogram[digit];
      __syncthreads();
      for (int i = 1; i < local_range; i *= 2)
      {
        T t = lmem[idx -  i]; 
        __syncthreads();
        lmem[idx] += t;     
        __syncthreads();
      }
      histogram[digit] = lmem[idx-1];

      //histogram[digit] = scanLocalMem(histogram[digit], lmem, 1);
      __syncthreads();
    }

    if (i < block_stop) // Make sure we don't write out of bounds
    {
      int address;
      address = histogram[key_4.x] + l_scanned_seeds[key_4.x] + l_block_counts[key_4.x];
      out[address] = val_4.x;
      histogram[key_4.x]++;

      address = histogram[key_4.y] + l_scanned_seeds[key_4.y] + l_block_counts[key_4.y];
      out[address] = val_4.y;
      histogram[key_4.y]++;

      address = histogram[key_4.z] + l_scanned_seeds[key_4.z] + l_block_counts[key_4.z];
      out[address] = val_4.z;
      histogram[key_4.z]++;

      address = histogram[key_4.w] + l_scanned_seeds[key_4.w] + l_block_counts[key_4.w];
      out[address] = val_4.w;
      histogram[key_4.w]++;
    }

    // Before proceeding, make sure everyone has finished their current
    // indexing computations.
    __syncthreads();
    // Now update the seed array.
    if (lid == local_range-1)
    {
      for (int q = 0; q < 16; q++)
      {
        l_block_counts[q] += histogram[q];
      }
    }
    __syncthreads();

    // Advance window
    window += local_range;
    i += local_range;
  }
}

