__global__ 
void kernel_dilated (
    const float*__restrict__ c_strel,
    const float*__restrict__ img,
    float*__restrict__ dilated,
    const int strel_m,
    const int strel_n,
    const int max_gicov_m,
    const int max_gicov_n)
{
  // Find the center of the structuring element
  int el_center_i = strel_m / 2;
  int el_center_j = strel_n / 2;

  // Determine this thread's location in the matrix
  int thread_id = blockIdx.x*blockDim.x+threadIdx.x;
  int i = thread_id % max_gicov_m;
  int j = thread_id / max_gicov_m;

  if(j >= max_gicov_n) return;

  // Initialize the maximum GICOV score seen so far to zero
  float max = 0.0f;

  // Iterate across the structuring element in one dimension
  int el_i, el_j, x, y;

  for (el_i = 0; el_i < strel_m; el_i++) {
    y = i - el_center_i + el_i;
    // Make sure we have not gone off the edge of the matrix
    if ( (y >= 0) && (y < max_gicov_m) ) {
      // Iterate across the structuring element in the other dimension
      for (el_j = 0; el_j < strel_n; el_j++) {
        x = j - el_center_j + el_j;
        // Make sure we have not gone off the edge of the matrix
        //  and that the current structuring element value is not zero
        if ( (x >= 0) &&
            (x < max_gicov_n) &&
            (c_strel[(el_i * strel_n) + el_j] != 0) ) {
          // Determine if this is the maximal value seen so far
          int addr = (x * max_gicov_m) + y;
          float temp = img[addr];
          if (temp > max) max = temp;
        }
      }
    }
  }
  // Store the maximum value found
  dilated[(i * max_gicov_n) + j] = max;
}

