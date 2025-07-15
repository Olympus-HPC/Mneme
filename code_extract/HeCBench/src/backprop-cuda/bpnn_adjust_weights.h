__global__ void kernel_adjust_weights (
  const float*__restrict__ ly, 
       float *__restrict__ w, 
  const float*__restrict__ delta, 
        float*__restrict__ oldw, 
  const int hid)
{
  int by = blockIdx.y; 
  int tx = threadIdx.x; 
  int ty = threadIdx.y;

  int index =  ( hid + 1 ) * HEIGHT * by + ( hid + 1 ) * ty + tx + 1 + ( hid + 1 ) ;  
  int index_y = HEIGHT * by + ty + 1;
  int index_x = tx + 1;

  w[index] += ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));
  oldw[index] = ((ETA * delta[index_x] * ly[index_y]) + (MOMENTUM * oldw[index]));

  __syncthreads();

  if (ty == 0 && by ==0){
    w[index_x] += ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
    oldw[index_x] = ((ETA * delta[index_x]) + (MOMENTUM * oldw[index_x]));
  }
}

