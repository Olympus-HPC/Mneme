__global__ void
kernel_find_index (
    const float*__restrict__ arrayX,
    const float*__restrict__ arrayY,
    const float*__restrict__ CDF,
    const float*__restrict__ u,
          float*__restrict__ xj,
          float*__restrict__ yj,
    const int Nparticles)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < Nparticles){
    int index = -1;
    int x;

    for(x = 0; x < Nparticles; x++){
      if(CDF[x] >= u[i]){
        index = x;
        break;
      }
    }
    if(index == -1){
      index = Nparticles-1;
    }

    xj[i] = arrayX[index];
    yj[i] = arrayY[index];
  }
}
