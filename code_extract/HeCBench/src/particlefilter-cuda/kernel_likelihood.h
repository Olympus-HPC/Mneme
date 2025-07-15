__global__ void
kernel_likelihood (
    float*__restrict__ arrayX, 
    float*__restrict__ arrayY, 
    const float*__restrict__ xj,
    const float*__restrict__ yj,
    int*__restrict__ ind,
    const int*__restrict__ objxy,
    float*__restrict__ likelihood,
    const unsigned char*__restrict__ I,
    float*__restrict__ weights,
    int*__restrict__ seed,
    float*__restrict__ partial_sums,
    const int Nparticles,
    const int countOnes,
    const int IszY,
    const int Nfr,
    const int k,
    const int max_size)
{
  __shared__ float weights_local[BLOCK_SIZE];

  int block_id = blockIdx.x; 
  int thread_id = threadIdx.x;
  int i = block_id * blockDim.x + thread_id;
  int y;
  int indX, indY;
  float u, v;

  if(i < Nparticles){
    arrayX[i] = xj[i]; 
    arrayY[i] = yj[i]; 

    weights[i] = 1.0f / ((float) (Nparticles)); 
    seed[i] = (A*seed[i] + C) % M;
    u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    v = fabsf(seed[i]/((float)M));
    arrayX[i] += 1.0f + 5.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));

    seed[i] = (A*seed[i] + C) % M;
    u = fabsf(seed[i]/((float)M));
    seed[i] = (A*seed[i] + C) % M;
    v = fabsf(seed[i]/((float)M));
    arrayY[i] += -2.0f + 2.0f*(sqrtf(-2.0f*logf(u))*cosf(2.0f*PI*v));
  }

  __syncthreads();


  if(i < Nparticles)
  {
    for(y = 0; y < countOnes; y++){

      int iX = arrayX[i];
      int iY = arrayY[i];
      int rnd_iX = (arrayX[i] - iX) < .5f ? iX : iX++;
      int rnd_iY = (arrayY[i] - iY) < .5f ? iY : iY++;
      indX = rnd_iX + objxy[y*2 + 1];
      indY = rnd_iY + objxy[y*2];

      ind[i*countOnes + y] = abs(indX*IszY*Nfr + indY*Nfr + k);
      if(ind[i*countOnes + y] >= max_size)
        ind[i*countOnes + y] = 0;
    }
    float likelihoodSum = 0.0f;
    for(int x = 0; x < countOnes; x++)
      likelihoodSum += ((I[ind[i*countOnes + x]] - 100) * (I[ind[i*countOnes + x]] - 100) -
          (I[ind[i*countOnes + x]] - 228) * (I[ind[i*countOnes + x]] - 228)) / 50.0f;
    likelihood[i] = likelihoodSum/countOnes-SCALE_FACTOR;

    weights[i] = weights[i] * expf(likelihood[i]); //Donnie Newell - added the missing exponential function call

  }

  weights_local[thread_id] = (i < Nparticles) ? weights[i] : 0.f;

  __syncthreads();

  for(unsigned int s=BLOCK_SIZE/2; s>0; s>>=1)
  {
    if(thread_id < s)
    {
      weights_local[thread_id] += weights_local[thread_id + s];
    }
    __syncthreads();
  }
  if(thread_id == 0)
  {
    partial_sums[block_id] = weights_local[0];
  }
}
