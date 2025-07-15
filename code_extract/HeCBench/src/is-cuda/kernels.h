#define R23 pow(0.5, 23.0)
#define R46 (R23*R23)
#define T23 pow(2.0, 23.0)
#define T46 (T23*T23)

__device__ double randlc_device(double* X, const double* A)
{
  double T1, T2, T3, T4;
  double A1;
  double A2;
  double X1;
  double X2;
  double Z;
  int j;

  /*
   * --------------------------------------------------------------------
   * break A into two parts such that A = 2^23 * A1 + A2 and set X = N.
   * --------------------------------------------------------------------
   */
  T1 = R23 * *A;
  j  = T1;
  A1 = j;
  A2 = *A - T23 * A1;

  /*
   * --------------------------------------------------------------------
   * break X into two parts such that X = 2^23 * X1 + X2, compute
   * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
   * X = 2^23 * Z + A2 * X2  (mod 2^46). 
   * --------------------------------------------------------------------
   */
  T1 = R23 * *X;
  j  = T1;
  X1 = j;
  X2 = *X - T23 * X1;
  T1 = A1 * X2 + A2 * X1;

  j  = R23 * T1;
  T2 = j;
  Z = T1 - T23 * T2;
  T3 = T23 * Z + A2 * X2;
  j  = R46 * T3;
  T4 = j;
  *X = T3 - T46 * T4;

  return(R46 * *X);
}

__device__ double find_my_seed_device(
    int kn,
    int np,
    long nn,
    double s,
    double a)
{
  double t1,t2;
  long mq,nq,kk,ik;

  if(kn==0) return s;

  mq = (nn/4 + np - 1) / np;
  nq = mq * 4 * kn;

  t1 = s;
  t2 = a;
  kk = nq;
  while(kk > 1){
    ik = kk / 2;
    if(2*ik==kk){
      (void)randlc_device(&t2, &t2);
      kk = ik;
    }else{
      (void)randlc_device(&t1, &t2);
      kk = kk - 1;
    }
  }
  (void)randlc_device(&t1, &t2);

  return(t1);
}

__global__ void create_seq_gpu_kernel(
    int* key_array,
    double seed,
    double a,
    int number_of_blocks,
    int amount_of_work)
{
  double x, s;
  double an = a;
  int i, k;
  int k1, k2;
  int myid, num_procs;
  int mq;

  myid = blockIdx.x*blockDim.x+threadIdx.x;
  num_procs = amount_of_work;

  mq = (NUM_KEYS + num_procs - 1) / num_procs;
  k1 = mq * myid;
  k2 = k1 + mq;
  if(k2 > NUM_KEYS) k2 = NUM_KEYS;

  s = find_my_seed_device(myid, num_procs, (long)4*NUM_KEYS, seed, an);

  k = MAX_KEY/4;

  for(i=k1; i<k2; i++){
    x = randlc_device(&s, &an);
    x += randlc_device(&s, &an);
    x += randlc_device(&s, &an);
    x += randlc_device(&s, &an);  
    key_array[i] = k*x;
  }
}

__global__ void full_verify_gpu_kernel_1(
    const int*__restrict__ key_array,
    int*__restrict__ key_buff2,
    int number_of_blocks,
    int amount_of_work)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  key_buff2[i] = key_array[i];
}

__global__ void full_verify_gpu_kernel_2(
    const int* __restrict__ key_buff2,
    int*__restrict__  key_buff_ptr_global,
    int*__restrict__  key_array,
    int number_of_blocks,
    int amount_of_work)
{    
  int value = key_buff2[blockIdx.x*blockDim.x+threadIdx.x];
  int index = atomicAdd(&key_buff_ptr_global[value], -1) - 1;
  key_array[index] = value;
}

__global__ void full_verify_gpu_kernel_3(
    const int*__restrict__ key_array,
    int*__restrict__ global_aux,
    int number_of_blocks,
    int amount_of_work)
{
  extern __shared__ int shared_data[];

  int i = (blockIdx.x*blockDim.x+threadIdx.x) + 1;

  if(i < NUM_KEYS){
    if(key_array[i-1] > key_array[i]) 
      shared_data[threadIdx.x]=1;
    else
      shared_data[threadIdx.x]=0;
  }else
    shared_data[threadIdx.x]=0;

  __syncthreads();

  for(i=blockDim.x/2; i>0; i>>=1) {
    if(threadIdx.x<i)
      shared_data[threadIdx.x] += shared_data[threadIdx.x+i];
    __syncthreads();
  }

  if(threadIdx.x==0) global_aux[blockIdx.x]=shared_data[0];
}


__global__ void rank_gpu_kernel_1(
    int*__restrict__ key_array,
    int*__restrict__ partial_verify_vals,
    const int*__restrict__ test_index_array,
    int iteration,
    int number_of_blocks,
    int amount_of_work)
{
  key_array[iteration] = iteration;
  key_array[iteration+MAX_ITERATIONS] = MAX_KEY - iteration;
  /*
   * --------------------------------------------------------------------
   * determine where the partial verify test keys are, 
   * --------------------------------------------------------------------
   * load into top of array bucket_size  
   * --------------------------------------------------------------------
   */
#pragma unroll
  for(int i=0; i<TEST_ARRAY_SIZE; i++){
    partial_verify_vals[i] = key_array[test_index_array[i]];
  }
}

__global__ void rank_gpu_kernel_2(
    int* key_buff1,
    int number_of_blocks,
    int amount_of_work)
{
  key_buff1[blockIdx.x*blockDim.x+threadIdx.x] = 0;
}

__global__ void rank_gpu_kernel_3(
    int*__restrict__ key_buff_ptr,
    const int*__restrict__ key_buff_ptr2,
    int number_of_blocks,
    int amount_of_work)
{
  /*
   * --------------------------------------------------------------------
   * in this section, the keys themselves are used as their 
   * own indexes to determine how many of each there are: their
   * individual population  
   * --------------------------------------------------------------------
   */
  atomicAdd(&key_buff_ptr[key_buff_ptr2[blockIdx.x*blockDim.x+threadIdx.x]], 1);
}

__global__ void rank_gpu_kernel_4(
    const int*__restrict__ source,
    int*__restrict__ destiny,
    int*__restrict__ sum,
    int number_of_blocks,
    int amount_of_work)
{
  extern __shared__ int shared_data[];

  shared_data[threadIdx.x] = 0;
  int position = blockDim.x + threadIdx.x;

  int factor = MAX_KEY / number_of_blocks;
  int start = factor * blockIdx.x;
  int end = start + factor;

  for(int i=start; i<end; i+=blockDim.x){
    shared_data[position] = source[i + threadIdx.x];

    for(uint offset=1; offset<blockDim.x; offset<<=1){
      __syncthreads();
      int t = shared_data[position] + shared_data[position - offset];
      __syncthreads();
      shared_data[position] = t;
    }

    int prv_val = (i == start) ? 0 : destiny[i - 1];
    destiny[i + threadIdx.x] = shared_data[position] + prv_val;
  }

  __syncthreads();
  if(threadIdx.x==0) sum[blockIdx.x] = destiny[end-1];
}

__global__ void rank_gpu_kernel_5(
    const int*__restrict__ source,
    int*__restrict__ destiny,
    int number_of_blocks,
    int amount_of_work)
{
  extern __shared__ int shared_data[];

  shared_data[threadIdx.x] = 0;
  int position = blockDim.x + threadIdx.x;
  shared_data[position] = source[threadIdx.x];

  for(uint offset=1; offset<blockDim.x; offset<<=1) {
    __syncthreads();
    int t = shared_data[position] + shared_data[position - offset];
    __syncthreads();
    shared_data[position] = t;
  }

  __syncthreads();

  destiny[threadIdx.x] = shared_data[position - 1];
}

__global__ void rank_gpu_kernel_6(
    const int*__restrict__ source,
    int*__restrict__ destiny,
    const int*__restrict__ offset,
    int number_of_blocks,
    int amount_of_work)
{
  int factor = MAX_KEY / number_of_blocks;
  int start = factor * blockIdx.x;
  int end = start + factor;
  int sum = offset[blockIdx.x];
  for(int i=start; i<end; i+=blockDim.x)
    destiny[i + threadIdx.x] = source[i + threadIdx.x] + sum;
}

__global__ void rank_gpu_kernel_7(
    const int*__restrict__ partial_verify_vals,
    const int*__restrict__ key_buff_ptr,
    const int*__restrict__ test_rank_array,
    int*__restrict__ passed_verification_device,
    int iteration,
    int number_of_blocks,
    int amount_of_work)
{
  /*
   * --------------------------------------------------------------------
   * this is the partial verify test section 
   * observe that test_rank_array vals are
   * shifted differently for different cases
   * --------------------------------------------------------------------
   */
  int i, k;
  int passed_verification = 0;
  for(i=0; i<TEST_ARRAY_SIZE; i++){  
    /* test vals were put here on partial_verify_vals */                                           
    k = partial_verify_vals[i];          
    if(0<k && k<=NUM_KEYS-1){
      int key_rank = key_buff_ptr[k-1];
      int failed = 0;
      switch(CLASS){
        case 'S':
          if(i<=2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'W':
          if(i<2){
            if(key_rank != test_rank_array[i]+(iteration-2))
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'A':
          if(i<=2){
            if(key_rank != test_rank_array[i]+(iteration-1))
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-(iteration-1))
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'B':
          if(i==1 || i==2 || i==4){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }
          else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'C':
          if(i<=2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
        case 'D':
          if(i<2){
            if(key_rank != test_rank_array[i]+iteration)
              failed = 1;
            else
              passed_verification++;
          }else{
            if(key_rank != test_rank_array[i]-iteration)
              failed = 1;
            else
              passed_verification++;
          }
          break;
      }
      if(failed==1){
        printf("Failed partial verification: iteration %d, test key %d\n", iteration, (int)i);
      }
    }
  }
  *passed_verification_device += passed_verification;
}

