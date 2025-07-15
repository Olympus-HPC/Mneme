/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 * 
 * Tridiagonal solvers.
 * Device code for cyclic reduction (CR).
 *
 * Original CUDA kernel: UC Davis, Yao Zhang & John Owens, 2009
 * 
 * NVIDIA, Nikolai Sakharnykh, 2009
 */


__global__ void cyclic_small_systems_kernel(
    const float*__restrict__ a_d, 
    const float*__restrict__ b_d, 
    const float*__restrict__ c_d, 
    const float*__restrict__ d_d, 
          float*__restrict__ x_d, 
    const int system_size, 
    const int num_systems, 
    const int iterations)
{
  extern __shared__ float shared[];

  int thid = threadIdx.x;
  int blid = blockIdx.x;

  int stride = 1;
  int half_size = system_size >> 1;
  int thid_num = half_size;

  float* a = shared;
  float* b = &a[system_size];
  float* c = &b[system_size];
  float* d = &c[system_size];
  float* x = &d[system_size];

  a[thid] = a_d[thid + blid * system_size];
  a[thid + thid_num] = a_d[thid + thid_num + blid * system_size];

  b[thid] = b_d[thid + blid * system_size];
  b[thid + thid_num] = b_d[thid + thid_num + blid * system_size];

  c[thid] = c_d[thid + blid * system_size];
  c[thid + thid_num] = c_d[thid + thid_num + blid * system_size];

  d[thid] = d_d[thid + blid * system_size];
  d[thid + thid_num] = d_d[thid + thid_num + blid * system_size];

  __syncthreads();

  // forward elimination
  for (int j = 0; j < iterations; j++)
  {
    __syncthreads();

    stride <<= 1;
    int delta = stride >> 1;
    if (thid < thid_num)
    { 
      int i = stride * thid + stride - 1;

      if (i == system_size - 1)
      {
#ifndef NATIVE_DIVIDE
        float tmp = a[i] / b[i-delta];
#else
        float tmp = __fdiv_rn(a[i], b[i-delta]);
#endif
        b[i] = b[i] - c[i-delta] * tmp;
        d[i] = d[i] - d[i-delta] * tmp;
        a[i] = -a[i-delta] * tmp;
        c[i] = 0;      
      }
      else
      {
#ifndef NATIVE_DIVIDE
        float tmp1 = a[i] / b[i-delta];
        float tmp2 = c[i] / b[i+delta];
#else
        float tmp1 = __fdiv_rn(a[i], b[i-delta]);
        float tmp2 = __fdiv_rn(c[i], b[i+delta]);
#endif
        b[i] = b[i] - c[i-delta] * tmp1 - a[i+delta] * tmp2;
        d[i] = d[i] - d[i-delta] * tmp1 - d[i+delta] * tmp2;
        a[i] = -a[i-delta] * tmp1;
        c[i] = -c[i+delta] * tmp2;
      }
    }
    thid_num >>= 1;
  }

  if (thid < 2)
  {
    int addr1 = stride - 1;
    int addr2 = (stride << 1) - 1;
    float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
#ifndef NATIVE_DIVIDE
    x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
    x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
#else
    x[addr1] = __fdiv_rn((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);
    x[addr2] = __fdiv_rn((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);
#endif
  }

  // backward substitution
  thid_num = 2;
  for (int j = 0; j < iterations; j++)
  {
    int delta = stride >> 1;
    __syncthreads();
    if (thid < thid_num)
    {
      int i = stride * thid + (stride >> 1) - 1;
#ifndef NATIVE_DIVIDE
      if (i == delta - 1)
        x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
      else
        x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
#else
      if (i == delta - 1)
        x[i] = __fdiv_rn((d[i] - c[i] * x[i+delta]), b[i]);
      else
        x[i] = __fdiv_rn((d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]), b[i]);
#endif
    }
    stride >>= 1;
    thid_num <<= 1;
  }

  __syncthreads();   

  x_d[thid + blid * system_size] = x[thid];
  x_d[thid + half_size + blid * system_size] = x[thid + half_size];
}

__global__ void cyclic_branch_free_kernel(
    const float*__restrict__ a_d, 
    const float*__restrict__ b_d, 
    const float*__restrict__ c_d, 
    const float*__restrict__ d_d, 
          float*__restrict__ x_d, 
    const int system_size, 
    const int num_systems, 
    const int iterations)
{
  extern __shared__ float shared[];

  int thid = threadIdx.x;
  int blid = blockIdx.x;

  int stride = 1;
  int half_size = system_size >> 1;
  int thid_num = half_size;

  float* a = shared;
  float* b = &a[system_size];
  float* c = &b[system_size];
  float* d = &c[system_size];
  float* x = &d[system_size];

  a[thid] = a_d[thid + blid * system_size];
  a[thid + thid_num] = a_d[thid + thid_num + blid * system_size];

  b[thid] = b_d[thid + blid * system_size];
  b[thid + thid_num] = b_d[thid + thid_num + blid * system_size];

  c[thid] = c_d[thid + blid * system_size];
  c[thid + thid_num] = c_d[thid + thid_num + blid * system_size];

  d[thid] = d_d[thid + blid * system_size];
  d[thid + thid_num] = d_d[thid + thid_num + blid * system_size];

  __syncthreads();

  // forward elimination
  for (int j = 0; j < iterations; j++)
  {
    __syncthreads();

    stride <<= 1;
    int delta = stride >> 1;
    if (thid < thid_num)
    { 
      int i = stride * thid + stride - 1;
      int iRight = i+delta;
      iRight = iRight & (system_size-1);
#ifndef NATIVE_DIVIDE
      float tmp1 = a[i] / b[i-delta];
      float tmp2 = c[i] / b[iRight];
#else
      float tmp1 = __fdiv_rn(a[i], b[i-delta]);
      float tmp2 = __fdiv_rn(c[i], b[iRight]);
#endif
      b[i] = b[i] - c[i-delta] * tmp1 - a[iRight] * tmp2;
      d[i] = d[i] - d[i-delta] * tmp1 - d[iRight] * tmp2;
      a[i] = -a[i-delta] * tmp1;
      c[i] = -c[iRight]  * tmp2;
    }

    thid_num >>= 1;
  }

  if (thid < 2)
  {
    int addr1 = stride - 1;
    int addr2 = (stride << 1) - 1;
    float tmp3 = b[addr2] * b[addr1] - c[addr1] * a[addr2];
#ifndef NATIVE_DIVIDE
    x[addr1] = (b[addr2] * d[addr1] - c[addr1] * d[addr2]) / tmp3;
    x[addr2] = (d[addr2] * b[addr1] - d[addr1] * a[addr2]) / tmp3;
#else
    x[addr1] = __fdiv_rn((b[addr2] * d[addr1] - c[addr1] * d[addr2]), tmp3);
    x[addr2] = __fdiv_rn((d[addr2] * b[addr1] - d[addr1] * a[addr2]), tmp3);
#endif
  }

  // backward substitution
  thid_num = 2;
  for (int j = 0; j < iterations; j++)
  {
    int delta = stride >> 1;
    __syncthreads();
    if (thid < thid_num)
    {
      int i = stride * thid + (stride >> 1) - 1;
#ifndef NATIVE_DIVIDE
      if (i == delta - 1)
        x[i] = (d[i] - c[i] * x[i+delta]) / b[i];
      else
        x[i] = (d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]) / b[i];
#else
      if (i == delta - 1)
        x[i] = __fdiv_rn((d[i] - c[i] * x[i+delta]), b[i]);
      else
        x[i] = __fdiv_rn((d[i] - a[i] * x[i-delta] - c[i] * x[i+delta]), b[i]);
#endif
    }
    stride >>= 1;
    thid_num <<= 1;
  }

  __syncthreads();   

  x_d[thid + blid * system_size] = x[thid];
  x_d[thid + half_size + blid * system_size] = x[thid + half_size];
}

