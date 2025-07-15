//  dc3 algorithm on GPU
//
//  Created by gangliao on 12/22/14.
//  Copyright (c) 2014 gangliao. All rights reserved.

__device__ int leq(int a1, int a2, int b1, int b2) {
  return (a1 < b1 || (a1 == b1 && a2 <= b2));
}

__device__ int leq2(int a1, int a2, int a3, int b1, int b2, int b3) {
  return (a1 < b1 || (a1 == b1 && leq(a2, a3, b2, b3)));
}

__global__ void Init_d_s12(int* s12, int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n) return;
  s12[index] = index + index / 2 + 1;
}

//d_SA12, d_s12, n02
__global__
void keybits(      int*__restrict__ SA12,
             const int*__restrict__ s12,
             const int*__restrict__ s,
             int n, int i)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n) return;
  SA12[index] = s[s12[index] + i];
}

__global__
void InitScan(const int*__restrict__ s,
              const int*__restrict__ SA12,
                    int*__restrict__ scan,
              int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n) return;
  if ((s[SA12[index]] == s[SA12[index + 1]]) && 
      (s[SA12[index] + 1] == s[SA12[index + 1] + 1]) &&
      (s[SA12[index] + 2] == s[SA12[index + 1] + 2]))
  {
    scan[index] = 0;
  }
  else
    scan[index] = 1;
}

__global__
void Set_suffix_rank(      int*__restrict__ s12,
                     const int*__restrict__ SA12,
                     const int*__restrict__ scan,
                     int n02, int n0)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n02) return;
  s12[SA12[index] / 3 + ((SA12[index] % 3) - 1) * n0] = scan[index] + 1;
}

__global__
void Store_unique_ranks(      int*__restrict__ s12,
                        const int*__restrict__ SA12,
                        int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n) return;
  s12[SA12[index]] = index + 1;
}

__global__
void Compute_SA_From_UniqueRank(const int*__restrict__ s12,
                                      int*__restrict__ SA12,
                                int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n) return;
  SA12[s12[index] - 1] = index;
}

__global__
void InitScan2(const int*__restrict__ SA12,
                     int*__restrict__ scan,
               int n0, int n02)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n02)
    return;
  if (SA12[index] < n0)
    scan[index] = 1;
  else
    scan[index] = 0;
}

__global__
void Set_S0(      int*__restrict__ s0,
            const int*__restrict__ SA12,
            const int*__restrict__ scan,
            int n0, int n02)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  if (index >= n02) return;
  if (SA12[index] < n0)
    s0[scan[index]] = 3 * SA12[index];
}

__global__
void merge_suffixes(const int*__restrict__ SA0,
                    const int*__restrict__ SA12,
                          int*__restrict__ SA,
                    const int*__restrict__ s,
                    const int*__restrict__ s12,
                    int n0, int n02, int n)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  int left, right, mid;
  int flag = 0;
  if (index >= n0 + n02) return;

  if (n != n0 + n02)
  {
    flag = 1;
    if (index == n0) return;  
  }

  int i, j;
  if (index < n0)
  {
    i = SA0[index];
    left = n0;
    right = n0 + n02;

    while (left < right)
    {
      mid = (left + right) / 2;
      if (SA12[mid - n0] < n0)
      {
        j = SA12[mid - n0] * 3 + 1;

        if (leq(s[j], s12[(j + 1) / 3 + ((j + 1) % 3 - 1)*n0],
                s[i], s12[i / 3]))
          left = mid + 1;
        else
          right = mid;
      }
      else
      {
        j = (SA12[mid - n0] - n0) * 3 + 2;

        if (leq2(s[j], s[j + 1], s12[(j + 2) / 3 + ((j + 2) % 3 - 1)*n0],
                 s[i], s[i + 1], s12[i / 3 + n0]))
          left = mid + 1;
        else
          right = mid;
      }

    }
    SA[index + left - n0 - flag] = i;
  }
  else
  {
    if (SA12[index - n0] < n0)
    {
      i = SA12[index - n0] * 3 + 1;
      left = 0;
      right = n0;
      while (left < right)
      {
        mid = (left + right) / 2;

        if (leq(s[SA0[mid]], s12[SA0[mid] / 3], s[i],
                s12[(i + 1) / 3 + ((i + 1) % 3 - 1)*n0]))
          left = mid + 1;
        else
          right = mid;
      }
      SA[index - n0 + left - flag] = i;
    }
    else
    {
      i = (SA12[index - n0] - n0) * 3 + 2;
      left = 0;
      right = n0;
      while (left < right)
      {
        mid = (left + right) / 2;

        if (leq2(s[SA0[mid]], s[SA0[mid] + 1], s12[SA0[mid] / 3 + n0],
                 s[i], s[i + 1], s12[(i + 2) / 3 + ((i + 2) % 3 - 1)*n0]))
          left = mid + 1;
        else
          right = mid;
      }
      SA[index - n0 + left - flag] = i;
    }
  }
}
