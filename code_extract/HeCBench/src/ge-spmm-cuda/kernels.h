#ifdef __HIPCC__
#define __syncwarp() 
#endif

template<typename T>
__global__ void spmm_test1(
  int A_nrows, int B_ncols,
  const int*__restrict__ A_csrRowPtr,
  const int*__restrict__ A_csrColInd,
  const T*__restrict__ A_csrVal,
  const T*__restrict__ B_dnVal,
        T*__restrict__ C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  T *val_sh = (T *)&sh[(blockDim.y<<5)];
  int shmem_offset = (threadIdx.y<<5);
  int thread_idx = shmem_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;

  if (rid<A_nrows) {
    int cid = (blockIdx.y<<5)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx.x;
    int offset;
    T acc=0;

    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
        }
        __syncwarp();
      }
      C_dnVal[(rid*B_ncols+cid)] = acc;
    }
    else {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (cid<B_ncols) {
            acc += val_sh[(shmem_offset+kk)]*B_dnVal[offset];
          }
        }
        __syncwarp();
      }
      if (cid<B_ncols) {
        C_dnVal[(rid*B_ncols+cid)] = acc;
      }
    }
  }
}

template<typename T>
__global__ void spmm_test2(
  int A_nrows, int B_ncols,
  const int*__restrict__ A_csrRowPtr,
  const int*__restrict__ A_csrColInd,
  const T*__restrict__ A_csrVal,
  const T*__restrict__ B_dnVal,
        T*__restrict__ C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  T *val_sh = (T *)&sh[(blockDim.y<<5)];
  int shmem_offset = (threadIdx.y<<5);
  int thread_idx = shmem_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;

  if (rid<A_nrows) {
    int cid = (blockIdx.y<<6)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx.x;
    int offset;
    T acc1=0, acc2=0, val;

    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
    }
  }
}

template<typename T>
__global__ void spmm_test3(
  int A_nrows, int B_ncols,
  const int*__restrict__ A_csrRowPtr,
  const int*__restrict__ A_csrColInd,
  const T*__restrict__ A_csrVal,
  const T*__restrict__ B_dnVal,
        T*__restrict__ C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  T *val_sh = (T *)&sh[(blockDim.y<<5)];
  int shmem_offset = (threadIdx.y<<5);
  int thread_idx = shmem_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;

  if (rid<A_nrows) {
    int cid = (blockIdx.y<<7)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx.x;
    int offset;
    T acc1=0, acc2=0, acc3=0, acc4=0, val;

    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
          acc3 += val*B_dnVal[offset+64];
          acc4 += val*B_dnVal[offset+96];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
      C_dnVal[offset+64] = acc3;
      C_dnVal[offset+96] = acc4;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
          if (nout>2) {
            acc3 += val*B_dnVal[offset+64];
          }
          if (nout>3) {
            acc4 += val*B_dnVal[offset+96];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
      if (nout>2) {
        C_dnVal[(offset+64)] = acc3;
      }
      if (nout>3) {
        C_dnVal[(offset+96)] = acc4;
      }
    }
  }
}

template<typename T>
__global__ void spmm_test4(
  int A_nrows, int B_ncols,
  const int*__restrict__ A_csrRowPtr,
  const int*__restrict__ A_csrColInd,
  const T*__restrict__ A_csrVal,
  const T*__restrict__ B_dnVal,
        T*__restrict__ C_dnVal)
{
  extern __shared__ int sh[];
  int *colInd_sh = sh;
  T *val_sh = (T *)&sh[(blockDim.y<<5)];
  int shmem_offset = (threadIdx.y<<5);
  int thread_idx = shmem_offset+threadIdx.x;

  int rid = blockDim.y*blockIdx.x+threadIdx.y;

  if (rid<A_nrows) {
    int cid = (blockIdx.y<<8)+threadIdx.x;
    int lb = A_csrRowPtr[rid];
    int hb = A_csrRowPtr[(rid+1)];
    int ptr = lb+threadIdx.x;
    int offset;
    T acc1=0, acc2=0, acc3=0, acc4=0, acc5=0,acc6=0,acc7=0,acc8=0,val;

    if (blockIdx.y != gridDim.y-1) {
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          val = val_sh[(shmem_offset+kk)];
          acc1 += val*B_dnVal[offset];
          acc2 += val*B_dnVal[offset+32];
          acc3 += val*B_dnVal[offset+64];
          acc4 += val*B_dnVal[offset+96];
          acc5 += val*B_dnVal[offset+128];
          acc6 += val*B_dnVal[offset+160];
          acc7 += val*B_dnVal[offset+192];
          acc8 += val*B_dnVal[offset+224];
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      C_dnVal[offset] = acc1;
      C_dnVal[offset+32] = acc2;
      C_dnVal[offset+64] = acc3;
      C_dnVal[offset+96] = acc4;
      C_dnVal[offset+128] = acc5;
      C_dnVal[offset+160] = acc6;
      C_dnVal[offset+192] = acc7;
      C_dnVal[offset+224] = acc8;
    }
    else {
      int nout = (B_ncols-cid+31)/32;
      for (int jj=lb; jj<hb; jj+=32) {
        if (ptr<hb) {
          val_sh[thread_idx] = A_csrVal[ptr];
          colInd_sh[thread_idx] = B_ncols*A_csrColInd[ptr];
        }
        __syncwarp();
        ptr += 32;

        for (int kk=0; kk<32&&jj+kk<hb; kk++) {
          val = val_sh[(shmem_offset+kk)];
          offset = colInd_sh[(shmem_offset+kk)] + cid;
          if (nout>0) {
            acc1 += val*B_dnVal[offset];
          }
          if (nout>1) {
            acc2 += val*B_dnVal[offset+32];  
          }
          if (nout>2) {
            acc3 += val*B_dnVal[offset+64];
          }
          if (nout>3) {
            acc4 += val*B_dnVal[offset+96];  
          }
          if (nout>4) {
            acc5 += val*B_dnVal[offset+128];  
          }
          if (nout>5) {
            acc6 += val*B_dnVal[offset+160];  
          }
          if (nout>6) {
            acc7 += val*B_dnVal[offset+192];  
          }
          if (nout>7) {
            acc8 += val*B_dnVal[offset+224];  
          }
        }
        __syncwarp();
      }
      offset = rid*B_ncols+cid;
      if (nout>0) {
        C_dnVal[offset] = acc1;
      }
      if (nout>1) {
        C_dnVal[(offset+32)] = acc2;
      }
      if (nout>2) {
        C_dnVal[(offset+64)] = acc3;
      }
      if (nout>3) {
        C_dnVal[(offset+96)] = acc4;
      }
      if (nout>4) {
        C_dnVal[(offset+128)] = acc5;
      }
      if (nout>5) {
        C_dnVal[(offset+160)] = acc6;
      }
      if (nout>6) {
        C_dnVal[(offset+192)] = acc7;
      }
      if (nout>7) {
        C_dnVal[(offset+224)] = acc8;
      }
    }
  }
}

