__global__ void fft1D_512 (T2* work) 
{
  __shared__  T smem[8*8*9];

  int tid = threadIdx.x;
  int gid = blockIdx.x * 512 + tid;
  int hi = tid>>3;
  int lo = tid&7;
  T2 data[8];
  //__local T smem[8*8*9];
  const int reversed[] = {0,4,2,6,1,5,3,7};

  // starting index of data to/from global memory
  // globalLoads8(data, work, 64)
  for( int i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

  FFT8( data );

  //twiddle8( data, tid, 512 );
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) ); 
  }                                                                   

  //transpose(data, &smem[hi*8+lo], 66, &smem[lo*66+hi], 8, 0xf);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8]; 
  __syncthreads(); 

  FFT8( data );

  //twiddle8( data, hi, 64 );
  for( int j = 1; j < 8; j++ ){                                       
      data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) ); 
  }                                                                   

  //transpose(data, &smem[hi*8+lo], 8*9, &smem[hi*8*9+lo], 8, 0xE);
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8]; 
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
  __syncthreads(); 
  for( int i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8]; 

  FFT8( data );

  //globalStores8(data, work, 64);
  for( int i = 0; i < 8; i++ )
    work[gid+i*64] = data[reversed[i]];
}
