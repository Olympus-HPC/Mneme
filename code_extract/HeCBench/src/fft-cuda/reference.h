template <int SIZE>
void fft1D_512_reference (T2* work, const int n_ffts)
{
  const int reversed[] = {0,4,2,6,1,5,3,7};
  for (int n = 0; n < n_ffts; n++) {
    T smem[9*64];
    T2 buffer[8*SIZE];
    T2 *data;
    int i, j, gid, tid, hi, lo;
    for (tid = 0; tid < SIZE; tid++) {
      gid = n * 512 + tid;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i] = work[gid+i*64];

      FFT8( data );

      for(j = 1; j < 8; j++ ) {
        data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)512)*tid) );
      }
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].x;
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x = smem[lo*66+hi+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*66] = data[reversed[i]].y;
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y= smem[lo*66+hi+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      data = buffer + tid * 8;
      FFT8( data );
      for(j = 1; j < 8; j++ ) {
        data[j] = cmplx_mul( data[j],exp_i(((T)-2*(T)M_PI*reversed[j]/(T)64)*hi) );
      }
    }

    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].x;
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].x = smem[hi*72+lo+i*8];
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) smem[hi*8+lo+i*72] = data[reversed[i]].y;
    }
    for (tid = 0; tid < SIZE; tid++) {
      hi = tid>>3;
      lo = tid&7;
      data = buffer + tid * 8;
      for(i = 0; i < 8; i++ ) data[i].y= smem[hi*72+lo+i*8];
    }

    for (tid = 0; tid < SIZE; tid++) {
      data = buffer + tid * 8;
      FFT8( data );
      for(i = 0; i < 8; i++ ) {
        gid = n * 512 + tid;
        work[gid+i*64] = data[reversed[i]];
      }
    }
  }
}
