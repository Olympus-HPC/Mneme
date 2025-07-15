__device__ __forceinline__
void clip_plus( const bool &clip, const int &n, int &plus ) {
  if ( plus >= n ) plus = clip ? n - 1 : plus - n;
}

__device__ __forceinline__
void clip_minus( const bool &clip, const int &n, int &minus ) {
  if ( minus < 0 ) minus = clip ? 0 : minus + n;
}

///////////////////////////////////////////////////////////////////////////////
//                          BOOLRELEXTREMA 1D                                //
///////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void relextrema_1D(
  const int  n,
  const int  order,
  const bool clip,
  const T *__restrict__ inp,
  bool *__restrict__ results)
{
  const int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  for ( int tid = tx; tid < n; tid += stride ) {

    const T data = inp[tid];
    bool    temp = true;

    for ( int o = 1; o <= order; o++ ) {
      int plus = tid + o;
      int minus = tid - o;

      clip_plus( clip, n, plus );
      clip_minus( clip, n, minus );

      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}


template<typename T>
__global__ void relextrema_2D(
  const int  in_x,
  const int  in_y,
  const int  order,
  const bool clip,
  const int  axis,
  const T *__restrict__ inp,
  bool *__restrict__ results) 
{
  const int ty = blockIdx.x * blockDim.x + threadIdx.x;
  const int tx = blockIdx.y * blockDim.y + threadIdx.y;

  if ( ( tx < in_y ) && ( ty < in_x ) ) {
    int tid = tx * in_x + ty ;

    const T data = inp[tid] ;
    bool    temp = true ;

    for ( int o = 1; o <= order; o++ ) {

      int plus;
      int minus;

      if ( axis == 0 ) {
        plus  = tx + o;
        minus = tx - o;

        clip_plus( clip, in_y, plus );
        clip_minus( clip, in_y, minus );

        plus  = plus * in_x + ty;
        minus = minus * in_x + ty;
      } else {
        plus  = ty + o;
        minus = ty - o;

        clip_plus( clip, in_x, plus );
        clip_minus( clip, in_x, minus );

        plus  = tx * in_x + plus;
        minus = tx * in_x + minus;
      }

      temp &= data > inp[plus] ;
      temp &= data >= inp[minus] ;
    }
    results[tid] = temp;
  }
}
