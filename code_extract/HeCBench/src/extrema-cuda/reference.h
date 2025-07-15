// reference clip
void clip_both ( const bool &clip, const int &n, int &plus, int &minus ) {
  if ( clip ) {
    if ( plus >= n ) {
      plus = n - 1;
    }
    if ( minus < 0 ) {
      minus = 0;
    }
  } else {
    if ( plus >= n ) {
      plus -= n;
    }
    if ( minus < 0 ) {
      minus += n;
    }
  }
}

template<typename T>
void cpu_relextrema_1D(
  const int  n,
  const int  order,
  const bool clip,
  const T *__restrict__ inp,
  bool *__restrict__ results)
{
  for ( int tid = 0; tid < n; tid++ ) {

    const T data = inp[tid];
    bool    temp = true;

    for ( int o = 1; o < ( order + 1 ); o++ ) {
      int plus = tid + o;
      int minus = tid - o;

      clip_both( clip, n, plus, minus );

      temp &= data > inp[plus];
      temp &= data >= inp[minus];
    }
    results[tid] = temp;
  }
}

template<typename T>
void cpu_relextrema_2D(
  const int  in_x,
  const int  in_y,
  const int  order,
  const bool clip,
  const int  axis,
  const T *__restrict__ inp,
  bool *__restrict__ results) 
{
  for (int tx = 0; tx < in_y; tx++)
    for (int ty = 0; ty < in_x; ty++) {

      int tid = tx * in_x + ty ;

      const T data = inp[tid] ;
      bool    temp = true ;

      for ( int o = 1; o < ( order + 1 ); o++ ) {

        int plus;
        int minus;

        if ( axis == 0 ) {
          plus  = tx + o;
          minus = tx - o;

          clip_both( clip, in_y, plus, minus );

          plus  = plus * in_x + ty;
          minus = minus * in_x + ty;
        } else {
          plus  = ty + o;
          minus = ty - o;

          clip_both( clip, in_x, plus, minus );

          plus  = tx * in_x + plus;
          minus = tx * in_x + minus;
        }

        temp &= data > inp[plus] ;
        temp &= data >= inp[minus] ;
      }
      results[tid] = temp;
    }
}

