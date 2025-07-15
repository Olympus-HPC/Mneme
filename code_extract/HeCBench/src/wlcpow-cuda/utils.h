typedef float r32;
typedef double r64;
typedef int i32;

__inline__ __device__
float minimum_image( float dr, float p )
{
  float p_half = p * 0.5f;
  return dr + ( dr > -p_half ? ( dr < p_half ? 0.f : -p ) : p );
}

#define _LN_2           6.9314718055994528623E-1
#define _2_TO_MINUS_31  4.6566128730773925781E-10
#define _2_TO_MINUS_32  2.3283064365386962891E-10
#define _TEA_K0      0xA341316C
#define _TEA_K1      0xC8013EA4
#define _TEA_K2      0xAD90777D
#define _TEA_K3      0x7E95761E
#define _TEA_DT      0x9E3779B9

template<typename T>
__inline__ __device__
T bound( T x, T lower, T upper )
{
  return max( lower, min( x, upper ) );
}

template<int N> __inline__ __device__
void __TEA_core( uint &v0, uint &v1, uint sum = 0 )
{
  sum += _TEA_DT;
  v0 += ( ( v1 << 4 ) + _TEA_K0 ) ^ ( v1 + sum ) ^ ( ( v1 >> 5 ) + _TEA_K1 );
  v1 += ( ( v0 << 4 ) + _TEA_K2 ) ^ ( v0 + sum ) ^ ( ( v0 >> 5 ) + _TEA_K3 );
  __TEA_core < N - 1 > ( v0, v1, sum );
}

template<> __inline__ __device__
void __TEA_core<0>( uint &v0, uint &v1, uint sum ) {}

template<int N> __inline__ __device__
float gaussian_TEA_fast( bool pred, int u, int v )
{
  uint v0 =  pred ? u : v;
  uint v1 = !pred ? u : v;
  __TEA_core<N>( v0, v1 );
  float f = sinpif( int( v0 ) * float(_2_TO_MINUS_31) );
  float r = sqrtf( -2.0f * float(_LN_2) * log2f( v1 * float(_2_TO_MINUS_32) ) );
  return bound( r * f, -4.0f, 4.0f );
}
