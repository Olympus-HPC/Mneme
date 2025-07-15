#define ESS __host__ __device__

/*
  Description: Finds t and s such that a*t + b*s = gcd(a, b)
  Output: return gcd(a, b).
*/
ESS
int ext_euclidian_alg(int a, int b, int& x, int& y) 
{ 
  x = 1, y = 0;
  int x1 = 0, y1 = 1, a1 = a, b1 = b;
  int s, t;
  while (b1) {
    int q = a1 / b1;
    s = x1; t = x - q * x1;
    x = s; x1 = t;
    s = y1; t = y - q * y1;
    y = s; y1 = t;
    s = b1; t = a1 - q * b1;
    a1 = s; b1 = t;
  }
  return a1;
} 

/*
  Input:  integer to make positive mod m (a)
          modulus (m)
  Output: return positive integer congruent to a mod m
*/
ESS
unsigned int make_positive(int a, unsigned int m) {
  while(a < 0) {
    a += m;
  }
  return a % m;
}

/*
  Input:  integer to find inverse of (a),
          modulus (m)
  Output: return inverse of a mod m.
*/
ESS
int find_inverse(int a, unsigned int m) 
{ 
  int t, s; 
  ext_euclidian_alg(a, m, t, s);
  return make_positive(t, m); 
} 

/*
  Input:  modulus (m)
          first point (x1, y1)
          second point (x2, y2)
  Output: sum of the input points (x3, y3)
  note: non-commutative (x1,y1) + (x2,y2) != (x2,y2) + (x1,y1)
*/
ESS
void point_addition(unsigned int m, int x1, int y1, 
                    int x2, int y2, 
                    int *x3, int *y3) 
{
  int temp = make_positive(x2 - x1, m);
  int slope = make_positive((y2 - y1) * find_inverse(temp, m), m);
  *x3 = make_positive((slope*slope - x1 - x2), m);
  *y3 = make_positive((slope * (x1 - *x3) - y1), m);
}

/*
  Input:  modulus (m)
          point (x1, y1)
  Output: sum of the input point (x3, y3)
*/
ESS
void point_doubling(unsigned int m, int a, int x1, int y1, int *x3, int *y3) 
{
  int slope = (3 * x1 * x1 + a) * find_inverse(2 * y1, m);
  *x3 = make_positive(slope * slope - 2 * x1, m);
  *y3 = make_positive(slope * (x1 - *x3) - y1, m);
}

/*
  Input:  integer to check (n)
  Output: return index of first set bit starting from the left
*/
ESS
int first_set_bit(int n)
{
  int i;
  for(i=(sizeof(int)*8)-1; i>=0; --i)
  {
    if( ((1 << i) & n) )
      return i;
  }
  return 0;
}

/*
  Input:  secret key (sk), 
          modulus (m),
          elliptic curve coeff (a), 
          primitive (P_x, P_y)
  Output: public key (T_x, T_y)
*/
ESS
void make_pk_fast(int sk, int P_x, int P_y, int *T_x, int *T_y, unsigned int m, int a)
{
  int i = 0;
  *T_x = P_x; 
  *T_y = P_y;

  for(i=first_set_bit( sk )-1; i>=0; --i) 
  {
    point_doubling(m, a, *T_x, *T_y, T_x, T_y);

    if( (1 << i) & sk ) // if the bit at index 'i' is 1 then point addition
      point_addition(m, *T_x, *T_y, P_x, P_y, T_x, T_y);
  }
}

/*
  Input:  secret key (sk), 
          modulus (m),
          elliptic curve coeff (a), 
          primitive (P_x, P_y)
  Output: public key (T_x, T_y)
*/
ESS
void make_pk_slow(int sk, int P_x, int P_y, int *T_x, int *T_y, unsigned int m, int a)
{
  point_doubling(m, a, P_x, P_y, T_x, T_y);

  sk -= 2;
  
  while(sk > 0)
  {
    point_addition(m, *T_x, *T_y, P_x, P_y, T_x, T_y);

    --sk;
  }
}

/*
  Input:  secret key (sk), 
          other entity public key (T_x, T_y),
          elliptic curve coeff (a),
          modulus (m)
  Output: shared secret key (shared_x, shared_y)
*/
ESS
void get_shared_key(int sk, int T_x, int T_y, int *shared_x, int *shared_y, unsigned int m, int a)
{
  make_pk_fast(sk, T_x, T_y, shared_x, shared_y, m, a);
}

__global__ void k_slow (
  const int sk,
  const int P_x,
  const int P_y, 
  int *__restrict__ T_x,
  int *__restrict__ T_y,
  const unsigned int m,
  const int a,
  const int num_pk)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pk)
    make_pk_slow(sk, P_x, P_y, T_x+i, T_y+i, m, a);
}

__global__ void k_fast (
  const int sk,
  const int P_x,
  const int P_y, 
  int *__restrict__ T_x,
  int *__restrict__ T_y,
  const unsigned int m,
  const int a,
  const int num_pk)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pk)
    make_pk_fast(sk, P_x, P_y, T_x+i, T_y+i, m, a);
}
