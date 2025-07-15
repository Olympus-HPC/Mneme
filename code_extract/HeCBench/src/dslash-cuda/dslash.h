#ifndef _DSLASH_H
#define _DSLASH_H

#include <iostream>
#include <vector>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/resource.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

extern size_t sites_on_node;
extern size_t even_sites_on_node;
extern std::vector<int> squaresize;
extern unsigned int verbose;
extern size_t       warmups;


#ifndef ITERATIONS
#  define ITERATIONS 100
#endif

#ifndef PRECISION
#  define PRECISION 2  // 1->single, 2->double
#endif

#ifndef LDIM
#  define LDIM 32       // Lattice size = LDIM^4
#endif

/* Adapted from su3.h in MILC version 7 */

/* generic precision complex number definition */
/* specific for float complex */
typedef struct {   
  float real;
  float imag; 
} fcomplex;  

/* specific for double complex */
typedef struct {
   double real;
   double imag;
} dcomplex;

typedef struct { fcomplex e[3][3]; } fsu3_matrix;
typedef struct { fcomplex c[3]; } fsu3_vector;

typedef struct { dcomplex e[3][3]; } dsu3_matrix;
typedef struct { dcomplex c[3]; } dsu3_vector;

#if (PRECISION==1)
  #define su3_matrix    fsu3_matrix
  #define su3_vector    fsu3_vector
  #define Real          float
  #define Complx        fcomplex
  #define EPISON        2E-5
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
  #define Complx        dcomplex
  #define EPISON        2E-6
#endif  /* PRECISION */

/*  c = a + b */
#define CADD(a,b,c)    { (c).real = (a).real + (b).real;  \
                         (c).imag = (a).imag + (b).imag; }
/*  c = a - b */
#define CSUB(a,b,c)    { (c).real = (a).real - (b).real;  \
                         (c).imag = (a).imag - (b).imag; }
/*  c += a * b */
#define CMULSUM(a,b,c) { (c).real += (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag += (a).real*(b).imag + (a).imag*(b).real; }
/*  c = a * b */
#define CMUL(a,b,c)    { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag = (a).real*(b).imag + (a).imag*(b).real; }
/*  a += b    */
#define CSUM(a,b)      { (a).real += (b).real; (a).imag += (b).imag; }
/*   b = a*   */
#define CONJG(a,b)     { (b).real = (a).real; (b).imag = -(a).imag; }

//--------------------------------------------------------------------------------
__device__
inline void su3_adjoint( const su3_matrix *a, su3_matrix *b ){
  int i,j;
  for(i=0;i<3;i++)for(j=0;j<3;j++){
      CONJG( a->e[j][i], b->e[i][j] );
    }
}
//--------------------------------------------------------------------------------
__host__ __device__
inline void mult_su3_mat_vec( const su3_matrix *a, const su3_vector *b, 
			      su3_vector *c  ){
  int i,j;
  for(i=0;i<3;i++){
    Complx x = {0.0, 0.0};
    for(j=0;j<3;j++){
      CMULSUM( a->e[i][j] , b->c[j] , x );
    }
    c->c[i] = x;
  }
}
//--------------------------------------------------------------------------------
__host__ __device__
inline void mult_su3_mat_vec_sum( const su3_matrix *a, const su3_vector *b, 
				  su3_vector *c ){
  int i,j;
  for(i=0;i<3;i++){
    Complx x = {0.0, 0.0};
    for(j=0;j<3;j++){
      CMULSUM( a->e[i][j] , b->c[j] , x );
    }
    c->c[i].real += x.real;
    c->c[i].imag += x.imag;
  }
}
//--------------------------------------------------------------------------------
inline void mult_su3_mat_vec_sum_4dir( su3_matrix *a, su3_vector *b0,
       su3_vector *b1, su3_vector *b2, su3_vector *b3, 
       su3_vector *c  ){
    mult_su3_mat_vec(       a,b0,c );
    mult_su3_mat_vec_sum( ++a,b1,c );
    mult_su3_mat_vec_sum( ++a,b2,c );
    mult_su3_mat_vec_sum( ++a,b3,c );
}
//--------------------------------------------------------------------------------
__host__ __device__
inline void add_su3_vector( const su3_vector *a, const su3_vector *b, su3_vector *c ){
  int i;
  for(i=0;i<3;i++){
    CADD( a->c[i], b->c[i], c->c[i] );
  }
}
//--------------------------------------------------------------------------------
__host__ __device__
inline void sub_su3_vector( const su3_vector *a, const su3_vector *b, su3_vector *c ){
  int i;
  for(i=0;i<3;i++){
    CSUB( a->c[i], b->c[i], c->c[i] );
  }
}
//--------------------------------------------------------------------------------
#include <stdio.h>

inline void dumpmat( su3_matrix *m ){
  int i,j;
  for(i=0;i<3;i++){
	  for(j=0;j<3;j++)
      printf("(%.2e,%.2e)\t", m->e[i][j].real,m->e[i][j].imag);
	  printf("\n");
  }
  printf("\n");
}

static int nx = LDIM;
static int ny = LDIM;
static int nz = LDIM;
static int nt = LDIM;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

//--------------------------------------------------------------------------------
inline size_t node_index(const int x, const int y, const int z, const int t) 
{
  size_t i;
  int xr,yr,zr,tr;
  xr = (x+nx)%squaresize[XUP]; yr = (y+ny)%squaresize[YUP];
  zr = (z+nz)%squaresize[ZUP]; tr = (t+nt)%squaresize[TUP];
  i = xr + squaresize[XUP]*( yr + squaresize[YUP]*( zr + squaresize[ZUP]*tr));
  if( (x+y+z+t)%2==0 ){	/* even site */
    return( i/2 );
  }
  else {
    return( (i + sites_on_node)/2 );
  }
}

//--------------------------------------------------------------------------------
// Set the indices for gathers from neighbors
inline void set_neighbors( size_t *fwd, size_t *bck, 
			   size_t *fwd3, size_t *bck3 )
{
  for(int x = 0; x < nx; x++)
    for(int y = 0; y < ny; y++)
      for(int z = 0; z < nz; z++)
	for(int t = 0; t < nt; t++)
	  {
            int i = node_index(x,y,z,t);
	    fwd[4*i+0]  = node_index(x+1,y,z,t);
	    bck[4*i+0]  = node_index(x-1,y,z,t);
	    fwd[4*i+1]  = node_index(x,y+1,z,t);
	    bck[4*i+1]  = node_index(x,y-1,z,t);
	    fwd[4*i+2]  = node_index(x,y,z+1,t);
	    bck[4*i+2]  = node_index(x,y,z-1,t);
	    fwd[4*i+3]  = node_index(x,y,z,t+1);
	    bck[4*i+3]  = node_index(x,y,z,t-1);
	    fwd3[4*i+0] = node_index(x+3,y,z,t);
	    bck3[4*i+0] = node_index(x-3,y,z,t);
	    fwd3[4*i+1] = node_index(x,y+3,z,t);
	    bck3[4*i+1] = node_index(x,y-3,z,t);
	    fwd3[4*i+2] = node_index(x,y,z+3,t);
	    bck3[4*i+2] = node_index(x,y,z-3,t);
	    fwd3[4*i+3] = node_index(x,y,z,t+3);
	    bck3[4*i+3] = node_index(x,y,z,t-3);
	  }
}

// Used for validation
void dslash_fn_field(su3_vector *src, su3_vector *dst,
		     int parity, su3_matrix *fat, su3_matrix *lng,
		     su3_matrix *fatbck, su3_matrix *lngbck );

double dslash_fn(
  const std::vector<su3_vector> &src, 
        std::vector<su3_vector> &dst,
  const std::vector<su3_matrix> &fat,
  const std::vector<su3_matrix> &lng,
        std::vector<su3_matrix> &fatbck,
        std::vector<su3_matrix> &lngbck,
  size_t *fwd, size_t *bck, size_t *fwd3, size_t *bck3,    
  const size_t iterations,
  size_t wgsize );

typedef std::chrono::system_clock Clock;


#endif
