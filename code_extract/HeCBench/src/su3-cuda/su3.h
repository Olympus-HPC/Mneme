#ifndef _SU3_H
#define _SU3_H
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
#else
  #define su3_matrix    dsu3_matrix
  #define su3_vector    dsu3_vector
  #define Real          double
  #define Complx        dcomplex
#endif  /* PRECISION */

/*  c += a * b */
#define CMULSUM(a,b,c) { (c).real += (a).real*(b).real - (a).imag*(b).imag; \
                         (c).imag += (a).real*(b).imag + (a).imag*(b).real; }
/*  c = a * b */
#define CMUL(a,b,c) { (c).real = (a).real*(b).real - (a).imag*(b).imag; \
                      (c).imag = (a).real*(b).imag + (a).imag*(b).real; }
/*  a += b    */
#define CSUM(a,b) { (a).real += (b).real; (a).imag += (b).imag; }

#endif  /* _SU3_H */

