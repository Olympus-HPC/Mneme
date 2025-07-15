#include <cuComplex.h>

__global__ void ref_complex_float (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return; 
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  float r1 = LCG_random_double(&seed);
  float r2 = LCG_random_double(&seed); 
  float r3 = LCG_random_double(&seed); 
  float r4 = LCG_random_double(&seed); 

  cuFloatComplex z1 = make_cuFloatComplex(r1, r2);
  cuFloatComplex z2 = make_cuFloatComplex(r3, r4);

  char s = fabsf(cuCabsf(Cmulf(z1, z2)) - cuCabsf(z1) * cuCabsf(z2)) < 1e-3f;

  s += fabsf(cuCabsf(cuCaddf(z1, z2)) * cuCabsf(cuCaddf(z1 , z2)) -
             cuCrealf(cuCmulf(cuCaddf(z1, z2) , cuCaddf(cuConjf(z1), cuConjf(z2))))) < 1e-3f; 

  s += fabsf(cuCabsf(cuCsubf(z1, z2)) * cuCabsf(cuCsubf(z1 , z2)) -
             cuCrealf(cuCmulf(cuCsubf(z1, z2) , cuCsubf(cuConjf(z1), cuConjf(z2))))) < 1e-3f;

  s += fabsf(cuCrealf(cuCaddf(cuCmulf(z1, cuConjf(z2)) , cuCmulf(z2, cuConjf(z1)))) -
             2.0f * (cuCrealf(z1) * cuCrealf(z2) + cuCimagf(z1) * cuCimagf(z2))) < 1e-3f;

  s += fabsf(cuCabsf(cuCdivf(cuConjf(z1), z2)) -
             cuCabsf(cuCdivf(cuConjf(z1), cuConjf(z2)))) < 1e-3f;

  checkSum[i] = s;
}

__global__ void ref_complex_double (char* checkSum, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= n) return; 
  uint64_t seed = 1ULL;
  seed = fast_forward_LCG(seed, i);
  double r1 = LCG_random_double(&seed);
  double r2 = LCG_random_double(&seed); 
  double r3 = LCG_random_double(&seed); 
  double r4 = LCG_random_double(&seed); 

  cuDoubleComplex z1 = make_cuDoubleComplex(r1, r2);
  cuDoubleComplex z2 = make_cuDoubleComplex(r3, r4);

  char s = fabs(cuCabs(cuCmul(z1, z2)) - cuCabs(z1) * cuCabs(z2)) < 1e-3;

  s += fabs(cuCabs(cuCadd(z1, z2)) * cuCabs(cuCadd(z1 , z2)) -
            cuCreal(cuCmul(cuCadd(z1, z2) , cuCadd(cuConj(z1), cuConj(z2))))) < 1e-3; 

  s += fabs(cuCabs(cuCsub(z1, z2)) * cuCabs(cuCsub(z1 , z2)) -
            cuCreal(cuCmul(cuCsub(z1, z2) , cuCsub(cuConj(z1), cuConj(z2))))) < 1e-3;

  s += fabs(cuCreal(cuCadd(cuCmul(z1, cuConj(z2)) , cuCmul(z2, cuConj(z1)))) -
            2.0 * (cuCreal(z1) * cuCreal(z2) + cuCimag(z1) * cuCimag(z2))) < 1e-3;

  s += fabs(cuCabs(cuCdiv(cuConj(z1), z2)) -
            cuCabs(cuCdiv(cuConj(z1), cuConj(z2)))) < 1e-3;

  checkSum[i] = s;
}
