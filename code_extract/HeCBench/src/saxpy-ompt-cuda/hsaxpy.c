/**
 * @file hsaxpy.c
 * @brief Function definition for performing the \c saxpy operation on host.
 *
 * This source file contains function definition for the \c saxpy operation,
 * which is defined as:
 *
 * y := a * x + y
 *
 * where:
 *
 * - a is a scalar.
 * - x and y are single-precision vectors each with n elements.
 *
 * @author Xin Wu (PC²)
 * @date 05.04.2020
 * @copyright CC BY-SA 2.0
 */

#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "wtcalc.h"
#include "hsaxpy.h"

void hsaxpy(const int n,
            const float a,
            const float *x,
                  float *y)
{
  struct timespec rt[2];

/*
 * - naive implementation
 */
  clock_gettime(CLOCK_REALTIME, rt + 0);
  #pragma omp parallel for simd schedule(simd:static) \
  default(none) shared(a, n, x, y)
  for (int i = 0; i < n; i++) {
    y[i] = a * x[i] + y[i];
  }
  clock_gettime(CLOCK_REALTIME, rt + 1);
  if (wtcalc >= 0.0) {
    wtcalc += (rt[1].tv_sec - rt[0].tv_sec) + 1.0e-9 * (rt[1].tv_nsec - rt[0].tv_nsec);
  }
}
