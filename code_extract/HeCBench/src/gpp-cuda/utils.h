#include <stdio.h>
#include <stdlib.h>

#define aqsmtemp_size      (number_bands * ncouls)
#define aqsntemp_size      (number_bands * ncouls)
#define I_eps_array_size   (ngpown * ncouls)
#define achtemp_size       (nend - nstart)
#define achtemp_re_size    (nend - nstart)
#define achtemp_im_size    (nend - nstart)
#define vcoul_size         ncouls
#define inv_igp_index_size ngpown
#define indinv_size        (ncouls + 1)
#define wx_array_size      (nend - nstart)
#define wtilde_array_size  (ngpown * ncouls)

#define aqsmtemp(n1, ig) aqsmtemp[n1 * ncouls + ig]
#define aqsntemp(n1, ig) aqsntemp[n1 * ncouls + ig]
#define I_eps_array(my_igp, ig) I_eps_array[my_igp * ncouls + ig]
#define wtilde_array(my_igp, ig) wtilde_array[my_igp * ncouls + ig]

inline void *safe_malloc(size_t n) {
  void *p = malloc(n);
  if (p == NULL) {
    fprintf(stderr, "Fatal: failed to allocate %zu bytes.\n", n);
    abort();
  }
  return p;
}

// Here we are checking to see if the answers are correct
inline void correctness(int problem_size, CustomComplex<dataType> result) {
  if (problem_size == 0) {
    dataType re_diff = result.get_real() - -24852.551547;
    dataType im_diff = result.get_imag() - 2957453.638101;

    if (re_diff < 0.00001 && im_diff < 0.00001)
      printf("\nBenchmark result: SUCCESS\n");
    else
      printf("\nBenchmark result: FAILURE\n");

  } else {
    dataType re_diff = result.get_real() - -0.096066;
    dataType im_diff = result.get_imag() - 11.431852;

    if (re_diff < 0.00001 && im_diff < 0.00001)
      printf("\nTest result: SUCCESS\n");
    else
      printf("\nTest result: FAILURE\n");
  }
}
