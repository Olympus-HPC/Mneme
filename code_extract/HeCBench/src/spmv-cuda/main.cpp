#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mv.h"

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n", argv[0]);
    return 1;
  }

  size_t nnz = atol(argv[1]);
  assert(nnz > 0);

  size_t num_rows = atol(argv[2]);
  assert(num_rows > 0);

  int repeat = atoi(argv[3]);

  size_t num_elems = num_rows * num_rows;
  assert(nnz <= num_elems);

  size_t vector_size_bytes = num_rows * sizeof(REAL);
  size_t matrix_size_bytes = num_elems * sizeof(REAL);
  size_t value_size_bytes  = nnz * sizeof(REAL);

  REAL *values = (REAL *) malloc (value_size_bytes);
  REAL *x = (REAL *) malloc (vector_size_bytes);
  REAL *y_ref = (REAL *) malloc (vector_size_bytes);
  REAL *y = (REAL *) malloc (vector_size_bytes);
  REAL *matrix = (REAL *) malloc (matrix_size_bytes);

  srand48(1<<12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);

  long elapsed = reference(repeat, num_rows, x, matrix, y_ref);  

  printf("Number of non-zero elements: %lu\n", nnz);
  printf("Number of rows in a square matrix: %lu\n", num_rows);
  printf("Sparsity: %lf%%\n", (num_elems - nnz) * 1.0 / num_elems * 100.0);

  elapsed = spmv_csr(repeat, num_rows, x, nnz, matrix, y);
  printf("Average kernel (CSR) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
  printf("Error rate: %f\n", check(y, y_ref, num_rows));

  elapsed = spmv_coo(repeat, num_rows, x, nnz, matrix, y);
  printf("Average kernel (COO) execution time (ms): %lf\n", elapsed * 1e-6 / repeat);
  printf("Error rate: %f\n", check(y, y_ref, num_rows));

  free(values);
  free(x);
  free(y);
  free(y_ref);
  free(matrix);
  return 0;
}
