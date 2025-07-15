#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "mv.h"

int main(int argc, char *argv[])
{
  size_t nnz = 0;
  size_t num_rows = 0;
  int repeat = 0;

  if (argc != 4) {
    printf("Usage %s <number of non-zero elements> <number of rows in a square matrix> <repeat>\n", argv[0]);
    return 1;
  }

  nnz = atol(argv[1]);
  num_rows = atol(argv[2]);
  repeat = atoi(argv[3]);

  size_t num_elems = num_rows * num_rows;

  assert(nnz > 0);
  assert(num_rows > 0);
  assert(nnz <= num_elems);

  size_t vector_size_bytes = num_rows * sizeof(REAL);
  size_t matrix_size_bytes = num_elems * sizeof(REAL);
  size_t value_size_bytes  = nnz * sizeof(REAL);
  size_t colidx_size_bytes = nnz * sizeof(size_t);
  size_t rowidx_size_bytes = (num_rows + 1) * sizeof(size_t);

  size_t *row_indices = (size_t *) malloc (rowidx_size_bytes);
  size_t *col_indices = (size_t *) malloc (colidx_size_bytes);
  REAL *values = (REAL *) malloc (value_size_bytes);

  REAL *y[4];

  REAL *x = (REAL *) malloc (vector_size_bytes);
  for (int i = 0; i < 4; i++)
    y[i] = (REAL *) malloc (vector_size_bytes);
  REAL *matrix = (REAL *) malloc (matrix_size_bytes);

  srand48(1<<12);
  init_matrix(matrix, num_rows, nnz);
  init_vector(x, num_rows);
  init_csr(row_indices, values, col_indices, matrix, num_rows, nnz);

  // reference results in y
  mv_csr_serial(num_rows, row_indices, col_indices, values, x, y[0]);

  printf("Number of non-zero elements: %lu\n", nnz);
  printf("Number of rows in a square matrix: %lu\n", num_rows);
  printf("Sparsity: %lf%%\n", (num_elems - nnz) * 1.0 / num_elems * 100.0);

  long elapsed[3];

  // thread block size
  for (int bs = 32; bs <= 1024; bs = bs * 2) {
    printf("\nThread block size: %d\n", bs);

    elapsed[0] = mv_dense_parallel(repeat, bs, num_rows, x, matrix, y[1]);
    elapsed[1] = mv_csr_parallel(repeat, bs, num_rows, row_indices,
                                 col_indices, values, x, nnz, matrix, y[2]);
    elapsed[2] = vector_mv_csr_parallel(repeat, bs, num_rows, row_indices,
                                        col_indices, values, x, nnz, matrix, y[3]);

    printf("Average dense, sparse, and vector sparse kernel execution time (ms):");
    for (int i = 0; i < 3; i++)
      printf(" %lf", elapsed[i] * 1e-6 / repeat);

    printf("\nError rate:");
    for (int i = 1; i < 4; i++) printf(" %f", check(y[0], y[i], num_rows));
    printf("\n");
  }

  free(row_indices);
  free(col_indices);
  free(values);
  free(x);
  free(matrix);
  for (int i = 0; i < 4; i++) free(y[i]);
  return 0;
}
