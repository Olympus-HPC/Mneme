#ifndef SPMV_H
#define SPMV_H

#define REAL float
//#define REAL double
void mv_csr_serial( const size_t num_rows, const size_t *row_indices, const size_t *col_indices, const REAL *data, const REAL *x, REAL *y);
long mv_csr_parallel(const int repeat, const int bs, const size_t num_rows, const size_t *row_indices, const size_t *col_indices, const REAL *data, const REAL *x, const size_t nnz, REAL *matrix, REAL *y);
long vector_mv_csr_parallel(const int repeat, const int bs, const size_t num_rows, const size_t *row_indices, const size_t *col_indices, const REAL *data, const REAL *x, const size_t nnz, REAL *matrix, REAL *y);
long mv_dense_parallel(const int repeat, const int bs, const size_t num_rows, const REAL *x, REAL* matrix, REAL *y);
float check(REAL *A, REAL *B, size_t n);
void init_vector(REAL *vector, size_t m);
void init_matrix(REAL *matrix, size_t num_rows, size_t nnz);
void init_csr(size_t *row_indices, REAL *values, size_t *col_indices, REAL *matrix, size_t num_rows, size_t nnz);

#endif
