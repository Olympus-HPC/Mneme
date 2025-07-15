#ifndef SPMV_H
#define SPMV_H

#define REAL float
//#define REAL double
long spmv_csr(const int repeat, const size_t num_rows, const REAL *x, const size_t nnz, REAL *matrix, REAL *y);
long spmv_coo(const int repeat, const size_t num_rows, const REAL *x, const size_t nnz, REAL *matrix, REAL *y);
long reference(const int repeat, const size_t num_rows, const REAL *x, REAL* matrix, REAL *y);
float check(REAL *A, REAL *B, size_t n);
void init_vector(REAL *vector, size_t m);
void init_matrix(REAL *matrix, size_t num_rows, size_t nnz);
void init_csr(size_t *row_indices, REAL *values, size_t *col_indices, REAL *matrix, size_t num_rows, size_t nnz);
void init_coo(size_t *row_indices, REAL *values, size_t *col_indices, REAL *matrix, size_t num_rows, size_t nnz);

#endif
