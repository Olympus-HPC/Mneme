#ifndef UTILS_H
#define UTILS_H

void init_matrix(float *matrix, int64_t num_rows, int64_t num_cols, int64_t nnz)
{
  int64_t n = (int64_t)num_rows * num_cols;

  float *d = (float *) malloc(n * sizeof(float));

  srand(123);
  for (int64_t i = 0; i < n; i++) d[i] = (float)i;
  for (int64_t i = n; i > 0; i--) {
    int64_t a = i-1;
    int64_t b = rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (int64_t i = 0; i < num_rows; i++) {
    for (int64_t j = 0; j < num_cols; j++) {
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0 : (float)(drand48()+1);
    }
  }

  free(d);
}

void init_csr(int64_t *row_indices, float *values,
              int64_t *col_indices, float *matrix,
              int64_t num_rows, int64_t num_cols, int64_t nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  int64_t *non_zero_elements = (int64_t*) malloc (num_rows * sizeof(int64_t));

  int64_t tmp = 0;
  for (int64_t i = 0; i < num_rows; i++) {
    int64_t nnz_per_row = 0; // nnz per row
    for (int64_t j = 0; j < num_cols; j++) {
      if(matrix[i*num_cols+j] != 0) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }

  for (int64_t i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i-1] + non_zero_elements[i-1];
  }

  free(non_zero_elements);
}

#endif
