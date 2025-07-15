#ifndef UTILS_H
#define UTILS_H

void init_matrix(float *matrix, size_t num_rows, size_t num_cols, size_t nnz)
{
  size_t n = num_rows * num_cols;

  float *d = (float *) malloc(n * sizeof(float));

  srand(123);
  for (size_t i = 0; i < n; i++) d[i] = (float)i;
  for (size_t i = n; i > 0; i--) {
    size_t a = i-1;
    size_t b = rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (size_t i = 0; i < num_rows; i++) {
    for (size_t j = 0; j < num_cols; j++) {
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0 : (float)(drand48()+1);
    }
  }

  free(d);
}

#endif
