#ifndef UTILS_H
#define UTILS_H

void init_matrix(float *matrix, int num_rows, int num_cols, int nnz)
{
  int n = (int)num_rows * num_cols;

  float *d = (float *) malloc(n * sizeof(float));

  srand(123);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  for (int i = n; i > 0; i--) {
    int a = i-1;
    int b = rand() % i;
    if (a != b) {
      auto t = d[a];
      d[a] = d[b];
      d[b] = t;
    }
  }

  srand48(123);
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0 : (float)(drand48()+1);
    }
  }

  free(d);
}

void init_csr(int *row_indices, float *values,
              int *col_indices, float *matrix,
              int num_rows, int num_cols, int nnz)
{
  row_indices[num_rows] = nnz;
  row_indices[0] = 0;
  int *non_zero_elements = (int*) malloc (num_rows * sizeof(int));

  int tmp = 0;
  for (int i = 0; i < num_rows; i++) {
    int nnz_per_row = 0; // nnz per row
    for (int j = 0; j < num_cols; j++) {
      if(matrix[i*num_cols+j] != 0) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++;
        nnz_per_row++;
      }
    }
    non_zero_elements[i] = nnz_per_row;
  }

  for (int i = 1; i < num_rows; i++) {
    row_indices[i] = row_indices[i-1] + non_zero_elements[i-1];
  }

  free(non_zero_elements);
}


void spsm (float *A, float *C, float *B,
           int A_num_rows, int C_num_cols)
{
/*
  printf("A matrix\n");
  for(int i = 0; i < A_num_rows; i++) {
    for (int j = 0; j < A_num_rows; j++) {
      printf("%f ", A[i*A_num_rows+j]);
    }
    printf("\n");
  }
  printf("\n");
*/

  for(int i = 0; i < A_num_rows; i++) {
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      // lower triangle
      for (int k = 0; k < A_num_rows; k++) { // B_num_rows
        if (k < i+1)
          s += A[i*A_num_rows+k] * C[k*C_num_cols+j];
      }
      B[i*C_num_cols+j] = s;
    }
  }
}

#endif

