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


void sddmm (float *A, float *B, float *C, float *result,
            int *row_indices, int *col_indices,
            int A_num_cols, int C_num_rows, int C_num_cols)
{
  for(int i = 0; i < C_num_rows; i++) {
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      for (int k = 0; k < A_num_cols; k++) { // B_num_rows
        s += A[i*A_num_cols+k] * B[k*C_num_cols+j];
      }
      C[i*C_num_cols+j] = s;
    }
  }

  /* print the result of A * B
  for(int i = 0; i < C_num_rows; i++) {
    for (int j = 0; j < C_num_cols; j++) {
      printf("%f ", C[i*C_num_cols+j]);
    }
    printf("\n");
  }
  printf("\n\n");
  */

  int k = 0;
  for(int row = 0; row < C_num_rows; row++){
    int row_start = row_indices[row];
    int row_end = row_indices[row+1];
    for ( int i = row_start; i < row_end; i++) {
      result[k++] = C[row*C_num_cols + col_indices[i]];
    }
  }
}

#endif
