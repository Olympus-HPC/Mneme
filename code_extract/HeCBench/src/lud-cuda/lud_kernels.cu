__global__ void
lud_diagonal (float *m, const int matrix_dim, const int offset) {
  __shared__ float shadow [BLOCK_SIZE*BLOCK_SIZE];
  int i,j;
  int tx = threadIdx.x;

  int array_offset = offset*matrix_dim+offset;
  for(i=0; i < BLOCK_SIZE; i++){
    shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
    array_offset += matrix_dim;
  }

  __syncthreads();

  for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
      shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

    __syncthreads();
    if (tx>i){

      for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }

    __syncthreads();
  }

  array_offset = (offset+1)*matrix_dim+offset;
  for(i=1; i < BLOCK_SIZE; i++){
    m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
    array_offset += matrix_dim;
  }
}

__global__ void
lud_perimeter (float *m, const int matrix_dim, const int offset) {
  __shared__ float dia [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_row [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_col [BLOCK_SIZE*BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  int  bx = blockIdx.x;  
  int  tx = threadIdx.x;

  if (tx < BLOCK_SIZE) {
    idx = tx;
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else {
    idx = tx-BLOCK_SIZE;

    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }

  }
  __syncthreads();

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];
      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
    }
  }

  __syncthreads();

  if (tx < BLOCK_SIZE) { //peri-row
    idx=tx;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=tx - BLOCK_SIZE;
    array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }
  }
}

__global__ void
lud_internal (float *m, const int matrix_dim, const int offset) {
  __shared__ float peri_row [BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float peri_col [BLOCK_SIZE*BLOCK_SIZE];
  int  bx = blockIdx.x;  
  int  by = blockIdx.y;  

  int  tx = threadIdx.x;
  int  ty = threadIdx.y;

  float sum;

  int global_row_id = offset + (by+1)*BLOCK_SIZE;
  int global_col_id = offset + (bx+1)*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

  __syncthreads();

  int i;
  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i]; // * peri_row[i * BLOCK_SIZE + tx];

  m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;
}
