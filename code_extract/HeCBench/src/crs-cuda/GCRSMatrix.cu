//
//  GCRSMatrix.c
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017 csliu. All rights reserved.
//

#include "GCRSMatrix.h"
#include "galois.h"
#include "jerasure.h"
#include "utils.h"


int gcrs_check_k_m_w(int k, int m, int w){
  if (k < MIN_K || k > MAX_K) {
    return -1;
  }

  if (k < m) {
    return -1;
  }

  if (w < MIN_W) {
    w = MIN_W;
  }

  while (pow(2, w) < (k+m)) {
    ++w;
  }

  return w;
}

int *gcrs_create_bitmatrix(int k, int m, int w){
  int i, j;
  int *matrix, *bitmatrix;

  if (gcrs_check_k_m_w(k, m, w) < 0) {
    return NULL;
  }

  matrix = talloc(int, k*m);

  if (matrix == NULL) {
    return NULL;
  }

  for (i = 0; i < m; i++) {
    for (j = 0; j < k; j++) {
      matrix[i*k+j] = galois_single_divide(1, i ^ (m + j), w);
    }
  }

  bitmatrix = jerasure_matrix_to_bitmatrix(k, m, w, matrix);

  free(matrix);

  return bitmatrix;
}

unsigned int *gcrs_create_column_coding_bitmatrix(int k, int m, int w, int *bitmatrix){
  int columnIdx, rowIdx;
  int freeBitmatrixMark = 0, bitmatrixIdx;
  int intbitmatrixIdx = 0;
  int bitIdx = 0;
  unsigned int bitOne = 0x01;
  unsigned int *column_encoded_bitmatrix;

  if (gcrs_check_k_m_w(k, m, w) < 0) {
    return NULL;
  }

  if (bitmatrix == NULL) {
    bitmatrix = gcrs_create_bitmatrix(k, m, w);
    freeBitmatrixMark = 1;
    if (bitmatrix == NULL) {
      return  NULL;
    }
  }

  //    int uIntSize = sizeof(unsigned int);
  //    int wUnitsPerUInt = uIntSize/w;
  //    int mwReqSize = m/wUnitsPerUInt;
  //    
  //    if (m%wUnitsPerUInt != 1) {
  //        mwReqSize = mwReqSize + 1;
  //    }

  column_encoded_bitmatrix = talloc(unsigned int, k * w * 2);

  if (column_encoded_bitmatrix == NULL) {
    free(bitmatrix);
    return NULL;
  }

  memset(column_encoded_bitmatrix, 0, k*w*2*sizeof(unsigned int));

  for (columnIdx = 0; columnIdx < (k*w); ++columnIdx) {

    //map a whole column to 4 bytes (mw <= 32) or 8 bytes (mw >= 32)
    for (rowIdx = 0; rowIdx < (m*w); ++rowIdx) {
      bitmatrixIdx = rowIdx * k * w + columnIdx;

      if (rowIdx % w == 0) {

        //if cannot put a whole w bits to the last several bits of the integer, put w bits to the start w-bits next integer
        if ((bitIdx + w) > sizeof(int) * 8) {
          bitIdx = 0;
          intbitmatrixIdx = intbitmatrixIdx + 1;
        }
      }

      if (bitIdx >= sizeof(int) * 8) {
        bitIdx = 0;
        intbitmatrixIdx = intbitmatrixIdx + 1;
      }

      if (bitmatrix[bitmatrixIdx] == 1) {
        //copy a one to the bitIdx-bit of intbitmatrix[intbitmatrixIdx]
        column_encoded_bitmatrix[intbitmatrixIdx] = column_encoded_bitmatrix[intbitmatrixIdx] +  (bitOne << bitIdx);
      }else{
        //copy a zero to the bitIdx-bit of intbitmatrix[intbitmatrixIdx]
      }

      ++bitIdx;
    }

    bitIdx = 0;
    intbitmatrixIdx = intbitmatrixIdx + 1;
  }

  if (freeBitmatrixMark == 1) {
    free(bitmatrix);
  }

  return column_encoded_bitmatrix;
}

int gcrs_invert_bitmatrix(int *mat, int *inv, int rows)
{
  int cols, i, j, k;
  int tmp;

  cols = rows;

  k = 0;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      inv[k] = (i == j) ? 1 : 0;
      k++;
    }
  }

  /* First -- convert into upper triangular */

  for (i = 0; i < cols; i++) {

    /* Swap rows if we have a zero i,i element.  If we can't swap, then the
       matrix was not invertible */

    if ((mat[i*cols+i]) == 0) {
      for (j = i+1; j < rows && (mat[j*cols+i]) == 0; j++) ;
      if (j == rows) return -1;
      for (k = 0; k < cols; k++) {
        tmp = mat[i*cols+k]; mat[i*cols+k] = mat[j*cols+k]; mat[j*cols+k] = tmp;
        tmp = inv[i*cols+k]; inv[i*cols+k] = inv[j*cols+k]; inv[j*cols+k] = tmp;
      }
    }

    /* Now for each j>i, add A_ji*Ai to Aj */
    for (j = i+1; j != rows; j++) {
      if (mat[j*cols+i] != 0) {
        for (k = 0; k < cols; k++) {
          mat[j*cols+k] ^= mat[i*cols+k];
          inv[j*cols+k] ^= inv[i*cols+k];
        }
      }
    }
  }

  /* Now the matrix is upper triangular.  Start at the top and multiply down */

  for (i = rows-1; i >= 0; i--) {
    for (j = 0; j < i; j++) {
      if (mat[j*cols+i]) {
        for (k = 0; k < cols; k++) {
          mat[j*cols+k] ^= mat[i*cols+k];
          inv[j*cols+k] ^= inv[i*cols+k];
        }
      }
    }
  }

  return 0;
}

int gcrs_create_decoding_data_bitmatrix(int k, int m, int w,
    int *matrix, int *decoding_data_matrix,
    int *erased, int *dm_ids){
  int i, j, *tmpmat;
  int index, mindex;

  if (gcrs_check_k_m_w(k, m, w) < 0) {
    return -1;
  }

  if (matrix == NULL) {
    return -1;
  }

  j = 0;
  for (i = 0; j < k; i++) {
    if (erased[i] == 0) {
      dm_ids[j] = i;
      j++;
    }
  }

  tmpmat = (int*)malloc(sizeof(int)*k*k*w*w);
  if (tmpmat == NULL) { return -1; }
  for (i = 0; i < k; i++) {
    if (dm_ids[i] < k) {
      index = i*k*w*w;
      for (j = 0; j < k*w*w; j++) tmpmat[index+j] = 0;
      index = i*k*w*w+dm_ids[i]*w;
      for (j = 0; j < w; j++) {
        tmpmat[index] = 1;
        index += (k*w+1);
      }
    } else {
      index = i*k*w*w;
      mindex = (dm_ids[i]-k)*k*w*w;
      for (j = 0; j < k*w*w; j++) {
        tmpmat[index+j] = matrix[mindex+j];
      }
    }
  }

  i = gcrs_invert_bitmatrix(tmpmat, decoding_data_matrix, k*w);
  free(tmpmat);
  return 0;
}

int *gcrs_create_decoding_bitmatrix(int k, int m, int w,
    int *matrix, int *mat_idx,
    int *erasures, int *dm_ids){
  int kIdx, mIdx, matIdx;
  int *erases;
  int *decoding_matrix, *decoding_data_matrix;
  int dFailedNum = 0;

  decoding_matrix = talloc(int, (k * w * w * m));

  if (decoding_matrix == NULL) {
    return NULL;
  }

  if ((erases = gcrs_erasures_to_erased(k, m, erasures)) == NULL) {
    return NULL;
  }

  if ((decoding_data_matrix =  talloc(int, k*k*w*w)) == NULL ) {
    return NULL;
  }

  for (kIdx = 0; kIdx < k; ++kIdx) {
    if (erases[kIdx] == 1) {
      ++dFailedNum;
    }
  }

  if (dFailedNum > 0) {
    if(gcrs_create_decoding_data_bitmatrix(k, m, w,
          matrix, decoding_data_matrix,
          erases, dm_ids) < 0){
      free(erases);
      free(decoding_data_matrix);

      return NULL;
    }
  }


  matIdx = 0;
  for (kIdx = 0; kIdx < k; ++kIdx) {

    if (erases[kIdx] == 1) {
      *(mat_idx + matIdx) = kIdx;
      memcpy((decoding_matrix + matIdx * k * w * w), decoding_data_matrix + kIdx * k * w * w , sizeof(int) * k * w * w);
      matIdx = matIdx + 1;
    }
  }

  for (mIdx = 0; mIdx < m; ++mIdx) {
    if (erases[mIdx + kIdx] == 1) {
      *(mat_idx + matIdx) = mIdx + k;
      //Generate the vector for restoring
      memset((decoding_matrix + matIdx * k * w * w), 0, sizeof(int) * k * w * w);
      gcrs_generate_coding_vector(k,m,w,mIdx,
          matrix, decoding_data_matrix, (decoding_matrix + matIdx * k * w * w),
          erases);
      matIdx = matIdx + 1;
    }
  }

  free(erases);
  free(decoding_data_matrix);

  return decoding_matrix;
}

int *gcrs_erasures_to_erased(int k, int m, int *erasures)
{
  int td;
  int t_non_erased;
  int *erased;
  int i;

  td = k+m;
  erased = talloc(int, td);
  if (erased == NULL) return NULL;
  t_non_erased = td;

  for (i = 0; i < td; i++) erased[i] = 0;

  for (i = 0; erasures[i] != -1; i++) {
    if (erased[erasures[i]] == 0) {
      erased[erasures[i]] = 1;
      t_non_erased--;
      if (t_non_erased < k) {
        free(erased);
        return NULL;
      }
    }
  }
  return erased;
}

int gcrs_generate_coding_vector(int k, int m, int w, int mIdx,
    int *matrix, int *invert_matrix, int *vector,
    int *erases){
  int kIdx, wIdx, mapIdx, mapMissingIdx, rowIdx;
  int mappingMatrixRowIdx;

  //records the position mapped to vector for each k
  int *kMap = talloc(int, k);
  int *kMissingMap = talloc(int, (m+1)); // at most m data block missing;
  int *mappingMatrix = talloc(int, k*w*w);

  if (kMap == NULL || kMissingMap == NULL || mappingMatrix == NULL) {
    free(kMap);
    free(kMissingMap);
    free(mappingMatrix);

    return -1;
  }

  for (kIdx = 0; kIdx < k; ++kIdx) {
    *(kMap + kIdx) = -1;

    if (kIdx <= m) {
      *(kMissingMap + kIdx) = -1;
    }
  }

  mapIdx = 0;
  mapMissingIdx = 0;

  //Reorder vector elements
  for (kIdx = 0; kIdx < k; ++kIdx) {
    if (erases[kIdx] != 1) {
      *(kMap + mapIdx) = kIdx;
      for (rowIdx = 0; rowIdx < w; ++rowIdx) {
        memcpy((vector + mapIdx * w + rowIdx * k * w), (matrix + mIdx * k * w * w + rowIdx * k * w + kIdx * w), sizeof(int) * w);
      }
      ++mapIdx;
    }else{
      *(kMissingMap + mapMissingIdx) = kIdx;
      ++mapMissingIdx;

      // Can't continue if more than m data blocks missing
      if (mapMissingIdx > m) {
        return -1;
      }
    }
  }


  for (rowIdx = 0; rowIdx < w; ++rowIdx) {

    for (mapMissingIdx = 0; *(kMissingMap + mapMissingIdx) != -1; ++mapMissingIdx) {
      memset(mappingMatrix, 0, sizeof(int) * k * w * w);
      mappingMatrixRowIdx = 0;

      for (wIdx = 0; wIdx < w; ++wIdx) {

        if (*(matrix + mIdx * k * w * w + rowIdx * k * w + *(kMissingMap + mapMissingIdx) * w + wIdx) == 1) {
          //We must put the vector for generating the block here
          memcpy((mappingMatrix + mappingMatrixRowIdx * k * w), (invert_matrix + (*(kMissingMap + mapMissingIdx)) * k * w * w + wIdx * k * w), sizeof(int) * k * w);
          ++mappingMatrixRowIdx;
        }
      }

      for (mappingMatrixRowIdx = 0; mappingMatrixRowIdx < w; ++mappingMatrixRowIdx) {
        int columnIdx = 0;
        for (; columnIdx < k*w; ++columnIdx) {
          *(vector + rowIdx * k * w + columnIdx) = (*(vector + rowIdx * k * w + columnIdx) + *(mappingMatrix + mappingMatrixRowIdx * k * w + columnIdx)) % 2;
        }
      }
    }

  }

  //Free what you allocated
  free(kMap);
  free(kMissingMap);
  free(mappingMatrix);

  return 0;
}

void printMatrix(int *mat, int coloum, int row){
  int rIdx, cIdx;

  for (rIdx = 0; rIdx < row; ++rIdx) {
    for (cIdx = 0; cIdx < coloum; ++cIdx) {
      printf("%d ", *(mat + rIdx * coloum + cIdx));
    }
    printf("\n");
  }
}

void gcrs_print_column_encoded_bitmatrix(unsigned int *column_encoded_bitmatrix,
    int k, int m, int w){

  int intIdx, intMatrixIdx = 0;
  int intBitIdx, intSize = sizeof(unsigned int)*8;
  unsigned int intBit = 0x01;

  for (intIdx = 0; intIdx < k*w; ++intIdx) {
    //        printf("\n%u\n",int_encoded_bitmatrix[intMatrixIdx]);
    if (m*w > sizeof(int)*8) {
      for (intBitIdx = 0; intBitIdx < intSize; ++intBitIdx) {
        if ( (column_encoded_bitmatrix[intMatrixIdx] & (intBit << intBitIdx)) != 0) {
          printf("1");
        }else{
          printf("0");
        }
      }
      ++intMatrixIdx;
    }

    for (intBitIdx = 0; intBitIdx < intSize; ++intBitIdx) {
      if ( (column_encoded_bitmatrix[intMatrixIdx] & (intBit << intBitIdx)) != 0) {
        printf("1");
      }else{
        printf("0");
      }
    }

    ++intMatrixIdx;

    printf("\n");

  }
}

