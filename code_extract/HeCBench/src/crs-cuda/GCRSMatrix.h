//
//  GCRSMatrix.h
//  NoSynFree
//
//  Created by Liu Chengjian on 17/8/29.
//  Copyright (c) 2017 csliu. All rights reserved.
//

#ifndef __NoSynFree__GCRSMatrix__
#define __NoSynFree__GCRSMatrix__

#include <stdio.h>
#include <string.h>
#include <math.h>

int gcrs_check_k_m_w(int k, int m, int w);

int *gcrs_create_bitmatrix(int k, int m, int w);

unsigned int *gcrs_create_column_coding_bitmatrix(int k, int m, int w,
                                                       int *bitmatrix);

int gcrs_create_decoding_data_bitmatrix(int k, int m, int w,
                                            int *matrix, int *decoding_data_matrix,
                                            int *erased, int *dm_ids);
int* gcrs_create_decoding_bitmatrix(int k, int m, int w,
                                        int *matrix, int *mat_idx,
                                        int *erasures, int *dm_ids);

int gcrs_generate_coding_vector(int k, int m, int w, int mIdx,
                                    int *matrix, int *invert_matrix, int *vector,
                                    int *erases);

int *gcrs_erasures_to_erased(int k, int m,
                                 int *erasures);

void printMatrix(int *mat, int coloum, int row);
void gcrs_print_column_encoded_bitmatrix(unsigned int *column_encoded_bitmatrix, int k, int m, int w);


#endif /* defined(__NoSynFree__GCRSMatrix__) */
