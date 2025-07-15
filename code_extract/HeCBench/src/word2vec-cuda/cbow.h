/*
 * cbow.h
 *
 *  Created on: Aug 29, 2015
 *      Author: gpgpu
 */

#ifndef CBOW_H_
#define CBOW_H_

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1024
#define MAX_CODE_LENGTH 40
#define MAX_SENTENCE_NUM 6
#define ALIGNMENT_FACTOR 32
#define THREADS_PER_WORD 128
#define BLOCK_SIZE 128
typedef float real;

void TrainGPU(int sentence_num);
void GetResultFromGPU();
void initializeGPU();
void cleanUpGPU();

#endif /* CBOW_H_ */
