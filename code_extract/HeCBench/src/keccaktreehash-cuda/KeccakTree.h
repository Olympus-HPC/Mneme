/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREE_H_INCLUDED
#define KECCAKTREE_H_INCLUDED

#define NB_THREADS 64  // 96   // 192 // Numbers of threads PER BLOCK MUST BE a multiple of NB_SNCD_STAGE_NODES 
									//MUST BE > 8 for streamcipher mode 

#define NB_THREADS_BLOCKS 64 //  64 //   32 

#define NB_STREAMS 2 //  4  // 2 MUST DIVIDE NB_THREADS_BLOCKS

#define INPUT_BLOCK_SIZE_B 32   // 256 bits in : 32 Bytes MUST BE multiple of 4 
#define OUTPUT_BLOCK_SIZE_B 32  // 256 bits out of each keccak hash MUST BE multiple of 4 
#define NB_INPUT_BLOCK 1024   // 128  // 64   number of input block of 256 bits

// 2 stage Treehash
#define NB_SCND_STAGE_THREADS 16 // MUST DIVIDE NB_THREADS  
#define NB_INPUT_BLOCK_SNCD_STAGE  2*NB_THREADS/NB_SCND_STAGE_THREADS //

//StreamCipher
#define SC_NB_OUTPUT_BLOCK 64 // number of output blocks in stream cipher mode

#endif // KECCAKTREE_H_INCLUDED
