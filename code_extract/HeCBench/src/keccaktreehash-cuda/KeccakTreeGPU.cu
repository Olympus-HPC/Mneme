/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "KeccakTree.h"
#include "KeccakF.h"

__host__ void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) 
  {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
        cudaGetErrorString( err) );
    exit(EXIT_FAILURE);
  }                         
}

/*GPU constants
  __constant__ tKeccakLane KeccakF_RoundConstants[22] =
  {
  (tKeccakLane)0x00000001 ,
  (tKeccakLane)0x00008082 ,
  (tKeccakLane)0x0000808a ,
  (tKeccakLane)0x80008000 ,
  (tKeccakLane)0x0000808b ,
  (tKeccakLane)0x80000001 ,
  (tKeccakLane)0x80008081 ,
  (tKeccakLane)0x00008009 ,
  (tKeccakLane)0x0000008a ,
  (tKeccakLane)0x00000088 ,
  (tKeccakLane)0x80008009 ,
  (tKeccakLane)0x8000000a ,
  (tKeccakLane)0x8000808b ,
  (tKeccakLane)0x0000008b ,
  (tKeccakLane)0x00008089 ,
  (tKeccakLane)0x00008003 ,
  (tKeccakLane)0x00008002 ,
  (tKeccakLane)0x00000080 ,
  (tKeccakLane)0x0000800a ,
  (tKeccakLane)0x8000000a ,
  (tKeccakLane)0x80008081 ,
  (tKeccakLane)0x00008080
  };
 */

//host constants
tKeccakLane KeccakF_RoundConstants_h[22] =
{
  (tKeccakLane)0x00000001 ,
  (tKeccakLane)0x00008082 ,
  (tKeccakLane)0x0000808a ,
  (tKeccakLane)0x80008000 ,
  (tKeccakLane)0x0000808b ,
  (tKeccakLane)0x80000001 ,
  (tKeccakLane)0x80008081 ,
  (tKeccakLane)0x00008009 ,
  (tKeccakLane)0x0000008a ,
  (tKeccakLane)0x00000088 ,
  (tKeccakLane)0x80008009 ,
  (tKeccakLane)0x8000000a ,
  (tKeccakLane)0x8000808b ,
  (tKeccakLane)0x0000008b ,
  (tKeccakLane)0x00008089 ,
  (tKeccakLane)0x00008003 ,
  (tKeccakLane)0x00008002 ,
  (tKeccakLane)0x00000080 ,
  (tKeccakLane)0x0000800a ,
  (tKeccakLane)0x8000000a ,
  (tKeccakLane)0x80008081 ,
  (tKeccakLane)0x00008080
};

// Device (GPU) Keccak-f function implementation
// unrolled
__device__ void KeccakFunr( tKeccakLane * state, const tKeccakLane *KeccakF_RoundConstants )
{
  unsigned int round; //try to avoid to many registers
  tKeccakLane BC[5];
  tKeccakLane temp;

  for ( round = 0; round < cKeccakNumberOfRounds; ++round )
  {
    {
      // Theta
      BC[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
      BC[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
      BC[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
      BC[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
      BC[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

      temp = BC[4] ^ ROL32(BC[1], 1);//x=0
      state[0] ^= temp;
      state[5] ^= temp;
      state[10] ^= temp;
      state[15] ^= temp;
      state[20] ^= temp;
      temp = BC[0] ^ ROL32(BC[2], 1);//x=1
      state[1] ^= temp;
      state[6] ^= temp;
      state[11] ^= temp;
      state[16] ^= temp;
      state[21] ^= temp;
      temp = BC[1] ^ ROL32(BC[3], 1);//x=2
      state[2] ^= temp;
      state[7] ^= temp;
      state[12] ^= temp;
      state[17] ^= temp;
      state[22] ^= temp;
      temp = BC[2] ^ ROL32(BC[4], 1);//x=3
      state[3] ^= temp;
      state[8] ^= temp;
      state[13] ^= temp;
      state[18] ^= temp;
      state[23] ^= temp;
      temp = BC[3] ^ ROL32(BC[0], 1);//x=4
      state[4] ^= temp;
      state[9] ^= temp;
      state[14] ^= temp;
      state[19] ^= temp;
      state[24] ^= temp;
    }//end Theta

    {
      // Rho Pi
      temp = state[1];
      BC[0] = state[10];
      state[10] = ROL32( temp, 1);
      temp = BC[0];//x=0
      BC[0] =  state[7];
      state[7] = ROL32( temp, 3);
      temp = BC[0];
      BC[0] = state[11];
      state[11] = ROL32( temp, 6);
      temp = BC[0];
      BC[0] = state[17];
      state[17] = ROL32( temp,10);
      temp = BC[0];
      BC[0] = state[18];
      state[18] = ROL32( temp,15);
      temp = BC[0];
      BC[0] =  state[3];
      state[3] = ROL32( temp,21);
      temp = BC[0];//x=5
      BC[0] =  state[5];
      state[5] = ROL32( temp,28);
      temp = BC[0];
      BC[0] = state[16];
      state[16] = ROL32( temp, 4);
      temp = BC[0];
      BC[0] =  state[8];
      state[8] = ROL32( temp,13);
      temp = BC[0];
      BC[0] = state[21];
      state[21] = ROL32( temp,23);
      temp = BC[0];
      BC[0] = state[24];
      state[24] = ROL32( temp, 2);
      temp = BC[0];//x=10
      BC[0] =  state[4];
      state[4] = ROL32( temp,14);
      temp = BC[0];
      BC[0] = state[15];
      state[15] = ROL32( temp,27);
      temp = BC[0];
      BC[0] = state[23];
      state[23] = ROL32( temp, 9);
      temp = BC[0];
      BC[0] = state[19];
      state[19] = ROL32( temp,24);
      temp = BC[0];
      BC[0] = state[13];
      state[13] = ROL32( temp, 8);
      temp = BC[0];//x=15
      BC[0] = state[12];
      state[12] = ROL32( temp,25);
      temp = BC[0];
      BC[0] =  state[2];
      state[2] = ROL32( temp,11);
      temp = BC[0];
      BC[0] = state[20];
      state[20] = ROL32( temp,30);
      temp = BC[0];
      BC[0] = state[14];
      state[14] = ROL32( temp,18);
      temp = BC[0];
      BC[0] = state[22];
      state[22] = ROL32( temp, 7);
      temp = BC[0];//x=20
      BC[0] =  state[9];
      state[9] = ROL32( temp,29);
      temp = BC[0];
      BC[0] =  state[6];
      state[6] = ROL32( temp,20);
      temp = BC[0];
      BC[0] =  state[1];
      state[1] = ROL32( temp,12);
      temp = BC[0];//x=23
    }//end Rho Pi

    {
      //  Chi
      BC[0] = state[0];
      BC[1] = state[1];
      BC[2] = state[2];
      BC[3] = state[3];
      BC[4] = state[4];
      state[0] = BC[0] ^((~BC[1]) & BC[2]);
      state[1] = BC[1] ^((~BC[2]) & BC[3]);
      state[2] = BC[2] ^((~BC[3]) & BC[4]);
      state[3] = BC[3] ^((~BC[4]) & BC[0]);
      state[4] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[5];
      BC[1] = state[6];
      BC[2] = state[7];
      BC[3] = state[8];
      BC[4] = state[9];
      state[5] = BC[0] ^((~BC[1]) & BC[2]);
      state[6] = BC[1] ^((~BC[2]) & BC[3]);
      state[7] = BC[2] ^((~BC[3]) & BC[4]);
      state[8] = BC[3] ^((~BC[4]) & BC[0]);
      state[9] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[10];
      BC[1] = state[11];
      BC[2] = state[12];
      BC[3] = state[13];
      BC[4] = state[14];
      state[10] = BC[0] ^((~BC[1]) & BC[2]);
      state[11] = BC[1] ^((~BC[2]) & BC[3]);
      state[12] = BC[2] ^((~BC[3]) & BC[4]);
      state[13] = BC[3] ^((~BC[4]) & BC[0]);
      state[14] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[15];
      BC[1] = state[16];
      BC[2] = state[17];
      BC[3] = state[18];
      BC[4] = state[19];
      state[15] = BC[0] ^((~BC[1]) & BC[2]);
      state[16] = BC[1] ^((~BC[2]) & BC[3]);
      state[17] = BC[2] ^((~BC[3]) & BC[4]);
      state[18] = BC[3] ^((~BC[4]) & BC[0]);
      state[19] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[20];
      BC[1] = state[21];
      BC[2] = state[22];
      BC[3] = state[23];
      BC[4] = state[24];
      state[20] = BC[0] ^((~BC[1]) & BC[2]);
      state[21] = BC[1] ^((~BC[2]) & BC[3]);
      state[22] = BC[2] ^((~BC[3]) & BC[4]);
      state[23] = BC[3] ^((~BC[4]) & BC[0]);
      state[24] = BC[4] ^((~BC[0]) & BC[1]);
    }//end Chi

    //  Iota
    state[0] ^= KeccakF_RoundConstants[round];
  }

}
//end unrolled

//Host Keccak-f function (pb with using the same constants between host and device) 
//unrolled
__host__ void KeccakFunr_h( tKeccakLane * state )
{
  unsigned int round; //try to avoid to many registers
  tKeccakLane BC[5];
  tKeccakLane temp;

  for ( round = 0; round < cKeccakNumberOfRounds; ++round )
  {
    {
      // Theta
      BC[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
      BC[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
      BC[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
      BC[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
      BC[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

      temp = BC[4] ^ ROL32(BC[1], 1);//x=0
      state[0] ^= temp;
      state[5] ^= temp;
      state[10] ^= temp;
      state[15] ^= temp;
      state[20] ^= temp;
      temp = BC[0] ^ ROL32(BC[2], 1);//x=1
      state[1] ^= temp;
      state[6] ^= temp;
      state[11] ^= temp;
      state[16] ^= temp;
      state[21] ^= temp;
      temp = BC[1] ^ ROL32(BC[3], 1);//x=2
      state[2] ^= temp;
      state[7] ^= temp;
      state[12] ^= temp;
      state[17] ^= temp;
      state[22] ^= temp;
      temp = BC[2] ^ ROL32(BC[4], 1);//x=3
      state[3] ^= temp;
      state[8] ^= temp;
      state[13] ^= temp;
      state[18] ^= temp;
      state[23] ^= temp;
      temp = BC[3] ^ ROL32(BC[0], 1);//x=4
      state[4] ^= temp;
      state[9] ^= temp;
      state[14] ^= temp;
      state[19] ^= temp;
      state[24] ^= temp;
    }//end Theta

    {
      // Rho Pi
      temp = state[1];
      BC[0] = state[10];
      state[10] = ROL32( temp, 1);
      temp = BC[0];//x=0
      BC[0] =  state[7];
      state[7] = ROL32( temp, 3);
      temp = BC[0];
      BC[0] = state[11];
      state[11] = ROL32( temp, 6);
      temp = BC[0];
      BC[0] = state[17];
      state[17] = ROL32( temp,10);
      temp = BC[0];
      BC[0] = state[18];
      state[18] = ROL32( temp,15);
      temp = BC[0];
      BC[0] =  state[3];
      state[3] = ROL32( temp,21);
      temp = BC[0];//x=5
      BC[0] =  state[5];
      state[5] = ROL32( temp,28);
      temp = BC[0];
      BC[0] = state[16];
      state[16] = ROL32( temp, 4);
      temp = BC[0];
      BC[0] =  state[8];
      state[8] = ROL32( temp,13);
      temp = BC[0];
      BC[0] = state[21];
      state[21] = ROL32( temp,23);
      temp = BC[0];
      BC[0] = state[24];
      state[24] = ROL32( temp, 2);
      temp = BC[0];//x=10
      BC[0] =  state[4];
      state[4] = ROL32( temp,14);
      temp = BC[0];
      BC[0] = state[15];
      state[15] = ROL32( temp,27);
      temp = BC[0];
      BC[0] = state[23];
      state[23] = ROL32( temp, 9);
      temp = BC[0];
      BC[0] = state[19];
      state[19] = ROL32( temp,24);
      temp = BC[0];
      BC[0] = state[13];
      state[13] = ROL32( temp, 8);
      temp = BC[0];//x=15
      BC[0] = state[12];
      state[12] = ROL32( temp,25);
      temp = BC[0];
      BC[0] =  state[2];
      state[2] = ROL32( temp,11);
      temp = BC[0];
      BC[0] = state[20];
      state[20] = ROL32( temp,30);
      temp = BC[0];
      BC[0] = state[14];
      state[14] = ROL32( temp,18);
      temp = BC[0];
      BC[0] = state[22];
      state[22] = ROL32( temp, 7);
      temp = BC[0];//x=20
      BC[0] =  state[9];
      state[9] = ROL32( temp,29);
      temp = BC[0];
      BC[0] =  state[6];
      state[6] = ROL32( temp,20);
      temp = BC[0];
      BC[0] =  state[1];
      state[1] = ROL32( temp,12);
      temp = BC[0];//x=23
    }//end Rho Pi

    {
      //  Chi
      BC[0] = state[0];
      BC[1] = state[1];
      BC[2] = state[2];
      BC[3] = state[3];
      BC[4] = state[4];
      state[0] = BC[0] ^((~BC[1]) & BC[2]);
      state[1] = BC[1] ^((~BC[2]) & BC[3]);
      state[2] = BC[2] ^((~BC[3]) & BC[4]);
      state[3] = BC[3] ^((~BC[4]) & BC[0]);
      state[4] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[5];
      BC[1] = state[6];
      BC[2] = state[7];
      BC[3] = state[8];
      BC[4] = state[9];
      state[5] = BC[0] ^((~BC[1]) & BC[2]);
      state[6] = BC[1] ^((~BC[2]) & BC[3]);
      state[7] = BC[2] ^((~BC[3]) & BC[4]);
      state[8] = BC[3] ^((~BC[4]) & BC[0]);
      state[9] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[10];
      BC[1] = state[11];
      BC[2] = state[12];
      BC[3] = state[13];
      BC[4] = state[14];
      state[10] = BC[0] ^((~BC[1]) & BC[2]);
      state[11] = BC[1] ^((~BC[2]) & BC[3]);
      state[12] = BC[2] ^((~BC[3]) & BC[4]);
      state[13] = BC[3] ^((~BC[4]) & BC[0]);
      state[14] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[15];
      BC[1] = state[16];
      BC[2] = state[17];
      BC[3] = state[18];
      BC[4] = state[19];
      state[15] = BC[0] ^((~BC[1]) & BC[2]);
      state[16] = BC[1] ^((~BC[2]) & BC[3]);
      state[17] = BC[2] ^((~BC[3]) & BC[4]);
      state[18] = BC[3] ^((~BC[4]) & BC[0]);
      state[19] = BC[4] ^((~BC[0]) & BC[1]);
      BC[0] = state[20];
      BC[1] = state[21];
      BC[2] = state[22];
      BC[3] = state[23];
      BC[4] = state[24];
      state[20] = BC[0] ^((~BC[1]) & BC[2]);
      state[21] = BC[1] ^((~BC[2]) & BC[3]);
      state[22] = BC[2] ^((~BC[3]) & BC[4]);
      state[23] = BC[3] ^((~BC[4]) & BC[0]);
      state[24] = BC[4] ^((~BC[0]) & BC[1]);
    }//end Chi

    //  Iota
    state[0] ^= KeccakF_RoundConstants_h[round];
  }

}
//end unrolled

//Keccak final node hashing results of previous nodes in sequential mode
__host__ void Keccak_top_GPU(tKeccakLane * Kstate, tKeccakLane *inBuffer , int block_number)
{
  int ind_word,k;

  for (k=0;k<block_number;k++)
  {

    for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
    {
      Kstate[ind_word] ^= inBuffer[ind_word + k * OUTPUT_BLOCK_SIZE_B/4];
    }
    KeccakFunr_h(Kstate);
  }
}

//************************************************************************
//kernel implementaing hash function, hashing NB_INPUT_BLOCK (of 256 bits)
//
__global__ void ker_Keccak(const tKeccakLane *__restrict__ d_inBuffer,
                                 tKeccakLane *__restrict__ d_outBuffer,
                           const tKeccakLane *__restrict__ KeccakF_RoundConstants)
{

  int ind_word,k;
  tKeccakLane Kstate[25];

  //zeroize the state
  for(ind_word=0; ind_word<25; ind_word++) {Kstate[ind_word]=0; } 

  for (k=0;k<NB_INPUT_BLOCK;k++)
  {
    //xor input into state
    for (ind_word=0; ind_word<(INPUT_BLOCK_SIZE_B/4 ); ind_word++)
    {

      Kstate[ind_word] ^= 
        d_inBuffer[threadIdx.x 
        + ind_word    * NB_THREADS 
        + k        * NB_THREADS * INPUT_BLOCK_SIZE_B/4
        + blockIdx.x  * NB_THREADS * INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK ];
    }
    //apply GPU Keccak permutation
    KeccakFunr(Kstate, KeccakF_RoundConstants);
  }

  //output hash in buffer
  for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
  {
    d_outBuffer[threadIdx.x 
      + ind_word *NB_THREADS
      + blockIdx.x   *NB_THREADS * OUTPUT_BLOCK_SIZE_B/4 ]= Kstate[ind_word];
  }
}

//********************************************************************************

//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************
__host__  void KeccakTreeGPU(tKeccakLane * h_inBuffer, tKeccakLane * d_inBuffer,
    tKeccakLane * h_outBuffer, tKeccakLane * d_outBuffer, tKeccakLane * d_KeccakF_RoundConstants )
{

  //copy host to device 
  cudaMemcpy(d_inBuffer,h_inBuffer, INPUT_BLOCK_SIZE_B * NB_THREADS * NB_INPUT_BLOCK*NB_THREADS_BLOCKS, cudaMemcpyHostToDevice);    
  checkCUDAError(" Memcpy htd");

  //exec kernels       
  ker_Keccak <<< dim3(NB_THREADS_BLOCKS), dim3(NB_THREADS) >>> (d_inBuffer, d_outBuffer, d_KeccakF_RoundConstants);
  checkCUDAError(" ker_keccak");

  //copy back device to host  
  cudaMemcpy(h_outBuffer,d_outBuffer, OUTPUT_BLOCK_SIZE_B * NB_THREADS*NB_THREADS_BLOCKS ,cudaMemcpyDeviceToHost);
  checkCUDAError(" Memcpy dth");
}
