/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "KeccakTreeCPU.h"
#include "KeccakF.h"

void KeccakTreeCPU(tKeccakLane * inBuffer, tKeccakLane * outBuffer)
{
  int thrIdx,blkIdx;
  int k,ind_word;

  for(blkIdx=0;blkIdx<NB_THREADS_BLOCKS;blkIdx++) //loop on threads blocks
  {
    for (thrIdx=0; thrIdx< NB_THREADS;thrIdx++)//loop on threads inside a threadblock
    {
      tKeccakLane  Kstate[25];            
      memset(Kstate, 0, 25 * sizeof(tKeccakLane));

      for (k=0;k<NB_INPUT_BLOCK;k++)
      {
        //xor input into state
        for (ind_word=0; ind_word<INPUT_BLOCK_SIZE_B/4; ind_word++)
        {

          Kstate[ind_word] ^= 
            inBuffer[thrIdx 
            + ind_word  * NB_THREADS 
            + k      * NB_THREADS * INPUT_BLOCK_SIZE_B/4
            + blkIdx  * NB_THREADS * INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK ];
        }
        //apply Keccak permutation
        KeccakF_CPU(Kstate);
      }

      //output hash in out buffer
      for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      {
        //printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        outBuffer[thrIdx 
          + ind_word *NB_THREADS
          + blkIdx   *NB_THREADS * OUTPUT_BLOCK_SIZE_B/4 ]= Kstate[ind_word];
      }
    }//end loop threads
  }//end loop on threadsblocks
}

// Implement a second stage on treehashing
// Use output of 2x OUTPUT_BLOCK_SIZE_B size to respect conditions for soundness of Treehashing

void KeccakTreeCPU_2stg(tKeccakLane * inBuffer, tKeccakLane * outBuffer)
{
  int thrIdx,blkIdx;
  int k,ind_word;

  for(blkIdx=0;blkIdx<NB_THREADS_BLOCKS;blkIdx++) //loop on threads blocks
  {
    //Shared Buffer to store first stage hash output
    //shared memory will be used on GPU
    tKeccakLane * SharedBuffer=NULL;

    //alloc and init to 0
    SharedBuffer = (tKeccakLane *)malloc( 2 * OUTPUT_BLOCK_SIZE_B * NB_THREADS );
    memset(SharedBuffer,0, 2* OUTPUT_BLOCK_SIZE_B * NB_THREADS);

    //printf("SharedBuffer malloc blkIdx : %d \n\n",blkIdx);

    for (thrIdx=0; thrIdx< NB_THREADS;thrIdx++)//loop on threads inside a threadblock
    {
      tKeccakLane  Kstate[25];          
      memset(Kstate, 0, 25 * sizeof(tKeccakLane));

      for (k=0;k<NB_INPUT_BLOCK;k++)
      {
        //xor input into state
        for (ind_word=0; ind_word<INPUT_BLOCK_SIZE_B/4; ind_word++)
        {  
          Kstate[ind_word] ^= 
            inBuffer[thrIdx 
            + ind_word  * NB_THREADS 
            + k      * NB_THREADS * INPUT_BLOCK_SIZE_B/4
            + blkIdx  * NB_THREADS * INPUT_BLOCK_SIZE_B/4 * NB_INPUT_BLOCK ];

        }
        //apply Keccak permutation
        KeccakF_CPU(Kstate);
      }

      //output hash in Shared buffer
      for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      {        
        //printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        SharedBuffer[thrIdx 
          + ind_word *NB_THREADS ]= Kstate[ind_word];
      }

      //need to squeeze to produce more hash output 
      KeccakF_CPU(Kstate);

      for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      {
        //printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        SharedBuffer[thrIdx 
          + ind_word *NB_THREADS
          + NB_THREADS * OUTPUT_BLOCK_SIZE_B/4 ]= Kstate[ind_word];
      }
    }//end first loop on threads

    //***************************
    //second stage of treehash
    //***************************

    for (thrIdx=0; thrIdx< NB_SCND_STAGE_THREADS;thrIdx++)// 2nd loop on threads inside a threadblock
    {
      tKeccakLane  Kstate[25];          
      memset(Kstate, 0, 25 * sizeof(tKeccakLane));

      //number of input block per thread is now NB_INPUT_BLOCK_SNCD_STAGE = 2* NB_THREADS/ NB_SNCD_STAGE_THREADS
      //
      for (k=0;k<NB_INPUT_BLOCK_SNCD_STAGE;k++)
      {
        //xor input into state
        for (ind_word=0; ind_word<INPUT_BLOCK_SIZE_B/4; ind_word++)
        {  
          Kstate[ind_word] ^= 
            SharedBuffer[thrIdx 
            + ind_word  * NB_SCND_STAGE_THREADS 
            + k      * NB_SCND_STAGE_THREADS * INPUT_BLOCK_SIZE_B/4 ];
        }

        //apply Keccak permutation
        KeccakF_CPU(Kstate);
      }

      //output hash in output buffer
      for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      {        
        //printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        outBuffer[thrIdx 
          + ind_word * NB_SCND_STAGE_THREADS 
          + blkIdx   * NB_SCND_STAGE_THREADS * 2*OUTPUT_BLOCK_SIZE_B/4 ]= Kstate[ind_word];
      }

      //need to squeeze to produce more hash output 
      KeccakF_CPU(Kstate);

      for (ind_word=0; ind_word<OUTPUT_BLOCK_SIZE_B/4; ind_word++)
      {
        //printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        outBuffer[thrIdx 
          + ind_word *NB_SCND_STAGE_THREADS 
          + NB_SCND_STAGE_THREADS * OUTPUT_BLOCK_SIZE_B/4 
          + blkIdx   * NB_SCND_STAGE_THREADS * 2* OUTPUT_BLOCK_SIZE_B/4 ]= Kstate[ind_word];
      }
    }//end second loop on threads

    //free shared buffer
    free(SharedBuffer);

  }//end loop on threadsblocks
}
