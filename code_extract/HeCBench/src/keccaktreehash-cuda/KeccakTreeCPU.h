/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef KECCAKTREECPU_H_INCLUDED
#define KECCAKTREECPU_H_INCLUDED

#include "KeccakTree.h"
#include "KeccakTypes.h"

//Implement Tree hash mode 1 on CPU
//data to be hashed is present in inBuffer
//output result is in outBuffer
void KeccakTreeCPU(tKeccakLane * inBuffer, tKeccakLane * outBuffer);


#endif // KECCAKTREECPU_H_INCLUDED
