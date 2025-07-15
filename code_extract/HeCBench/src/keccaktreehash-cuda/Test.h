/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/

#ifndef TEST_H_INCLUDED
#define TEST_H_INCLUDED

//********************
// Basic treehash mode
//********************

//test Tree hash mode 1 in CPU only
//integer in argument is to lower the workload for CPU only test (slower than GPU tests)
void TestCPU(int);

//Test Tree hash mode 1 with GPU and CPU
void TestGPU(void);

//Test Tree hash mode 1 , GPU and CPU, CPU computation overlapped with GPU computation
void TestGPU_OverlapCPU(void);

void TestGPU_Split(void);

//Test Tree hash mode 1 , GPU and CPU, GPU computation is overlapped with memory transfers (Host to device) 
void TestGPU_Stream(void);

//Test Tree hash mode 1 , GPU and CPU, GPU computation is overlapped with memory transfers , and with CPU computation
void TestGPU_Stream_OverlapCPU(void);

//use of mapped memory : untested, unsupported by authors hardware
void TestGPU_MappedMemory(void);

//*************
//2 stages hash
//*************
void TestCPU_2stg(int);

void TestGPU_2stg(void);

void TestGPU_2stg_Stream_OverlapCPU(void);

//***************************
//Keccak in StreamCipher mode
//***************************
void TestGPU_SCipher(void);

// Other function

//Empirically Test if all words in input data are taken into the hash function
void Test_Completness(void);

//print GPU device info
void Device_Info(void);

//print Tree hash mode params set in KeccakTree.h
void Print_Param(void);

#endif // TEST_H_INCLUDED
