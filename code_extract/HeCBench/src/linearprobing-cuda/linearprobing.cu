#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include "linearprobing.h"

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k & (kHashTableCapacity-1);
}

// Insert the key/values in kvs into the hashtable
__global__ void
k_hashtable_insert(KeyValue*__restrict__ hashtable,
                   const KeyValue*__restrict__ kvs,
                   unsigned int numkvs)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);

    while (true)
    {
      uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
      if (prev == kEmpty || prev == key)
      {
        hashtable[slot].value = value;
        return;
      }

      slot = (slot + 1) & (kHashTableCapacity-1);
    }
  }
}

double insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  k_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, kvs, (uint32_t)num_kvs);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  return time;
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__ void 
k_hashtable_delete(KeyValue*__restrict__ hashtable, 
                   const KeyValue*__restrict__ kvs,
                   unsigned int numkvs)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);

    while (true)
    {
      if (hashtable[slot].key == key)
      {
        hashtable[slot].value = kEmpty;
        return;
      }
      if (hashtable[slot].key == kEmpty)
      {
        return;
      }
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
  }
}

double delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  k_hashtable_delete<<<gridsize, threadblocksize>>>(pHashTable, kvs, (uint32_t)num_kvs);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__ void
k_iterate_hashtable(KeyValue*__restrict__ pHashTable,
                    KeyValue*__restrict__ kvs,
                    uint32_t* kvs_size)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < kHashTableCapacity) 
  {
    if (pHashTable[tid].key != kEmpty) 
    {
      uint32_t value = pHashTable[tid].value;
      if (value != kEmpty)
      {
        uint32_t size = atomicAdd(kvs_size, 1);
        kvs[size] = pHashTable[tid];
      }
    }
  }
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
  uint32_t* device_num_kvs;
  cudaMalloc((void**) &device_num_kvs, sizeof(uint32_t));
  cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

  KeyValue* device_kvs;
  cudaMalloc((void**) &device_kvs, sizeof(KeyValue) * kNumKeyValues);

  const int threadblocksize = 256;
  int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  k_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (iterate): %f (s)\n", time * 1e-9f);

  uint32_t num_kvs;
  cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

  std::vector<KeyValue> kvs;
  kvs.resize(num_kvs);

  cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

  cudaFree(device_kvs);
  cudaFree(device_num_kvs);

  return kvs;
}
