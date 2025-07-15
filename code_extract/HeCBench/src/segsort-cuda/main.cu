/*
 * (c) 2015-2019 Virginia Polytechnic Institute & State University (Virginia Tech)
 *          2020 Robin Kobus (kobus@uni-mainz.de)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, version 2.1
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License, version 2.1, for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda.h>
#include "src/bb_segsort.cuh"
#include "src/bb_segsort_keys.cuh"

using std::vector;
using std::pair;

using index_t = int;
using offset_t = int;
using key_t = int;
using val_t = uint64_t;


#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
  std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
  return 0; }


template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, const vector<offset_t> &seg, index_t num_segs)
{
  vector<pair<K,T>> pairs;
  for(index_t i = 0; i < num_segs; i++)
  {
    offset_t st = seg[i];
    offset_t ed = seg[i+1];
    offset_t size = ed - st;
    pairs.reserve(size);
    for(index_t j = st; j < ed; j++)
    {
      pairs.emplace_back(key[j], val[j]);
    }
    stable_sort(pairs.begin(), pairs.end(), [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
    // sort(pairs.begin(), pairs.begin(), [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});

    for(index_t j = st; j < ed; j++)
    {
      key[j] = pairs[j-st].first;
      val[j] = pairs[j-st].second;
    }
    pairs.clear();
  }
}


template<class K>
void gold_segsort(vector<K> &key, const vector<offset_t> &seg, index_t num_segs)
{
  for(index_t i = 0; i < num_segs; i++)
  {
    offset_t st = seg[i];
    offset_t ed = seg[i+1];
    stable_sort(key.begin()+st, key.begin()+ed, [&](K a, K b){ return a < b;});
    // sort(key.begin()+st, key.begin()+ed, [&](K a, K b){ return a < b;});
  }
}


template<class K, class T>
void sort_vals_of_same_key(const vector<K> &key, vector<T> &val, const vector<offset_t> &seg, index_t num_segs)
{
  for(index_t i = 0; i < num_segs; i++)
  {
    offset_t st = seg[i];
    offset_t ed = seg[i+1];
    for(offset_t i = st; i < ed;)
    {
      auto range = std::equal_range(key.begin()+st, key.begin()+ed, key[i]);
      offset_t i_st = std::distance(key.begin(), range.first);
      offset_t i_ed = std::distance(key.begin(), range.second);
      sort(val.begin()+i_st, val.begin()+i_ed, [&](T a, T b){ return a < b;});
      i += i_ed - i_st;
    }
  }
}


int show_mem_usage()
{
  size_t free_byte ;
  size_t total_byte ;
  cudaError_t err = cudaMemGetInfo(&free_byte, &total_byte);
  CUDA_CHECK(err, "check memory info.");

  size_t used_byte  = total_byte - free_byte;
  printf("GPU memory usage: used = %4.2lf MB, free = %4.2lf MB, total = %4.2lf MB\n",
      used_byte/1024.0/1024.0, free_byte/1024.0/1024.0, total_byte/1024.0/1024.0);
  return cudaSuccess;
}


int segsort(index_t num_elements, bool keys_only = true)
{
  int seed = -278642091;
  std::cout << "seed: " << seed << '\n';
  std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dis(0, num_elements);

  cudaError_t err;

  vector<key_t> key(num_elements, 0);
  for(auto &k: key)
    k = dis(gen);

  offset_t max_seg_sz = 5000;
  offset_t min_seg_sz = 0;
  std::uniform_int_distribution<> seg_dis(min_seg_sz, max_seg_sz);
  vector<offset_t> seg;
  offset_t off = 0;
  while(off < num_elements)
  {
    seg.push_back(off);
    offset_t sz = seg_dis(gen);
    off = seg.back()+sz;
  }
  seg.push_back(num_elements);

  index_t num_segs = seg.size()-1;
  printf("synthesized segments # %d (max_size: %d, min_size: %d)\n", num_segs, max_seg_sz, min_seg_sz);

  vector<val_t> val;
  if(!keys_only)
  {
    val.resize(num_elements, 0);
    for(auto &v: val)
      v = (val_t)(dis(gen));
  }

  // cout << "key:\n"; for(auto k: key) cout << k << ", "; cout << endl;
  // cout << "val:\n"; for(auto v: val) cout << v << ", "; cout << endl;
  // cout << "seg:\n"; for(auto s: seg) cout << s << ", "; cout << endl;

  key_t *key_d;
  err = cudaMalloc((void**)&key_d, sizeof(key_t)*num_elements);
  CUDA_CHECK(err, "alloc key_d");
  err = cudaMemcpy(key_d, &key[0], sizeof(key_t)*num_elements, cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to key_d");

  val_t *val_d;
  if(!keys_only)
  {
    err = cudaMalloc((void**)&val_d, sizeof(val_t)*num_elements);
    CUDA_CHECK(err, "alloc val_d");
    err = cudaMemcpy(val_d, &val[0], sizeof(val_t)*num_elements, cudaMemcpyHostToDevice);
    CUDA_CHECK(err, "copy to val_d");
  }

  offset_t *seg_d;
  err = cudaMalloc((void**)&seg_d, sizeof(offset_t)*(num_segs+1));
  CUDA_CHECK(err, "alloc seg_d");
  err = cudaMemcpy(seg_d, &seg[0], sizeof(offset_t)*(num_segs+1), cudaMemcpyHostToDevice);
  CUDA_CHECK(err, "copy to seg_d");

  show_mem_usage();

  if(keys_only)
    gold_segsort(key, seg, num_segs);
  else
    gold_segsort(key, val, seg, num_segs);

  // cout << "key:\n"; for(auto k: key) cout << k << ", "; cout << endl;
  // cout << "val:\n"; for(auto v: val) cout << v << ", "; cout << endl;

  if(keys_only)
    bb_segsort(key_d, num_elements, seg_d, seg_d+1, num_segs);
  else
    bb_segsort(key_d, val_d, num_elements, seg_d, seg_d+1, num_segs);

  vector<key_t> key_h(num_elements, 0);
  err = cudaMemcpy(&key_h[0], key_d, sizeof(key_t)*num_elements, cudaMemcpyDeviceToHost);
  CUDA_CHECK(err, "copy from key_d");

  vector<val_t> val_h;
  if(!keys_only)
  {
    val_h.resize(num_elements, 0);
    err = cudaMemcpy(&val_h[0], val_d, sizeof(val_t)*num_elements, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err, "copy from val_d");
  }

  // cout << "key_h:\n"; for(auto k: key_h) cout << k << ", "; cout << endl;
  // cout << "val_h:\n"; for(auto v: val_h) cout << v << ", "; cout << endl;

  index_t cnt = 0;
  for(index_t i = 0; i < num_elements; i++)
    if(key[i] != key_h[i]) cnt++;
  if(cnt != 0) printf("[NOT PASSED] checking keys: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/num_elements);
  else printf("[PASSED] checking keys\n");

  if(!keys_only)
  {
    sort_vals_of_same_key(key, val, seg, num_segs);
    sort_vals_of_same_key(key, val_h, seg, num_segs);
    cnt = 0;
    for(index_t i = 0; i < num_elements; i++)
      if(val[i] != val_h[i]) cnt++;
    if(cnt != 0) printf("[NOT PASSED] checking vals: #err = %i (%4.2f%% #nnz)\n", cnt, 100.0*(double)cnt/num_elements);
    else printf("[PASSED] checking vals\n");
  }

  err = cudaFree(key_d);
  CUDA_CHECK(err, "free key_d");
  if(!keys_only) {
    err = cudaFree(val_d);
    CUDA_CHECK(err, "free val_d");
  }
  err = cudaFree(seg_d);
  CUDA_CHECK(err, "free seg_d");

  return 0;
}

int segsort_keys(index_t num_elements)
{
  return segsort(num_elements, true);
}

int segsort_pairs(index_t num_elements)
{
  return segsort(num_elements, false);
}


int main()
{
  index_t num_elements = 1UL << 25;
  printf("The number of elements is %d\n", num_elements);

  printf("Running key only test\n");
  segsort_keys(num_elements);

  printf("\nRunning key-value test\n");
  segsort_pairs(num_elements);

  return 0;
}
