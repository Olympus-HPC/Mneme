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

#ifndef _H_BB_SEGSORT
#define _H_BB_SEGSORT

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#include "bb_bin.cuh"
#include "bb_comput_s.cuh"
#include "bb_comput_l.cuh"

#include "bb_segsort_common.cuh"

#define CUDA_CHECK(_e, _s) if(_e != cudaSuccess) { \
        std::cout << "CUDA error (" << _s << "): " << cudaGetErrorString(_e) << std::endl; \
        return 0; }


template<class K, class T, class Offset>
void dispatch_kernels(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
    const Offset *d_seg_begins, const Offset *d_seg_ends,
    const int *d_bin_segs_id, const int *h_bin_counter, const int max_segsize,
    cudaStream_t stream)
{
    int subwarp_size, subwarp_num, factor;
    int threads_per_block;
    int num_blocks;

    threads_per_block = 256;
    subwarp_num = h_bin_counter[1]-h_bin_counter[0];
    num_blocks = (subwarp_num+threads_per_block-1)/threads_per_block;
    if(subwarp_num > 0)
    gen_copy<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[0], subwarp_num);

    threads_per_block = 256;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[2]-h_bin_counter[1];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp2_tc1_r2_r2_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[1], subwarp_num);

    threads_per_block = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[3]-h_bin_counter[2];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc2_r3_r4_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[2], subwarp_num);

    threads_per_block = 128;
    subwarp_size = 2;
    subwarp_num = h_bin_counter[4]-h_bin_counter[3];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp2_tc4_r5_r8_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[3], subwarp_num);

    threads_per_block = 128;
    subwarp_size = 4;
    subwarp_num = h_bin_counter[5]-h_bin_counter[4];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp4_tc4_r9_r16_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[4], subwarp_num);

    threads_per_block = 128;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[6]-h_bin_counter[5];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp8_tc4_r17_r32_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[5], subwarp_num);

    threads_per_block = 128;
    subwarp_size = 16;
    subwarp_num = h_bin_counter[7]-h_bin_counter[6];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk128_wp16_tc4_r33_r64_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[6], subwarp_num);

    threads_per_block = 256;
    subwarp_size = 8;
    subwarp_num = h_bin_counter[8]-h_bin_counter[7];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp8_tc16_r65_r128_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[7], subwarp_num);

    threads_per_block = 256;
    subwarp_size = 32;
    subwarp_num = h_bin_counter[9]-h_bin_counter[8];
    factor = threads_per_block/subwarp_size;
    num_blocks = (subwarp_num+factor-1)/factor;
    if(subwarp_num > 0)
    gen_bk256_wp32_tc8_r129_r256_strd<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[8], subwarp_num);

    threads_per_block = 128;
    subwarp_num = h_bin_counter[10]-h_bin_counter[9];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    gen_bk128_tc4_r257_r512_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[9], subwarp_num);

    threads_per_block = 256;
    subwarp_num = h_bin_counter[11]-h_bin_counter[10];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    gen_bk256_tc4_r513_r1024_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[10], subwarp_num);

    threads_per_block = 512;
    subwarp_num = h_bin_counter[12]-h_bin_counter[11];
    num_blocks = subwarp_num;
    if(subwarp_num > 0)
    gen_bk512_tc4_r1025_r2048_orig<<<num_blocks, threads_per_block, 0, stream>>>(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[11], subwarp_num);

    // sort long segments
    subwarp_num = h_bin_counter[13]-h_bin_counter[12];
    if(subwarp_num > 0)
    gen_grid_kern_r2049(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id+h_bin_counter[12], subwarp_num, max_segsize,
        stream);
}


template<class K, class T, class Offset>
void bb_segsort_run(
    K *keys_d, T *vals_d, K *keysB_d, T *valsB_d,
    const Offset *d_seg_begins, const Offset *d_seg_ends, const int num_segs,
    int *d_bin_segs_id, int *h_bin_counter, int *d_bin_counter,
    cudaStream_t stream, cudaEvent_t event)
{
    bb_bin(d_seg_begins, d_seg_ends, num_segs,
        d_bin_segs_id, d_bin_counter, h_bin_counter,
        stream, event);

    int max_segsize = h_bin_counter[13];
    // std::cout << "max segsize: " << max_segsize << '\n';
    h_bin_counter[13] = num_segs;

    dispatch_kernels(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, d_bin_segs_id, h_bin_counter, max_segsize,
        stream);
}


template<class K, class T, class Offset>
int bb_segsort(
    K * & keys_d, T * & vals_d, const int num_elements,
    const Offset *d_seg_begins, const Offset *d_seg_ends, const int num_segs)
{
    cudaError_t cuda_err;

    int *h_bin_counter;
    int *d_bin_counter;
    int *d_bin_segs_id;
    cuda_err = cudaMallocHost((void **)&h_bin_counter, (SEGBIN_NUM+1) * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc h_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_counter, (SEGBIN_NUM+1) * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_counter");
    cuda_err = cudaMalloc((void **)&d_bin_segs_id, num_segs * sizeof(int));
    CUDA_CHECK(cuda_err, "alloc d_bin_segs_id");

    K *keysB_d;
    T *valsB_d;
    cuda_err = cudaMalloc((void **)&keysB_d, num_elements * sizeof(K));
    CUDA_CHECK(cuda_err, "alloc keysB_d");
    cuda_err = cudaMalloc((void **)&valsB_d, num_elements * sizeof(T));
    CUDA_CHECK(cuda_err, "alloc valsB_d");

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t event;
    cudaEventCreate(&event);

    auto start = std::chrono::steady_clock::now();

    bb_segsort_run(
        keys_d, vals_d, keysB_d, valsB_d,
        d_seg_begins, d_seg_ends, num_segs,
        d_bin_segs_id, h_bin_counter, d_bin_counter,
        stream, event);

    cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time: %f (s)\n", time * 1e-9f);

    std::swap(keys_d, keysB_d);
    std::swap(vals_d, valsB_d);

    cuda_err = cudaFreeHost(h_bin_counter);
    CUDA_CHECK(cuda_err, "free h_bin_counter");
    cuda_err = cudaFree(d_bin_counter);
    CUDA_CHECK(cuda_err, "free d_bin_counter");
    cuda_err = cudaFree(d_bin_segs_id);
    CUDA_CHECK(cuda_err, "free d_bin_segs_id");
    cuda_err = cudaFree(keysB_d);
    CUDA_CHECK(cuda_err, "free keysB");
    cuda_err = cudaFree(valsB_d);
    CUDA_CHECK(cuda_err, "free valsB");

    cudaEventDestroy(event);
    cudaStreamDestroy(stream);
    return 1;
}

#endif
