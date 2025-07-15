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

#ifndef _H_BB_COMPUT_S_KEYS
#define _H_BB_COMPUT_S_KEYS

#include <limits>

#include "bb_exch_keys.cuh"
#include "bb_comput_common.cuh"

template<class K, class Offset>
__global__
void gen_copy(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = gid;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        if(seg_size == 1)
        {
            keyB[k] = key[k];
        }
    }
}

/* block tcf subwarp coalesced quiet real_kern */
/*   256   1       2     false  true      true */
template<class K, class Offset>
__global__
void gen_bk256_wp2_tc1_r2_r2_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    K rg_k0 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        // sort 2 elements
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,
                        0x1,bit1);
        if((tid<<0)+0 <seg_size) keyB[k+(tid<<0)+0 ] = rg_k0 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   2       2     false  true      true */
template<class K, class Offset>
__global__
void gen_bk128_wp2_tc2_r3_r4_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:std::numeric_limits<K>::max();
        // sort 4 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
        if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       2     false  true      true */
template<class K, class Offset>
__global__
void gen_bk128_wp2_tc4_r5_r8_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>1);
    const int tid = (threadIdx.x & 1);
    const int bit1 = (tid>>0)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+2   <seg_size)?key[k+tid+2   ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+4   <seg_size)?key[k+tid+4   ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+6   <seg_size)?key[k+tid+6   ]:std::numeric_limits<K>::max();
        // sort 8 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       4      true  true      true */
template<class K, class Offset>
__global__
void gen_bk128_wp4_tc4_r9_r16_strd(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>2);
    const int tid = (threadIdx.x & 3);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int normalized_bin_size = (bin_size/8)*8;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+4   <seg_size)?key[k+tid+4   ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+8   <seg_size)?key[k+tid+8   ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+12  <seg_size)?key[k+tid+12  ]:std::numeric_limits<K>::max();
        // sort 16 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x3,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        if(lane_id&0x10) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x10) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        int kk;
        int ss;
        int base = (lane_id/16)*16;
        kk = __shfl_sync(0xffffffff,k, 0 );
        ss = __shfl_sync(0xffffffff,seg_size, 0 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl_sync(0xffffffff,k, 4 );
        ss = __shfl_sync(0xffffffff,seg_size, 4 );
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k0 ;
        kk = __shfl_sync(0xffffffff,k, 8 );
        ss = __shfl_sync(0xffffffff,seg_size, 8 );
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl_sync(0xffffffff,k, 12);
        ss = __shfl_sync(0xffffffff,seg_size, 12);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k2 ;
        kk = __shfl_sync(0xffffffff,k, 16);
        ss = __shfl_sync(0xffffffff,seg_size, 16);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl_sync(0xffffffff,k, 20);
        ss = __shfl_sync(0xffffffff,seg_size, 20);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k1 ;
        kk = __shfl_sync(0xffffffff,k, 24);
        ss = __shfl_sync(0xffffffff,seg_size, 24);
        if((lane_id>>4)==0&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
        kk = __shfl_sync(0xffffffff,k, 28);
        ss = __shfl_sync(0xffffffff,seg_size, 28);
        if((lane_id>>4)==1&&lane_id-base<ss) keyB[kk+lane_id-base] = rg_k3 ;
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4       8      true  true      true */
template<class K, class Offset>
__global__
void gen_bk128_wp8_tc4_r17_r32_strd(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:std::numeric_limits<K>::max();
        // sort 32 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x3,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x7,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        if(lane_id&0x10) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x10) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        int kk;
        int ss;
        kk = __shfl_sync(0xffffffff,k, 0 );
        ss = __shfl_sync(0xffffffff,seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        kk = __shfl_sync(0xffffffff,k, 8 );
        ss = __shfl_sync(0xffffffff,seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k2 ;
        kk = __shfl_sync(0xffffffff,k, 16);
        ss = __shfl_sync(0xffffffff,seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        kk = __shfl_sync(0xffffffff,k, 24);
        ss = __shfl_sync(0xffffffff,seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k3 ;
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   128   4      16      true  true      true */
template<class K, class Offset>
__global__
void gen_bk128_wp16_tc4_r33_r64_strd(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>4);
    const int tid = (threadIdx.x & 15);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int normalized_bin_size = (bin_size/2)*2;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+16  <seg_size)?key[k+tid+16  ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+32  <seg_size)?key[k+tid+32  ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+48  <seg_size)?key[k+tid+48  ]:std::numeric_limits<K>::max();
        // sort 64 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x3,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x7,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0xf,bit4);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x4,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x4 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x4 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k1 , rg_k3 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        if(lane_id&0x10) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x10) SWP_KEY(K, rg_k2 , rg_k3 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        int kk;
        int ss;
        kk = __shfl_sync(0xffffffff,k, 0 );
        ss = __shfl_sync(0xffffffff,seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        kk = __shfl_sync(0xffffffff,k, 16);
        ss = __shfl_sync(0xffffffff,seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
    } else if(bin_it < bin_size) {
        if((tid<<2)+0 <seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256  16       8      true  true      true */
template<class K, class Offset>
__global__
void gen_bk256_wp8_tc16_r65_r128_strd(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>3);
    const int tid = (threadIdx.x & 7);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    K rg_k4 ;
    K rg_k5 ;
    K rg_k6 ;
    K rg_k7 ;
    K rg_k8 ;
    K rg_k9 ;
    K rg_k10;
    K rg_k11;
    K rg_k12;
    K rg_k13;
    K rg_k14;
    K rg_k15;
    int normalized_bin_size = (bin_size/4)*4;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+8   <seg_size)?key[k+tid+8   ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+16  <seg_size)?key[k+tid+16  ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+24  <seg_size)?key[k+tid+24  ]:std::numeric_limits<K>::max();
        rg_k4  = (tid+32  <seg_size)?key[k+tid+32  ]:std::numeric_limits<K>::max();
        rg_k5  = (tid+40  <seg_size)?key[k+tid+40  ]:std::numeric_limits<K>::max();
        rg_k6  = (tid+48  <seg_size)?key[k+tid+48  ]:std::numeric_limits<K>::max();
        rg_k7  = (tid+56  <seg_size)?key[k+tid+56  ]:std::numeric_limits<K>::max();
        rg_k8  = (tid+64  <seg_size)?key[k+tid+64  ]:std::numeric_limits<K>::max();
        rg_k9  = (tid+72  <seg_size)?key[k+tid+72  ]:std::numeric_limits<K>::max();
        rg_k10 = (tid+80  <seg_size)?key[k+tid+80  ]:std::numeric_limits<K>::max();
        rg_k11 = (tid+88  <seg_size)?key[k+tid+88  ]:std::numeric_limits<K>::max();
        rg_k12 = (tid+96  <seg_size)?key[k+tid+96  ]:std::numeric_limits<K>::max();
        rg_k13 = (tid+104 <seg_size)?key[k+tid+104 ]:std::numeric_limits<K>::max();
        rg_k14 = (tid+112 <seg_size)?key[k+tid+112 ]:std::numeric_limits<K>::max();
        rg_k15 = (tid+120 <seg_size)?key[k+tid+120 ]:std::numeric_limits<K>::max();
        // sort 128 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k11);
        CMP_SWP_KEY(K,rg_k9 ,rg_k10);
        CMP_SWP_KEY(K,rg_k12,rg_k15);
        CMP_SWP_KEY(K,rg_k13,rg_k14);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k15);
        CMP_SWP_KEY(K,rg_k9 ,rg_k14);
        CMP_SWP_KEY(K,rg_k10,rg_k13);
        CMP_SWP_KEY(K,rg_k11,rg_k12);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k10);
        CMP_SWP_KEY(K,rg_k9 ,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k14);
        CMP_SWP_KEY(K,rg_k13,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k15);
        CMP_SWP_KEY(K,rg_k1 ,rg_k14);
        CMP_SWP_KEY(K,rg_k2 ,rg_k13);
        CMP_SWP_KEY(K,rg_k3 ,rg_k12);
        CMP_SWP_KEY(K,rg_k4 ,rg_k11);
        CMP_SWP_KEY(K,rg_k5 ,rg_k10);
        CMP_SWP_KEY(K,rg_k6 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k7 ,rg_k8 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k12);
        CMP_SWP_KEY(K,rg_k9 ,rg_k13);
        CMP_SWP_KEY(K,rg_k10,rg_k14);
        CMP_SWP_KEY(K,rg_k11,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k10);
        CMP_SWP_KEY(K,rg_k9 ,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k14);
        CMP_SWP_KEY(K,rg_k13,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k8 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k10);
        CMP_SWP_KEY(K,rg_k3 ,rg_k11);
        CMP_SWP_KEY(K,rg_k4 ,rg_k12);
        CMP_SWP_KEY(K,rg_k5 ,rg_k13);
        CMP_SWP_KEY(K,rg_k6 ,rg_k14);
        CMP_SWP_KEY(K,rg_k7 ,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k12);
        CMP_SWP_KEY(K,rg_k9 ,rg_k13);
        CMP_SWP_KEY(K,rg_k10,rg_k14);
        CMP_SWP_KEY(K,rg_k11,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k10);
        CMP_SWP_KEY(K,rg_k9 ,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k14);
        CMP_SWP_KEY(K,rg_k13,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x3,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k8 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k10);
        CMP_SWP_KEY(K,rg_k3 ,rg_k11);
        CMP_SWP_KEY(K,rg_k4 ,rg_k12);
        CMP_SWP_KEY(K,rg_k5 ,rg_k13);
        CMP_SWP_KEY(K,rg_k6 ,rg_k14);
        CMP_SWP_KEY(K,rg_k7 ,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k12);
        CMP_SWP_KEY(K,rg_k9 ,rg_k13);
        CMP_SWP_KEY(K,rg_k10,rg_k14);
        CMP_SWP_KEY(K,rg_k11,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k10);
        CMP_SWP_KEY(K,rg_k9 ,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k14);
        CMP_SWP_KEY(K,rg_k13,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x7,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        rg_k8 ,rg_k9 ,rg_k10,rg_k11,rg_k12,rg_k13,rg_k14,rg_k15,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k8 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k10);
        CMP_SWP_KEY(K,rg_k3 ,rg_k11);
        CMP_SWP_KEY(K,rg_k4 ,rg_k12);
        CMP_SWP_KEY(K,rg_k5 ,rg_k13);
        CMP_SWP_KEY(K,rg_k6 ,rg_k14);
        CMP_SWP_KEY(K,rg_k7 ,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k12);
        CMP_SWP_KEY(K,rg_k9 ,rg_k13);
        CMP_SWP_KEY(K,rg_k10,rg_k14);
        CMP_SWP_KEY(K,rg_k11,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k10);
        CMP_SWP_KEY(K,rg_k9 ,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k14);
        CMP_SWP_KEY(K,rg_k13,rg_k15);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k8 ,rg_k9 );
        CMP_SWP_KEY(K,rg_k10,rg_k11);
        CMP_SWP_KEY(K,rg_k12,rg_k13);
        CMP_SWP_KEY(K,rg_k14,rg_k15);
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x1 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x1 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k2 , rg_k3 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k4 , rg_k5 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k6 , rg_k7 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k8 , rg_k9 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k10, rg_k11);
        if(lane_id&0x1 ) SWP_KEY(K, rg_k12, rg_k13);
        if(lane_id&0x1 ) SWP_KEY(K, rg_k14, rg_k15);
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x1 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x1 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x2 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x2 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x2 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k1 , rg_k3 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k4 , rg_k6 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k5 , rg_k7 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k8 , rg_k10);
        if(lane_id&0x2 ) SWP_KEY(K, rg_k9 , rg_k11);
        if(lane_id&0x2 ) SWP_KEY(K, rg_k12, rg_k14);
        if(lane_id&0x2 ) SWP_KEY(K, rg_k13, rg_k15);
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x2 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x2 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x2 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x2 );
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x4 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x4 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x4 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k0 , rg_k4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k1 , rg_k5 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k2 , rg_k6 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k3 , rg_k7 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k8 , rg_k12);
        if(lane_id&0x4 ) SWP_KEY(K, rg_k9 , rg_k13);
        if(lane_id&0x4 ) SWP_KEY(K, rg_k10, rg_k14);
        if(lane_id&0x4 ) SWP_KEY(K, rg_k11, rg_k15);
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x4 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x4 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x4 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x4 );
        rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x8 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x8 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x8 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x8 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x8 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k0 , rg_k8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k1 , rg_k9 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k2 , rg_k10);
        if(lane_id&0x8 ) SWP_KEY(K, rg_k3 , rg_k11);
        if(lane_id&0x8 ) SWP_KEY(K, rg_k4 , rg_k12);
        if(lane_id&0x8 ) SWP_KEY(K, rg_k5 , rg_k13);
        if(lane_id&0x8 ) SWP_KEY(K, rg_k6 , rg_k14);
        if(lane_id&0x8 ) SWP_KEY(K, rg_k7 , rg_k15);
        rg_k8  = __shfl_xor_sync(0xffffffff,rg_k8 , 0x8 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x8 );
        rg_k10 = __shfl_xor_sync(0xffffffff,rg_k10, 0x8 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x8 );
        rg_k12 = __shfl_xor_sync(0xffffffff,rg_k12, 0x8 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x8 );
        rg_k14 = __shfl_xor_sync(0xffffffff,rg_k14, 0x8 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x8 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x10);
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x10);
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x10);
        if(lane_id&0x10) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x10) SWP_KEY(K, rg_k2 , rg_k3 );
        if(lane_id&0x10) SWP_KEY(K, rg_k4 , rg_k5 );
        if(lane_id&0x10) SWP_KEY(K, rg_k6 , rg_k7 );
        if(lane_id&0x10) SWP_KEY(K, rg_k8 , rg_k9 );
        if(lane_id&0x10) SWP_KEY(K, rg_k10, rg_k11);
        if(lane_id&0x10) SWP_KEY(K, rg_k12, rg_k13);
        if(lane_id&0x10) SWP_KEY(K, rg_k14, rg_k15);
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x10);
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x10);
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x10);
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x10);
        int kk;
        int ss;
        kk = __shfl_sync(0xffffffff,k, 0 );
        ss = __shfl_sync(0xffffffff,seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k2 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k4 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k6 ;
        kk = __shfl_sync(0xffffffff,k, 8 );
        ss = __shfl_sync(0xffffffff,seg_size, 8 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k8 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k10;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k12;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k14;
        kk = __shfl_sync(0xffffffff,k, 16);
        ss = __shfl_sync(0xffffffff,seg_size, 16);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k1 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k3 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k5 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k7 ;
        kk = __shfl_sync(0xffffffff,k, 24);
        ss = __shfl_sync(0xffffffff,seg_size, 24);
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k9 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k11;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k13;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k15;
    } else if(bin_it < bin_size) {
        if((tid<<4)+0 <seg_size) keyB[k+(tid<<4)+0 ] = rg_k0 ;
        if((tid<<4)+1 <seg_size) keyB[k+(tid<<4)+1 ] = rg_k1 ;
        if((tid<<4)+2 <seg_size) keyB[k+(tid<<4)+2 ] = rg_k2 ;
        if((tid<<4)+3 <seg_size) keyB[k+(tid<<4)+3 ] = rg_k3 ;
        if((tid<<4)+4 <seg_size) keyB[k+(tid<<4)+4 ] = rg_k4 ;
        if((tid<<4)+5 <seg_size) keyB[k+(tid<<4)+5 ] = rg_k5 ;
        if((tid<<4)+6 <seg_size) keyB[k+(tid<<4)+6 ] = rg_k6 ;
        if((tid<<4)+7 <seg_size) keyB[k+(tid<<4)+7 ] = rg_k7 ;
        if((tid<<4)+8 <seg_size) keyB[k+(tid<<4)+8 ] = rg_k8 ;
        if((tid<<4)+9 <seg_size) keyB[k+(tid<<4)+9 ] = rg_k9 ;
        if((tid<<4)+10<seg_size) keyB[k+(tid<<4)+10] = rg_k10;
        if((tid<<4)+11<seg_size) keyB[k+(tid<<4)+11] = rg_k11;
        if((tid<<4)+12<seg_size) keyB[k+(tid<<4)+12] = rg_k12;
        if((tid<<4)+13<seg_size) keyB[k+(tid<<4)+13] = rg_k13;
        if((tid<<4)+14<seg_size) keyB[k+(tid<<4)+14] = rg_k14;
        if((tid<<4)+15<seg_size) keyB[k+(tid<<4)+15] = rg_k15;
    }
}
/* block tcf subwarp coalesced quiet real_kern */
/*   256   8      32      true  true      true */
template<class K, class Offset>
__global__
void gen_bk256_wp32_tc8_r129_r256_strd(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    const int bin_it = (gid>>5);
    const int tid = (threadIdx.x & 31);
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    K rg_k4 ;
    K rg_k5 ;
    K rg_k6 ;
    K rg_k7 ;
    int normalized_bin_size = (bin_size/1)*1;
    int k;
    int seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        rg_k0  = (tid+0   <seg_size)?key[k+tid+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (tid+32  <seg_size)?key[k+tid+32  ]:std::numeric_limits<K>::max();
        rg_k2  = (tid+64  <seg_size)?key[k+tid+64  ]:std::numeric_limits<K>::max();
        rg_k3  = (tid+96  <seg_size)?key[k+tid+96  ]:std::numeric_limits<K>::max();
        rg_k4  = (tid+128 <seg_size)?key[k+tid+128 ]:std::numeric_limits<K>::max();
        rg_k5  = (tid+160 <seg_size)?key[k+tid+160 ]:std::numeric_limits<K>::max();
        rg_k6  = (tid+192 <seg_size)?key[k+tid+192 ]:std::numeric_limits<K>::max();
        rg_k7  = (tid+224 <seg_size)?key[k+tid+224 ]:std::numeric_limits<K>::max();
        // sort 256 elements
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k6 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k7 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k4 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x3,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x7,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0xf,bit4);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x4,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
        // exch_intxn: generate exch_intxn_keys()
        exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1f,bit5);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x8,bit4);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x4,bit3);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x2,bit2);
        // exch_paral: generate exch_paral_keys()
        exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,rg_k4 ,rg_k5 ,rg_k6 ,rg_k7 ,
                        0x1,bit1);
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k4 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k3 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
        CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k6 );
        CMP_SWP_KEY(K,rg_k5 ,rg_k7 );
        // exch_paral: switch to exch_local()
        CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        CMP_SWP_KEY(K,rg_k4 ,rg_k5 );
        CMP_SWP_KEY(K,rg_k6 ,rg_k7 );
    }

    if(bin_it < normalized_bin_size) {
        // store back the results
        int lane_id = threadIdx.x & 31;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k2 , rg_k3 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k4 , rg_k5 );
        if(lane_id&0x1 ) SWP_KEY(K, rg_k6 , rg_k7 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k1 , rg_k3 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k4 , rg_k6 );
        if(lane_id&0x2 ) SWP_KEY(K, rg_k5 , rg_k7 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x2 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x2 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x2 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x2 );
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k0 , rg_k4 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k1 , rg_k5 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k2 , rg_k6 );
        if(lane_id&0x4 ) SWP_KEY(K, rg_k3 , rg_k7 );
        rg_k4  = __shfl_xor_sync(0xffffffff,rg_k4 , 0x4 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x4 );
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x4 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x4 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k0 , rg_k1 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k2 , rg_k3 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k4 , rg_k5 );
        if(lane_id&0x8 ) SWP_KEY(K, rg_k6 , rg_k7 );
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x8 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x8 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x8 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x8 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        if(lane_id&0x10) SWP_KEY(K, rg_k0 , rg_k2 );
        if(lane_id&0x10) SWP_KEY(K, rg_k1 , rg_k3 );
        if(lane_id&0x10) SWP_KEY(K, rg_k4 , rg_k6 );
        if(lane_id&0x10) SWP_KEY(K, rg_k5 , rg_k7 );
        rg_k2  = __shfl_xor_sync(0xffffffff,rg_k2 , 0x10);
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x10);
        rg_k6  = __shfl_xor_sync(0xffffffff,rg_k6 , 0x10);
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x10);
        int kk;
        int ss;
        kk = __shfl_sync(0xffffffff,k, 0 );
        ss = __shfl_sync(0xffffffff,seg_size, 0 );
        if(lane_id+0  <ss) keyB[kk+lane_id+0  ] = rg_k0 ;
        if(lane_id+32 <ss) keyB[kk+lane_id+32 ] = rg_k4 ;
        if(lane_id+64 <ss) keyB[kk+lane_id+64 ] = rg_k1 ;
        if(lane_id+96 <ss) keyB[kk+lane_id+96 ] = rg_k5 ;
        if(lane_id+128<ss) keyB[kk+lane_id+128] = rg_k2 ;
        if(lane_id+160<ss) keyB[kk+lane_id+160] = rg_k6 ;
        if(lane_id+192<ss) keyB[kk+lane_id+192] = rg_k3 ;
        if(lane_id+224<ss) keyB[kk+lane_id+224] = rg_k7 ;
    } else if(bin_it < bin_size) {
        if((tid<<3)+0 <seg_size) keyB[k+(tid<<3)+0 ] = rg_k0 ;
        if((tid<<3)+1 <seg_size) keyB[k+(tid<<3)+1 ] = rg_k1 ;
        if((tid<<3)+2 <seg_size) keyB[k+(tid<<3)+2 ] = rg_k2 ;
        if((tid<<3)+3 <seg_size) keyB[k+(tid<<3)+3 ] = rg_k3 ;
        if((tid<<3)+4 <seg_size) keyB[k+(tid<<3)+4 ] = rg_k4 ;
        if((tid<<3)+5 <seg_size) keyB[k+(tid<<3)+5 ] = rg_k5 ;
        if((tid<<3)+6 <seg_size) keyB[k+(tid<<3)+6 ] = rg_k6 ;
        if((tid<<3)+7 <seg_size) keyB[k+(tid<<3)+7 ] = rg_k7 ;
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   128    2    4  true      true */
template<class K, class Offset>
__global__
void gen_bk128_tc4_r257_r512_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ K smem[512];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:std::numeric_limits<K>::max();
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:std::numeric_limits<K>::max();
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:std::numeric_limits<K>::max();
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:std::numeric_limits<K>::max();
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Merge in 2 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        K tmp_k0;
        K tmp_k1;
        K *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   256    2    4  true      true */
template<class K, class Offset>
__global__
void gen_bk256_tc4_r513_r1024_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ K smem[1024];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:std::numeric_limits<K>::max();
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:std::numeric_limits<K>::max();
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:std::numeric_limits<K>::max();
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:std::numeric_limits<K>::max();
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Merge in 3 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        K tmp_k0;
        K tmp_k1;
        K *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
        }
    }
}
/* block tcf1 tcf2 quiet real_kern */
/*   512    2    4  true      true */
template<class K, class Offset>
__global__
void gen_bk512_tc4_r1025_r2048_orig(
    K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends,
    const int *bin, const int bin_size)
{
    const int tid = threadIdx.x;
    const int bin_it = blockIdx.x;
    __shared__ K smem[2048];
    const int bit1 = (tid>>0)&0x1;
    const int bit2 = (tid>>1)&0x1;
    const int bit3 = (tid>>2)&0x1;
    const int bit4 = (tid>>3)&0x1;
    const int bit5 = (tid>>4)&0x1;
    const int tid1 = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    K rg_k0 ;
    K rg_k1 ;
    K rg_k2 ;
    K rg_k3 ;
    int k;
    int seg_size;
    int ext_seg_size;
    if(bin_it < bin_size) {
        k = seg_begins[bin[bin_it]];
        seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
        ext_seg_size = ((seg_size + 63) / 64) * 64;
        int big_wp = (ext_seg_size - blockDim.x * 2) / 64;
        int sml_wp = blockDim.x / 32 - big_wp;
        int sml_len = sml_wp * 64;
        const int big_warp_id = (warp_id - sml_wp < 0)? 0: warp_id - sml_wp;
        bool sml_warp = warp_id < sml_wp;
        if(sml_warp) {
            rg_k0 = key[k+(warp_id<<6)+tid1+0   ];
            rg_k1 = key[k+(warp_id<<6)+tid1+32  ];
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
        } else {
            rg_k0  = (sml_len+tid1+(big_warp_id<<7)+0   <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+0   ]:std::numeric_limits<K>::max();
            rg_k1  = (sml_len+tid1+(big_warp_id<<7)+32  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+32  ]:std::numeric_limits<K>::max();
            rg_k2  = (sml_len+tid1+(big_warp_id<<7)+64  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+64  ]:std::numeric_limits<K>::max();
            rg_k3  = (sml_len+tid1+(big_warp_id<<7)+96  <seg_size)?key[k+sml_len+tid1+(big_warp_id<<7)+96  ]:std::numeric_limits<K>::max();
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k3 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k2 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x3,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x7,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0xf,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
            // exch_intxn: generate exch_intxn_keys()
            exch_intxn_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1f,bit5);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x8,bit4);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x4,bit3);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x2,bit2);
            // exch_paral: generate exch_paral_keys()
            exch_paral_keys(rg_k0 ,rg_k1 ,rg_k2 ,rg_k3 ,
                            0x1,bit1);
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k2 );
            CMP_SWP_KEY(K,rg_k1 ,rg_k3 );
            // exch_paral: switch to exch_local()
            CMP_SWP_KEY(K,rg_k0 ,rg_k1 );
            CMP_SWP_KEY(K,rg_k2 ,rg_k3 );
        }
        // Store register results to shared memory
        if(sml_warp) {
            smem[(warp_id<<6)+(tid1<<1)+0 ] = rg_k0 ;
            smem[(warp_id<<6)+(tid1<<1)+1 ] = rg_k1 ;
        } else {
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+0 ] = rg_k0 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+1 ] = rg_k1 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+2 ] = rg_k2 ;
            smem[sml_len+(big_warp_id<<7)+(tid1<<2)+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Merge in 4 steps
        int grp_start_wp_id;
        int grp_start_off;
        int tmp_wp_id;
        int lhs_len;
        int rhs_len;
        int gran;
        int s_a;
        int s_b;
        bool p;
        K tmp_k0;
        K tmp_k1;
        K *start;
        // Step 0
        grp_start_wp_id = ((warp_id>>1)<<1);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+1<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&3)==0){
            gran += 0;
        }
        if((warp_id&3)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&3)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&3)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&7)==0){
            gran += 0;
        }
        if((warp_id&7)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&7)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&7)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&7)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&7)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&7)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&7)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        __syncthreads();
        // Store merged results back to shared memory
        if(sml_warp){
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
        } else {
            smem[grp_start_off+gran+0 ] = rg_k0 ;
            smem[grp_start_off+gran+1 ] = rg_k1 ;
            smem[grp_start_off+gran+2 ] = rg_k2 ;
            smem[grp_start_off+gran+3 ] = rg_k3 ;
        }
        __syncthreads();
        // Step 3
        grp_start_wp_id = ((warp_id>>4)<<4);
        grp_start_off = (grp_start_wp_id<sml_wp)?grp_start_wp_id*64:sml_len+(grp_start_wp_id-sml_wp)*128;
        tmp_wp_id = grp_start_wp_id;
        lhs_len = ((tmp_wp_id+0<sml_wp)?64  :128 )+
                  ((tmp_wp_id+1<sml_wp)?64  :128 )+
                  ((tmp_wp_id+2<sml_wp)?64  :128 )+
                  ((tmp_wp_id+3<sml_wp)?64  :128 )+
                  ((tmp_wp_id+4<sml_wp)?64  :128 )+
                  ((tmp_wp_id+5<sml_wp)?64  :128 )+
                  ((tmp_wp_id+6<sml_wp)?64  :128 )+
                  ((tmp_wp_id+7<sml_wp)?64  :128 );
        rhs_len = ((tmp_wp_id+8<sml_wp)?64  :128 )+
                  ((tmp_wp_id+9<sml_wp)?64  :128 )+
                  ((tmp_wp_id+10<sml_wp)?64  :128 )+
                  ((tmp_wp_id+11<sml_wp)?64  :128 )+
                  ((tmp_wp_id+12<sml_wp)?64  :128 )+
                  ((tmp_wp_id+13<sml_wp)?64  :128 )+
                  ((tmp_wp_id+14<sml_wp)?64  :128 )+
                  ((tmp_wp_id+15<sml_wp)?64  :128 );
        gran = (warp_id<sml_wp)?(tid1<<1): (tid1<<2);
        if((warp_id&15)==0){
            gran += 0;
        }
        if((warp_id&15)==1){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 );
        }
        if((warp_id&15)==2){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 );
        }
        if((warp_id&15)==3){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 );
        }
        if((warp_id&15)==4){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 );
        }
        if((warp_id&15)==5){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 );
        }
        if((warp_id&15)==6){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 );
        }
        if((warp_id&15)==7){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 );
        }
        if((warp_id&15)==8){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 );
        }
        if((warp_id&15)==9){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 );
        }
        if((warp_id&15)==10){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 );
        }
        if((warp_id&15)==11){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 );
        }
        if((warp_id&15)==12){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 );
        }
        if((warp_id&15)==13){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 );
        }
        if((warp_id&15)==14){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 );
        }
        if((warp_id&15)==15){
            gran += ((tmp_wp_id+0<sml_wp)?64  :128 )+
                    ((tmp_wp_id+1<sml_wp)?64  :128 )+
                    ((tmp_wp_id+2<sml_wp)?64  :128 )+
                    ((tmp_wp_id+3<sml_wp)?64  :128 )+
                    ((tmp_wp_id+4<sml_wp)?64  :128 )+
                    ((tmp_wp_id+5<sml_wp)?64  :128 )+
                    ((tmp_wp_id+6<sml_wp)?64  :128 )+
                    ((tmp_wp_id+7<sml_wp)?64  :128 )+
                    ((tmp_wp_id+8<sml_wp)?64  :128 )+
                    ((tmp_wp_id+9<sml_wp)?64  :128 )+
                    ((tmp_wp_id+10<sml_wp)?64  :128 )+
                    ((tmp_wp_id+11<sml_wp)?64  :128 )+
                    ((tmp_wp_id+12<sml_wp)?64  :128 )+
                    ((tmp_wp_id+13<sml_wp)?64  :128 )+
                    ((tmp_wp_id+14<sml_wp)?64  :128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
        if(sml_warp){
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
        } else {
            tmp_k0 = (s_a<lhs_len        )?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k0 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k1 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k2 = p ? tmp_k0 : tmp_k1;
            if(p) {
                ++s_a;
                tmp_k0 = (s_a<lhs_len)?smem[grp_start_off+s_a]:std::numeric_limits<K>::max();
            } else {
                ++s_b;
                tmp_k1 = (s_b<lhs_len+rhs_len)?smem[grp_start_off+s_b]:std::numeric_limits<K>::max();
            }
            p = (s_b>=lhs_len+rhs_len)||((s_a<lhs_len)&&(tmp_k0<=tmp_k1));
            rg_k3 = p ? tmp_k0 : tmp_k1;
        }
        if(sml_warp){
        } else {
        }
        if(sml_warp){
            if((tid<<1)+0 <seg_size) keyB[k+(tid<<1)+0 ] = rg_k0 ;
            if((tid<<1)+1 <seg_size) keyB[k+(tid<<1)+1 ] = rg_k1 ;
        } else {
            if((tid<<2)+0 -sml_len<seg_size) keyB[k+(tid<<2)+0 -sml_len] = rg_k0 ;
            if((tid<<2)+1 -sml_len<seg_size) keyB[k+(tid<<2)+1 -sml_len] = rg_k1 ;
            if((tid<<2)+2 -sml_len<seg_size) keyB[k+(tid<<2)+2 -sml_len] = rg_k2 ;
            if((tid<<2)+3 -sml_len<seg_size) keyB[k+(tid<<2)+3 -sml_len] = rg_k3 ;
        }
    }
}

#endif
