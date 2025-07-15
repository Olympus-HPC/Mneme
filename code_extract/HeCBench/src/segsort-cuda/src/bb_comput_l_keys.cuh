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

#ifndef _H_BB_COMPUT_L_KEYS
#define _H_BB_COMPUT_L_KEYS

#include <limits>

#include "bb_exch_keys.cuh"
#include "bb_comput_common.cuh"

template<class K, class Offset>
__global__
void kern_block_sort(
    const K *key, K *keyB,
    const Offset *seg_begins, const Offset *seg_ends, const int *bin,
    const int workloads_per_block)
{
    const int bin_it = blockIdx.x;
    const int innerbid = blockIdx.y;

    const int seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
    const int blk_stat = (seg_size+workloads_per_block-1)/workloads_per_block;

    if(innerbid < blk_stat)
    {
        const int tid = threadIdx.x;
        __shared__ K smem[2048];
        const int bit1 = (tid>>0)&0x1;
        const int bit2 = (tid>>1)&0x1;
        const int bit3 = (tid>>2)&0x1;
        const int bit4 = (tid>>3)&0x1;
        const int bit5 = (tid>>4)&0x1;
        const int warp_lane = threadIdx.x & 31;
        const int warp_id = threadIdx.x / 32;
        K rg_k0 ;
        K rg_k1 ;
        K rg_k2 ;
        K rg_k3 ;
        // int k;
        // int ext_seg_size;
        /*** codegen ***/
        int k = seg_begins[bin[bin_it]];
        k = k + (innerbid<<11);
        int inner_seg_size = min(seg_size-(innerbid<<11), 2048);
        /*** codegen ***/
        rg_k0  = (warp_lane+(warp_id<<7)+0   <inner_seg_size)?key[k+warp_lane+(warp_id<<7)+0   ]:std::numeric_limits<K>::max();
        rg_k1  = (warp_lane+(warp_id<<7)+32  <inner_seg_size)?key[k+warp_lane+(warp_id<<7)+32  ]:std::numeric_limits<K>::max();
        rg_k2  = (warp_lane+(warp_id<<7)+64  <inner_seg_size)?key[k+warp_lane+(warp_id<<7)+64  ]:std::numeric_limits<K>::max();
        rg_k3  = (warp_lane+(warp_id<<7)+96  <inner_seg_size)?key[k+warp_lane+(warp_id<<7)+96  ]:std::numeric_limits<K>::max();
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

        smem[(warp_id<<7)+(warp_lane<<2)+0 ] = rg_k0 ;
        smem[(warp_id<<7)+(warp_lane<<2)+1 ] = rg_k1 ;
        smem[(warp_id<<7)+(warp_lane<<2)+2 ] = rg_k2 ;
        smem[(warp_id<<7)+(warp_lane<<2)+3 ] = rg_k3 ;
        __syncthreads();
        // Merge in 4 steps
        int grp_start_wp_id;
        int grp_start_off;
        // int tmp_wp_id;
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
        grp_start_off = (grp_start_wp_id)*128;
        // tmp_wp_id = grp_start_wp_id;
        lhs_len = (128 );
        rhs_len = (128 );
        gran = (warp_lane<<2);
        if((warp_id&1)==0){
            gran += 0;
        }
        if((warp_id&1)==1){
            gran += (128 );
        }
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;

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
        __syncthreads();
        // Store merged results back to shared memory

        smem[grp_start_off+gran+0 ] = rg_k0 ;
        smem[grp_start_off+gran+1 ] = rg_k1 ;
        smem[grp_start_off+gran+2 ] = rg_k2 ;
        smem[grp_start_off+gran+3 ] = rg_k3 ;
        __syncthreads();
        // Step 1
        grp_start_wp_id = ((warp_id>>2)<<2);
        grp_start_off = (grp_start_wp_id)*128;
        // tmp_wp_id = grp_start_wp_id;
        lhs_len = (256 );
        rhs_len = (256 );
        gran = (warp_lane<<2);
        gran += (warp_id&3)*128;
        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;

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
        __syncthreads();
        // Store merged results back to shared memory

        smem[grp_start_off+gran+0 ] = rg_k0 ;
        smem[grp_start_off+gran+1 ] = rg_k1 ;
        smem[grp_start_off+gran+2 ] = rg_k2 ;
        smem[grp_start_off+gran+3 ] = rg_k3 ;
        __syncthreads();
        // Step 2
        grp_start_wp_id = ((warp_id>>3)<<3);
        grp_start_off = (grp_start_wp_id)*128;
        // tmp_wp_id = grp_start_wp_id;
        lhs_len = (512 );
        rhs_len = (512 );
        gran = (warp_lane<<2);
        gran += (warp_id&7)*128;

        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;
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
        __syncthreads();
        // Store merged results back to shared memory

        smem[grp_start_off+gran+0 ] = rg_k0 ;
        smem[grp_start_off+gran+1 ] = rg_k1 ;
        smem[grp_start_off+gran+2 ] = rg_k2 ;
        smem[grp_start_off+gran+3 ] = rg_k3 ;
        __syncthreads();
        // Step 3
        grp_start_wp_id = ((warp_id>>4)<<4);
        grp_start_off = (grp_start_wp_id)*128;
        // tmp_wp_id = grp_start_wp_id;
        lhs_len = (1024);
        rhs_len = (1024);
        gran = (warp_lane<<2);
        gran += (warp_id&15)*128;

        start = smem + grp_start_off;
        s_a = find_kth3(start, lhs_len, start+lhs_len, rhs_len, gran);
        s_b = lhs_len + gran - s_a;

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

        if((tid<<2)+0 <inner_seg_size) keyB[k+(tid<<2)+0 ] = rg_k0 ;
        if((tid<<2)+1 <inner_seg_size) keyB[k+(tid<<2)+1 ] = rg_k1 ;
        if((tid<<2)+2 <inner_seg_size) keyB[k+(tid<<2)+2 ] = rg_k2 ;
        if((tid<<2)+3 <inner_seg_size) keyB[k+(tid<<2)+3 ] = rg_k3 ;
    }
}

template<class K, class Offset>
__global__
void kern_block_merge(
    const K *keys, K *keysB,
    const Offset *seg_begins, const Offset *seg_ends, const int *bin,
    const int stride, const int workloads_per_block)
{
    const int bin_it = blockIdx.x;
    const int innerbid = blockIdx.y;

    const int seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
    const int blk_stat = (seg_size+workloads_per_block-1)/workloads_per_block;

    if(innerbid < blk_stat && stride < seg_size)
    {
        const int tid = threadIdx.x;
        __shared__ K smem[128*16];
        const int k = seg_begins[bin[bin_it]];

        int loc_a, loc_b;
        int cnt_a, cnt_b;
        int coop = (stride<<1)>>11; // how many blocks coop
        int coop_bid = innerbid%coop;
        int l_gran, r_gran;
        loc_a = (innerbid/coop)*(stride<<1);
        cnt_a = min(stride, seg_size-loc_a);
        loc_b = min(loc_a + stride, seg_size);
        cnt_b = min(stride, seg_size-loc_b);
        l_gran = coop_bid<<11;
        r_gran = min((coop_bid+1)<<11, seg_size-loc_a);
        int l_s_a, l_s_b;
        int r_s_a, r_s_b;
        l_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, l_gran);
        l_s_b = l_gran - l_s_a;
        r_s_a = find_kth3(keys+k+loc_a, cnt_a, keys+k+loc_b, cnt_b, r_gran);
        r_s_b = r_gran - r_s_a;
        int l_st = 0;
        int l_cnt = r_s_a - l_s_a;
        if(l_s_a+tid     <r_s_a) smem[l_st+tid     ] = keys[k+loc_a+l_s_a+tid     ];
        if(l_s_a+tid+128 <r_s_a) smem[l_st+tid+128 ] = keys[k+loc_a+l_s_a+tid+128 ];
        if(l_s_a+tid+256 <r_s_a) smem[l_st+tid+256 ] = keys[k+loc_a+l_s_a+tid+256 ];
        if(l_s_a+tid+384 <r_s_a) smem[l_st+tid+384 ] = keys[k+loc_a+l_s_a+tid+384 ];
        if(l_s_a+tid+512 <r_s_a) smem[l_st+tid+512 ] = keys[k+loc_a+l_s_a+tid+512 ];
        if(l_s_a+tid+640 <r_s_a) smem[l_st+tid+640 ] = keys[k+loc_a+l_s_a+tid+640 ];
        if(l_s_a+tid+768 <r_s_a) smem[l_st+tid+768 ] = keys[k+loc_a+l_s_a+tid+768 ];
        if(l_s_a+tid+896 <r_s_a) smem[l_st+tid+896 ] = keys[k+loc_a+l_s_a+tid+896 ];
        if(l_s_a+tid+1024<r_s_a) smem[l_st+tid+1024] = keys[k+loc_a+l_s_a+tid+1024];
        if(l_s_a+tid+1152<r_s_a) smem[l_st+tid+1152] = keys[k+loc_a+l_s_a+tid+1152];
        if(l_s_a+tid+1280<r_s_a) smem[l_st+tid+1280] = keys[k+loc_a+l_s_a+tid+1280];
        if(l_s_a+tid+1408<r_s_a) smem[l_st+tid+1408] = keys[k+loc_a+l_s_a+tid+1408];
        if(l_s_a+tid+1536<r_s_a) smem[l_st+tid+1536] = keys[k+loc_a+l_s_a+tid+1536];
        if(l_s_a+tid+1664<r_s_a) smem[l_st+tid+1664] = keys[k+loc_a+l_s_a+tid+1664];
        if(l_s_a+tid+1792<r_s_a) smem[l_st+tid+1792] = keys[k+loc_a+l_s_a+tid+1792];
        if(l_s_a+tid+1920<r_s_a) smem[l_st+tid+1920] = keys[k+loc_a+l_s_a+tid+1920];
        int r_st = r_s_a - l_s_a;
        int r_cnt = r_s_b - l_s_b;
        if(l_s_b+tid     <r_s_b) smem[r_st+tid     ] = keys[k+loc_b+l_s_b+tid     ];
        if(l_s_b+tid+128 <r_s_b) smem[r_st+tid+128 ] = keys[k+loc_b+l_s_b+tid+128 ];
        if(l_s_b+tid+256 <r_s_b) smem[r_st+tid+256 ] = keys[k+loc_b+l_s_b+tid+256 ];
        if(l_s_b+tid+384 <r_s_b) smem[r_st+tid+384 ] = keys[k+loc_b+l_s_b+tid+384 ];
        if(l_s_b+tid+512 <r_s_b) smem[r_st+tid+512 ] = keys[k+loc_b+l_s_b+tid+512 ];
        if(l_s_b+tid+640 <r_s_b) smem[r_st+tid+640 ] = keys[k+loc_b+l_s_b+tid+640 ];
        if(l_s_b+tid+768 <r_s_b) smem[r_st+tid+768 ] = keys[k+loc_b+l_s_b+tid+768 ];
        if(l_s_b+tid+896 <r_s_b) smem[r_st+tid+896 ] = keys[k+loc_b+l_s_b+tid+896 ];
        if(l_s_b+tid+1024<r_s_b) smem[r_st+tid+1024] = keys[k+loc_b+l_s_b+tid+1024];
        if(l_s_b+tid+1152<r_s_b) smem[r_st+tid+1152] = keys[k+loc_b+l_s_b+tid+1152];
        if(l_s_b+tid+1280<r_s_b) smem[r_st+tid+1280] = keys[k+loc_b+l_s_b+tid+1280];
        if(l_s_b+tid+1408<r_s_b) smem[r_st+tid+1408] = keys[k+loc_b+l_s_b+tid+1408];
        if(l_s_b+tid+1536<r_s_b) smem[r_st+tid+1536] = keys[k+loc_b+l_s_b+tid+1536];
        if(l_s_b+tid+1664<r_s_b) smem[r_st+tid+1664] = keys[k+loc_b+l_s_b+tid+1664];
        if(l_s_b+tid+1792<r_s_b) smem[r_st+tid+1792] = keys[k+loc_b+l_s_b+tid+1792];
        if(l_s_b+tid+1920<r_s_b) smem[r_st+tid+1920] = keys[k+loc_b+l_s_b+tid+1920];
        __syncthreads();

        int gran = tid<<4;
        int s_a, s_b;
        bool p;
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
        K tmp_k0,tmp_k1;

        s_a = find_kth3(smem+l_st, l_cnt, smem+r_st, r_cnt, gran);
        s_b = gran - s_a;
        tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k0 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k1 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k2 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k3 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k4 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k5 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k6 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k7 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k8 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k9 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k10 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k11 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k12 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k13 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k14 = p ? tmp_k0 : tmp_k1;
        if(p) {
            ++s_a;
            tmp_k0 = (s_a < l_cnt)? smem[l_st+s_a]:std::numeric_limits<K>::max();
        } else {
            ++s_b;
            tmp_k1 = (s_b < r_cnt)? smem[r_st+s_b]:std::numeric_limits<K>::max();
        }
        p = (s_b >= r_cnt) || ((s_a < l_cnt) && (tmp_k0 <= tmp_k1));
        rg_k15 = p ? tmp_k0 : tmp_k1;

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        rg_k1  = __shfl_xor_sync(0xffffffff,rg_k1 , 0x1 );
        rg_k3  = __shfl_xor_sync(0xffffffff,rg_k3 , 0x1 );
        rg_k5  = __shfl_xor_sync(0xffffffff,rg_k5 , 0x1 );
        rg_k7  = __shfl_xor_sync(0xffffffff,rg_k7 , 0x1 );
        rg_k9  = __shfl_xor_sync(0xffffffff,rg_k9 , 0x1 );
        rg_k11 = __shfl_xor_sync(0xffffffff,rg_k11, 0x1 );
        rg_k13 = __shfl_xor_sync(0xffffffff,rg_k13, 0x1 );
        rg_k15 = __shfl_xor_sync(0xffffffff,rg_k15, 0x1 );
        if(lane_id&0x1) SWP_KEY(K, rg_k0 ,rg_k1 );
        if(lane_id&0x1) SWP_KEY(K, rg_k2 ,rg_k3 );
        if(lane_id&0x1) SWP_KEY(K, rg_k4 ,rg_k5 );
        if(lane_id&0x1) SWP_KEY(K, rg_k6 ,rg_k7 );
        if(lane_id&0x1) SWP_KEY(K, rg_k8 ,rg_k9 );
        if(lane_id&0x1) SWP_KEY(K, rg_k10,rg_k11);
        if(lane_id&0x1) SWP_KEY(K, rg_k12,rg_k13);
        if(lane_id&0x1) SWP_KEY(K, rg_k14,rg_k15);
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

        if((innerbid<<11)+(warp_id<<9)+0  +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+0  +lane_id] = rg_k0 ;
        if((innerbid<<11)+(warp_id<<9)+32 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+32 +lane_id] = rg_k2 ;
        if((innerbid<<11)+(warp_id<<9)+64 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+64 +lane_id] = rg_k4 ;
        if((innerbid<<11)+(warp_id<<9)+96 +lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+96 +lane_id] = rg_k6 ;
        if((innerbid<<11)+(warp_id<<9)+128+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+128+lane_id] = rg_k8 ;
        if((innerbid<<11)+(warp_id<<9)+160+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+160+lane_id] = rg_k10;
        if((innerbid<<11)+(warp_id<<9)+192+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+192+lane_id] = rg_k12;
        if((innerbid<<11)+(warp_id<<9)+224+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+224+lane_id] = rg_k14;
        if((innerbid<<11)+(warp_id<<9)+256+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+256+lane_id] = rg_k1 ;
        if((innerbid<<11)+(warp_id<<9)+288+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+288+lane_id] = rg_k3 ;
        if((innerbid<<11)+(warp_id<<9)+320+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+320+lane_id] = rg_k5 ;
        if((innerbid<<11)+(warp_id<<9)+352+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+352+lane_id] = rg_k7 ;
        if((innerbid<<11)+(warp_id<<9)+384+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+384+lane_id] = rg_k9 ;
        if((innerbid<<11)+(warp_id<<9)+416+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+416+lane_id] = rg_k11;
        if((innerbid<<11)+(warp_id<<9)+448+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+448+lane_id] = rg_k13;
        if((innerbid<<11)+(warp_id<<9)+480+lane_id<seg_size) keysB[k+(innerbid<<11)+(warp_id<<9)+480+lane_id] = rg_k15;
    }
}

template<class K, class Offset>
__global__
void kern_copy(
    const K *srck, K *dstk,
    const Offset *seg_begins, const Offset *seg_ends, const int *bin,
    const int workloads_per_block)
{
    const int bin_it = blockIdx.x;
    const int innerbid = blockIdx.y;

    const int seg_size = seg_ends[bin[bin_it]]-seg_begins[bin[bin_it]];
    const int blk_stat = (seg_size+workloads_per_block-1)/workloads_per_block;

    if(innerbid < blk_stat)
    {
        const int tid = threadIdx.x;
        int k = seg_begins[bin[bin_it]];
        int stride = upper_power_of_two(seg_size);
        int steps = log2(stride/2048);

        if((steps&1))
        {
            if((innerbid<<11)+tid     <seg_size) dstk[k+(innerbid<<11)+tid     ] = srck[k+(innerbid<<11)+tid     ];
            if((innerbid<<11)+tid+128 <seg_size) dstk[k+(innerbid<<11)+tid+128 ] = srck[k+(innerbid<<11)+tid+128 ];
            if((innerbid<<11)+tid+256 <seg_size) dstk[k+(innerbid<<11)+tid+256 ] = srck[k+(innerbid<<11)+tid+256 ];
            if((innerbid<<11)+tid+384 <seg_size) dstk[k+(innerbid<<11)+tid+384 ] = srck[k+(innerbid<<11)+tid+384 ];
            if((innerbid<<11)+tid+512 <seg_size) dstk[k+(innerbid<<11)+tid+512 ] = srck[k+(innerbid<<11)+tid+512 ];
            if((innerbid<<11)+tid+640 <seg_size) dstk[k+(innerbid<<11)+tid+640 ] = srck[k+(innerbid<<11)+tid+640 ];
            if((innerbid<<11)+tid+768 <seg_size) dstk[k+(innerbid<<11)+tid+768 ] = srck[k+(innerbid<<11)+tid+768 ];
            if((innerbid<<11)+tid+896 <seg_size) dstk[k+(innerbid<<11)+tid+896 ] = srck[k+(innerbid<<11)+tid+896 ];
            if((innerbid<<11)+tid+1024<seg_size) dstk[k+(innerbid<<11)+tid+1024] = srck[k+(innerbid<<11)+tid+1024];
            if((innerbid<<11)+tid+1152<seg_size) dstk[k+(innerbid<<11)+tid+1152] = srck[k+(innerbid<<11)+tid+1152];
            if((innerbid<<11)+tid+1280<seg_size) dstk[k+(innerbid<<11)+tid+1280] = srck[k+(innerbid<<11)+tid+1280];
            if((innerbid<<11)+tid+1408<seg_size) dstk[k+(innerbid<<11)+tid+1408] = srck[k+(innerbid<<11)+tid+1408];
            if((innerbid<<11)+tid+1536<seg_size) dstk[k+(innerbid<<11)+tid+1536] = srck[k+(innerbid<<11)+tid+1536];
            if((innerbid<<11)+tid+1664<seg_size) dstk[k+(innerbid<<11)+tid+1664] = srck[k+(innerbid<<11)+tid+1664];
            if((innerbid<<11)+tid+1792<seg_size) dstk[k+(innerbid<<11)+tid+1792] = srck[k+(innerbid<<11)+tid+1792];
            if((innerbid<<11)+tid+1920<seg_size) dstk[k+(innerbid<<11)+tid+1920] = srck[k+(innerbid<<11)+tid+1920];
        }
    }
}

template<class K, class Offset>
void gen_grid_kern_r2049(
    K * keys_d, K * keysB_d,
    const Offset *seg_begins_d, const Offset *seg_ends_d,
    const int *bin_d, const int bin_size, const int max_segsize,
    cudaStream_t stream)
{
    const int workloads_per_block = 2048;

    dim3 block_per_grid(1, 1, 1);
    block_per_grid.x = bin_size;
    block_per_grid.y = (max_segsize+workloads_per_block-1)/workloads_per_block;

    int threads_per_block = 512;
    kern_block_sort<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys_d, keysB_d,
        seg_begins_d, seg_ends_d, bin_d,
        workloads_per_block);

    std::swap(keys_d, keysB_d);
    int cnt_swaps = 1;

    threads_per_block = 128;
    for(int stride = 2048; // unit for already sorted
        stride < max_segsize;
        stride <<= 1)
    {
        kern_block_merge<<<block_per_grid, threads_per_block, 0, stream>>>(
            keys_d, keysB_d,
            seg_begins_d, seg_ends_d, bin_d,
            stride, workloads_per_block);
        std::swap(keys_d, keysB_d);
        cnt_swaps++;
    }
    // std::cout << "cnt_swaps " << cnt_swaps << std::endl;

    if((cnt_swaps&1))
        std::swap(keys_d, keysB_d);

    threads_per_block = 128;
    kern_copy<<<block_per_grid, threads_per_block, 0, stream>>>(
        keys_d, keysB_d,
        seg_begins_d, seg_ends_d, bin_d,
        workloads_per_block);
}

#endif
