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

#ifndef _H_BB_EXCH_KEYS
#define _H_BB_EXCH_KEYS

#define CMP_SWP_KEY(t1,_a,_b) if(_a>_b)  {t1 _t=_a;_a=_b;_b=_t;}
#define EQL_SWP_KEY(t1,_a,_b) if(_a!=_b) {t1 _t=_a;_a=_b;_b=_t;}
#define     SWP_KEY(t1,_a,_b)            {t1 _t=_a;_a=_b;_b=_t;}
// Exchange intersection for 1 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, int mask, const int bit) {
    K ex_k0, ex_k1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k0, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
}
// Exchange intersection for 2 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, K &k1, int mask, const int bit) {
    K ex_k0, ex_k1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
}
// Exchange intersection for 4 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, K &k1, K &k2, K &k3, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k2);
    if(bit) SWP_KEY(K, k1, k3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k2);
    if(bit) SWP_KEY(K, k1, k3);
}
// Exchange intersection for 8 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k6);
    if(bit) SWP_KEY(K, k1, k7);
    if(bit) SWP_KEY(K, k2, k4);
    if(bit) SWP_KEY(K, k3, k5);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k6);
    if(bit) SWP_KEY(K, k1, k7);
    if(bit) SWP_KEY(K, k2, k4);
    if(bit) SWP_KEY(K, k3, k5);
}
// Exchange intersection for 16 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k14);
    if(bit) SWP_KEY(K, k1, k15);
    if(bit) SWP_KEY(K, k2, k12);
    if(bit) SWP_KEY(K, k3, k13);
    if(bit) SWP_KEY(K, k4, k10);
    if(bit) SWP_KEY(K, k5, k11);
    if(bit) SWP_KEY(K, k6, k8);
    if(bit) SWP_KEY(K, k7, k9);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff,k9, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff,k11, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff,k13, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff,k15, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k14);
    if(bit) SWP_KEY(K, k1, k15);
    if(bit) SWP_KEY(K, k2, k12);
    if(bit) SWP_KEY(K, k3, k13);
    if(bit) SWP_KEY(K, k4, k10);
    if(bit) SWP_KEY(K, k5, k11);
    if(bit) SWP_KEY(K, k6, k8);
    if(bit) SWP_KEY(K, k7, k9);
}
// Exchange intersection for 32 keys.
template<class K>
__device__ inline void exch_intxn_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, K &k16, K &k17, K &k18, K &k19, K &k20, K &k21, K &k22, K &k23, K &k24, K &k25, K &k26, K &k27, K &k28, K &k29, K &k30, K &k31, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k30);
    if(bit) SWP_KEY(K, k1, k31);
    if(bit) SWP_KEY(K, k2, k28);
    if(bit) SWP_KEY(K, k3, k29);
    if(bit) SWP_KEY(K, k4, k26);
    if(bit) SWP_KEY(K, k5, k27);
    if(bit) SWP_KEY(K, k6, k24);
    if(bit) SWP_KEY(K, k7, k25);
    if(bit) SWP_KEY(K, k8, k22);
    if(bit) SWP_KEY(K, k9, k23);
    if(bit) SWP_KEY(K, k10, k20);
    if(bit) SWP_KEY(K, k11, k21);
    if(bit) SWP_KEY(K, k12, k18);
    if(bit) SWP_KEY(K, k13, k19);
    if(bit) SWP_KEY(K, k14, k16);
    if(bit) SWP_KEY(K, k15, k17);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff,k9, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff,k11, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff,k13, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff,k15, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor_sync(0xffffffff,k17, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k16 = ex_k0;
    k17 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor_sync(0xffffffff,k19, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k18 = ex_k0;
    k19 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor_sync(0xffffffff,k21, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k20 = ex_k0;
    k21 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor_sync(0xffffffff,k23, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k22 = ex_k0;
    k23 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor_sync(0xffffffff,k25, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k24 = ex_k0;
    k25 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor_sync(0xffffffff,k27, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k26 = ex_k0;
    k27 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor_sync(0xffffffff,k29, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k28 = ex_k0;
    k29 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor_sync(0xffffffff,k31, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k30 = ex_k0;
    k31 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k30);
    if(bit) SWP_KEY(K, k1, k31);
    if(bit) SWP_KEY(K, k2, k28);
    if(bit) SWP_KEY(K, k3, k29);
    if(bit) SWP_KEY(K, k4, k26);
    if(bit) SWP_KEY(K, k5, k27);
    if(bit) SWP_KEY(K, k6, k24);
    if(bit) SWP_KEY(K, k7, k25);
    if(bit) SWP_KEY(K, k8, k22);
    if(bit) SWP_KEY(K, k9, k23);
    if(bit) SWP_KEY(K, k10, k20);
    if(bit) SWP_KEY(K, k11, k21);
    if(bit) SWP_KEY(K, k12, k18);
    if(bit) SWP_KEY(K, k13, k19);
    if(bit) SWP_KEY(K, k14, k16);
    if(bit) SWP_KEY(K, k15, k17);
}
// Exchange parallel for 1 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, int mask, const int bit) {
    K ex_k0, ex_k1;
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k0, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
}
// Exchange parallel for 2 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, K &k1, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k1);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k1);
}
// Exchange parallel for 4 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, K &k1, K &k2, K &k3, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
}
// Exchange parallel for 8 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
}
// Exchange parallel for 16 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
    if(bit) SWP_KEY(K, k8, k9);
    if(bit) SWP_KEY(K, k10, k11);
    if(bit) SWP_KEY(K, k12, k13);
    if(bit) SWP_KEY(K, k14, k15);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff,k9, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff,k11, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff,k13, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff,k15, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
    if(bit) SWP_KEY(K, k8, k9);
    if(bit) SWP_KEY(K, k10, k11);
    if(bit) SWP_KEY(K, k12, k13);
    if(bit) SWP_KEY(K, k14, k15);
}
// Exchange parallel for 32 keys.
template<class K>
__device__ inline void exch_paral_keys(K &k0, K &k1, K &k2, K &k3, K &k4, K &k5, K &k6, K &k7, K &k8, K &k9, K &k10, K &k11, K &k12, K &k13, K &k14, K &k15, K &k16, K &k17, K &k18, K &k19, K &k20, K &k21, K &k22, K &k23, K &k24, K &k25, K &k26, K &k27, K &k28, K &k29, K &k30, K &k31, int mask, const int bit) {
    K ex_k0, ex_k1;
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
    if(bit) SWP_KEY(K, k8, k9);
    if(bit) SWP_KEY(K, k10, k11);
    if(bit) SWP_KEY(K, k12, k13);
    if(bit) SWP_KEY(K, k14, k15);
    if(bit) SWP_KEY(K, k16, k17);
    if(bit) SWP_KEY(K, k18, k19);
    if(bit) SWP_KEY(K, k20, k21);
    if(bit) SWP_KEY(K, k22, k23);
    if(bit) SWP_KEY(K, k24, k25);
    if(bit) SWP_KEY(K, k26, k27);
    if(bit) SWP_KEY(K, k28, k29);
    if(bit) SWP_KEY(K, k30, k31);
    ex_k0 = k0;
    ex_k1 = __shfl_xor_sync(0xffffffff,k1, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k0 = ex_k0;
    k1 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k2;
    ex_k1 = __shfl_xor_sync(0xffffffff,k3, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k2 = ex_k0;
    k3 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k4;
    ex_k1 = __shfl_xor_sync(0xffffffff,k5, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k4 = ex_k0;
    k5 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k6;
    ex_k1 = __shfl_xor_sync(0xffffffff,k7, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k6 = ex_k0;
    k7 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k8;
    ex_k1 = __shfl_xor_sync(0xffffffff,k9, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k8 = ex_k0;
    k9 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k10;
    ex_k1 = __shfl_xor_sync(0xffffffff,k11, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k10 = ex_k0;
    k11 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k12;
    ex_k1 = __shfl_xor_sync(0xffffffff,k13, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k12 = ex_k0;
    k13 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k14;
    ex_k1 = __shfl_xor_sync(0xffffffff,k15, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k14 = ex_k0;
    k15 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k16;
    ex_k1 = __shfl_xor_sync(0xffffffff,k17, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k16 = ex_k0;
    k17 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k18;
    ex_k1 = __shfl_xor_sync(0xffffffff,k19, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k18 = ex_k0;
    k19 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k20;
    ex_k1 = __shfl_xor_sync(0xffffffff,k21, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k20 = ex_k0;
    k21 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k22;
    ex_k1 = __shfl_xor_sync(0xffffffff,k23, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k22 = ex_k0;
    k23 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k24;
    ex_k1 = __shfl_xor_sync(0xffffffff,k25, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k24 = ex_k0;
    k25 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k26;
    ex_k1 = __shfl_xor_sync(0xffffffff,k27, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k26 = ex_k0;
    k27 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k28;
    ex_k1 = __shfl_xor_sync(0xffffffff,k29, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k28 = ex_k0;
    k29 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    ex_k0 = k30;
    ex_k1 = __shfl_xor_sync(0xffffffff,k31, mask);
    CMP_SWP_KEY(K, ex_k0, ex_k1);
    if(bit) EQL_SWP_KEY(K, ex_k0, ex_k1);
    k30 = ex_k0;
    k31 = __shfl_xor_sync(0xffffffff,ex_k1, mask);
    if(bit) SWP_KEY(K, k0, k1);
    if(bit) SWP_KEY(K, k2, k3);
    if(bit) SWP_KEY(K, k4, k5);
    if(bit) SWP_KEY(K, k6, k7);
    if(bit) SWP_KEY(K, k8, k9);
    if(bit) SWP_KEY(K, k10, k11);
    if(bit) SWP_KEY(K, k12, k13);
    if(bit) SWP_KEY(K, k14, k15);
    if(bit) SWP_KEY(K, k16, k17);
    if(bit) SWP_KEY(K, k18, k19);
    if(bit) SWP_KEY(K, k20, k21);
    if(bit) SWP_KEY(K, k22, k23);
    if(bit) SWP_KEY(K, k24, k25);
    if(bit) SWP_KEY(K, k26, k27);
    if(bit) SWP_KEY(K, k28, k29);
    if(bit) SWP_KEY(K, k30, k31);
}

#endif
