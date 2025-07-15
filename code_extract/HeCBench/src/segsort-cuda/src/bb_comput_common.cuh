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

#ifndef _H_BB_COMPUT_COMMON
#define _H_BB_COMPUT_COMMON

__device__ inline
int upper_power_of_two(int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;

}

__device__ inline
int log2(int u)
{
    int s, t;
    t = (u > 0xffff) << 4; u >>= t;
    s = (u > 0xff  ) << 3; u >>= s, t |= s;
    s = (u > 0xf   ) << 2; u >>= s, t |= s;
    s = (u > 0x3   ) << 1; u >>= s, t |= s;
    return (t | (u >> 1));
}


template<class K>
__device__
int find_kth3(K* a, int aCount, K* b, int bCount, int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);

    while(begin < end) {
        int mid = (begin + end)>> 1;
        K aKey = a[mid];
        K bKey = b[diag - 1 - mid];
        bool pred = aKey <= bKey;
        if(pred) begin = mid + 1;
        else end = mid;
    }
    return begin;
}

#endif
