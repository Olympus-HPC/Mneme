/*
Copyright (c) 2014-2019, Intel Corporation
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:
* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
      * Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef QUICKSORT_H
#define QUICKSORT_H

template <class T>
__host__
T median_host(T x1, T x2, T x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
			if (x1 < x3) {
				return x3;
			} else {
				return x1;
			}
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
			if (x2 < x3) {
				return x2;
			} else {
				return x3;
			}
		}
	}
}

template <typename T, typename P>
__device__
T select (T a, T b, P c) {
  return c ? b : a;
}

__device__
uint median(uint x1, uint x2, uint x3) {
	if (x1 < x2) {
		if (x2 < x3) {
			return x2;
		} else {
      return select(x1, x3, x1 < x3);
		}
	} else { // x1 >= x2
		if (x1 < x3) {
			return x1;
		} else { // x1 >= x3
      return select(x2, x3, x2 < x3);
		}
	}
}

#define TRUST_BUT_VERIFY 1
// Note that SORT_THRESHOLD should always be 2X LQSORT_LOCAL_WORKGROUP_SIZE due to the use of bitonic sort
// Always try LQSORT_LOCAL_WORKGROUP_SIZE to be 8X smaller than QUICKSORT_BLOCK_SIZE - then try everything else :)
#ifdef CPU_DEVICE
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#else
#ifdef NVIDIA_GPU
#define QUICKSORT_BLOCK_SIZE         1024 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128
#define LQSORT_LOCAL_WORKGROUP_SIZE   256 
#define SORT_THRESHOLD                512 
#else 
#define QUICKSORT_BLOCK_SIZE         1728 
#define GQSORT_LOCAL_WORKGROUP_SIZE   128 
#define LQSORT_LOCAL_WORKGROUP_SIZE   128 
#define SORT_THRESHOLD                256 
#endif 
#endif

#define EMPTY_RECORD             42

// work record contains info about the part of array that is still longer than QUICKSORT_BLOCK_SIZE and 
// therefore cannot be processed by lqsort_kernel yet. It contins the start and the end indexes into 
// an array to be sorted, associated pivot and direction of the sort. 
template <class T>
struct work_record {
	uint start;
	uint end;
	T pivot;
	uint direction;
__host__ 
	work_record() : 
		start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD) {}
__host__ __device__
	work_record(uint s, uint e, uint p, uint d) : 
		start(s), end(e), pivot(p), direction(d) {}
};


// parent record contains everything kernels need to know about the parent of a set of blocks:
// initially, the first two fields equal to the third and fourth fields respectively
// blockcount contains the total number of blocks associated with the parent.
// During processing, sstart and send get incremented. At the end of gqsort_kernel, all the 
// parent record fields are used to calculate new pivots and new work records.
typedef struct parent_record {
	uint sstart, send, oldstart, oldend, blockcount; 
__host__
	parent_record(uint ss, uint se, uint os, uint oe, uint bc) : 
		sstart(ss), send(se), oldstart(os), oldend(oe), blockcount(bc) {}
} parent_record;

// block record contains everything kernels needs to know about the block:
// start and end indexes into input array, pivot, direction of sorting and the parent record index
template <class T>
struct block_record {
	uint start;
	uint end;
	T pivot;
	uint direction;
	uint parent;
__host__
	block_record() : start(0), end(0), pivot(T(0)), direction(EMPTY_RECORD), parent(0) {}
__host__
	block_record(uint s, uint e, T p, uint d, uint prnt) : 
		start(s), end(e), pivot(p), direction(d), parent(prnt) {}
};
#endif // QUICKSORT_H
