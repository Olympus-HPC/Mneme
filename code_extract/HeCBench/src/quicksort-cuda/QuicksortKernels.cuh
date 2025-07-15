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

#include "Quicksort.h"



template <class T>
__device__
void plus_prescan( T *a,  T *b) {
  T av = *a;
  T bv = *b;
  *a = bv;
  *b = bv + av;
}

/// bitonic_sort: sort 2*LOCAL_THREADCOUNT elements
template <class T>
  __device__
void bitonic_sort( T* sh_data, const uint localid)
{
  for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
    for (uint j = ulevel; j > 0; j >>= 1) {
      uint pos = 2*localid - (localid & (j - 1));

      uint direction = localid & ulevel;
      uint av = sh_data[pos], bv = sh_data[pos + j];
      const bool sortThem = av > bv;
      const uint greater = select(bv, av, sortThem);
      const uint lesser  = select(av, bv, sortThem);

      sh_data[pos]     = select(lesser, greater, direction);
      sh_data[pos + j] = select(greater, lesser, direction);
      __syncthreads();
    }
  }

  for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
    uint pos = 2*localid - (localid & (j - 1));

    uint av = sh_data[pos], bv = sh_data[pos + j];
    const bool sortThem = av > bv;
    sh_data[pos]      = select(av, bv, sortThem);
    sh_data[pos + j]  = select(bv, av, sortThem);

    __syncthreads();
  }
}

template <typename T>
  __device__
void sort_threshold( T* data_in, 
    T* data_out,
    uint start, 
    uint end,  
    T* temp, 
    uint localid) 
{
  uint tsum = end - start;
  if (tsum == SORT_THRESHOLD) {
    bitonic_sort(data_in+start, localid);
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      data_out[start + i] = data_in[start + i];
    }
  } else if (tsum > 1) {
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      if (i < tsum) {
        temp[i] = data_in[start + i];
      } else {
        temp[i] = UINT_MAX;
      }
    }
    __syncthreads();
    bitonic_sort(temp, localid);

    for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      data_out[start + i] = temp[i];
    }
  } else if (tsum == 1 && localid == 0) {
    data_out[start] = data_in[start];
  } 
}

//----------------------------------------------------------------------------
// Kernel implements gqsort_kernel
//----------------------------------------------------------------------------

template <class T>
  __global__ void
gqsort_kernel( T* d,  T* dn,  block_record<T>* blocks,  parent_record* parents,  work_record<T>* result) 
{
  const uint blockid    = blockIdx.x;
  const uint localid    = threadIdx.x;
  __shared__ uint lt[GQSORT_LOCAL_WORKGROUP_SIZE+1], gt[GQSORT_LOCAL_WORKGROUP_SIZE+1], ltsum, gtsum, lbeg, gbeg;
  uint i, lfrom, gfrom, lpivot, gpivot, tmp, ltp = 0, gtp = 0;

  // Get the sequence block assigned to this work group
  block_record<T> block = blocks[blockid];
  uint start = block.start, end = block.end, direction = block.direction;
  T pivot = block.pivot; 

  parent_record* pparent = parents + block.parent; 
  uint* psstart, *psend, *poldstart, *poldend, *pblockcount;
  T *s, *sn;

  // GPU-Quicksort cannot sort in place, as the regular quicksort algorithm can.
  // It therefore needs two arrays to sort things out. We start sorting in the 
  // direction of d -> dn and then change direction after each run of gqsort_kernel.
  // Which direction we are sorting: d -> dn or dn -> d?
  if (direction == 1) {
    s = d;
    sn = dn;
  } else {
    s = dn;
    sn = d;
  }

  // Set thread __shared__ counters to zero
  lt[localid] = gt[localid] = 0;
  __syncthreads();

  // Align thread accesses for coalesced reads.
  // Go through data...
  for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
    tmp = s[i];
    // counting elements that are smaller ...
    if (tmp < pivot)
      ltp++;
    // or larger compared to the pivot.
    if (tmp > pivot) 
      gtp++;
  }
  lt[localid] = ltp;
  gt[localid] = gtp;
  __syncthreads();

  // calculate cumulative sums
  uint n;
  for(i = 1; i < GQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
    n = 2*i - 1;
    if ((localid & n) == n) {
      lt[localid] += lt[localid-i];
      gt[localid] += gt[localid-i];
    }
    __syncthreads();
  }

  if ((localid & n) == n) {
    lt[GQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
    gt[GQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
    lt[localid] = 0;
    gt[localid] = 0;
  }

  for(i = GQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
    n = 2*i - 1;
    if ((localid & n) == n) {
      plus_prescan(&lt[localid - i], &lt[localid]);
      plus_prescan(&gt[localid - i], &gt[localid]);
    }
    __syncthreads();
  }

  // Allocate memory in the sequence this block is a part of
  if (localid == 0) {
    // get shared variables
    psstart = &pparent->sstart;
    psend = &pparent->send;
    poldstart = &pparent->oldstart;
    poldend = &pparent->oldend;
    pblockcount = &pparent->blockcount;
    // Atomic increment allocates memory to write to.
    lbeg = atomicAdd(psstart, ltsum);
    // Atomic is necessary since multiple blocks access this
    gbeg = atomicSub(psend, gtsum) - gtsum;
  }
  __syncthreads();

  // Allocate locations for work items
  lfrom = lbeg + lt[localid];
  gfrom = gbeg + gt[localid];

  // go thru data again writing elements to their correct position
  for(i = start + localid; i < end; i += GQSORT_LOCAL_WORKGROUP_SIZE) {
    tmp = s[i];
    // increment counts
    if (tmp < pivot) 
      sn[lfrom++] = tmp;

    if (tmp > pivot) 
      sn[gfrom++] = tmp;
  }
  __syncthreads();

  if (localid == 0) {
    //if (atomic_dec(pblockcount) == 0) {
    if (atomicSub(pblockcount, 1) == 0) {
      uint sstart = *psstart;
      uint send = *psend;
      uint oldstart = *poldstart;
      uint oldend = *poldend;

      // Store the pivot value between the new sequences
      for(i = sstart; i < send; i ++) {
        d[i] = pivot;
      }

      lpivot = sn[oldstart];
      gpivot = sn[oldend-1];
      if (oldstart < sstart) {
        lpivot = median(lpivot,sn[(oldstart+sstart) >> 1], sn[sstart-1]);
      } 
      if (send < oldend) {
        gpivot = median(sn[send],sn[(oldend+send) >> 1], gpivot);
      }

      work_record<T>* result1 = result + 2*blockid;
      work_record<T>* result2 = result1 + 1;

      // change the direction of the sort.
      direction ^= 1;

      work_record<T> r1 = {oldstart, sstart, lpivot, direction};
      *result1 = r1;

      work_record<T> r2 = {send, oldend, gpivot, direction};
      *result2 = r2;
    }
  }
  }

  // record to push start of the sequence, end of the sequence and direction of sorting on internal stack
  typedef struct workstack_record {
    uint start;
    uint end;
    uint direction;
  } workstack_record;

#define PUSH(START, END)       if (localid == 0) { \
  ++workstack_pointer; \
  workstack_record wr = { (START), (END), direction ^ 1 }; \
  workstack[workstack_pointer] = wr; \
} \
__syncthreads();


//---------------------------------------------------------------------------------------
// Kernel implements the last stage of GPU-Quicksort, when all the subsequences are small
// enough to be processed in __shared__ memory. It uses similar algorithm to gqsort_kernel to 
// move items around the pivot and then switches to bitonic sort for sequences in
// the range [1, SORT_THRESHOLD] 
//
// d - input array
// dn - scratch array of the same size as the input array
// seqs - array of records to be sorted in a __shared__ memory, one sequence per work group.
//---------------------------------------------------------------------------------------

template <class T>
  __global__ void 
lqsort_kernel(T* d, T* dn, work_record<T>* seqs) 
{
  const uint blockid    = blockIdx.x;
  const uint localid    = threadIdx.x;

  // workstack: stores the start and end of the sequences, direction of sort
  // If the sequence is less that SORT_THRESHOLD, it gets sorted. 
  // It will only be pushed on the stack if it greater than the SORT_THRESHOLD. 
  // Note, that the sum of ltsum + gtsum is less than QUICKSORT_BLOCK_SIZE. 
  // The total sum of the length of records on the stack cannot exceed QUICKSORT_BLOCK_SIZE, 
  // but each individual record should be greater than SORT_THRESHOLD, so the maximum length 
  // of the stack is QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD - in the case of BDW GT2 the length 
  // of the stack is 2 :)
  __shared__ workstack_record workstack[QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD]; 
  __shared__ int workstack_pointer;

  __shared__ T mys[QUICKSORT_BLOCK_SIZE], mysn[QUICKSORT_BLOCK_SIZE], temp[SORT_THRESHOLD];
  __shared__ T *s, *sn;
  __shared__ uint ltsum, gtsum;
  __shared__ uint lt[LQSORT_LOCAL_WORKGROUP_SIZE+1], gt[LQSORT_LOCAL_WORKGROUP_SIZE+1];
  uint i, tmp, ltp, gtp;

  work_record<T> block = seqs[blockid];
  const uint d_offset = block.start;
  uint start = 0; 
  uint end   = block.end - d_offset;

  uint direction = 1; // which direction to sort
  // initialize workstack and workstack_pointer: push the initial sequence on the stack
  if (localid == 0) {
    workstack_pointer = 0; // beginning of the stack
    workstack_record wr = { start, end, direction };
    workstack[0] = wr;
  }
  // copy block of data to be sorted by one workgroup into __shared__ memory
  // note that indeces of __shared__ data go from 0 to end-start-1
  if (block.direction == 1) {
    for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      mys[i] = d[i+d_offset];
    }
  } else {
    for (i = localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      mys[i] = dn[i+d_offset];
    }
  }
  __syncthreads();

  while (workstack_pointer >= 0) { 
    // pop up the stack
    workstack_record wr = workstack[workstack_pointer];
    start = wr.start;
    end = wr.end;
    direction = wr.direction;
    __syncthreads();
    if (localid == 0) {
      --workstack_pointer;

      ltsum = gtsum = 0;  
    }
    if (direction == 1) {
      s = mys;
      sn = mysn;
    } else {
      s = mysn;
      sn = mys;
    }
    // Set thread __shared__ counters to zero
    lt[localid] = gt[localid] = 0;
    ltp = gtp = 0;
    __syncthreads();

    // Pick a pivot
    uint pivot = s[start];
    if (start < end) {
      pivot = median(pivot, s[(start+end) >> 1], s[end-1]);
    }
    // Align work item accesses for coalesced reads.
    // Go through data...
    for(i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      tmp = s[i];
      // counting elements that are smaller ...
      if (tmp < pivot)
        ltp++;
      // or larger compared to the pivot.
      if (tmp > pivot) 
        gtp++;
    }
    lt[localid] = ltp;
    gt[localid] = gtp;
    __syncthreads();

    // calculate cumulative sums
    uint n;
    for(i = 1; i < LQSORT_LOCAL_WORKGROUP_SIZE; i <<= 1) {
      n = 2*i - 1;
      if ((localid & n) == n) {
        lt[localid] += lt[localid-i];
        gt[localid] += gt[localid-i];
      }
      __syncthreads();
    }

    if ((localid & n) == n) {
      lt[LQSORT_LOCAL_WORKGROUP_SIZE] = ltsum = lt[localid];
      gt[LQSORT_LOCAL_WORKGROUP_SIZE] = gtsum = gt[localid];
      lt[localid] = 0;
      gt[localid] = 0;
    }

    for(i = LQSORT_LOCAL_WORKGROUP_SIZE/2; i >= 1; i >>= 1) {
      n = 2*i - 1;
      if ((localid & n) == n) {
        plus_prescan(&lt[localid - i], &lt[localid]);
        plus_prescan(&gt[localid - i], &gt[localid]);
      }
      __syncthreads();
    }

    // Allocate locations for work items
    uint lfrom = start + lt[localid];
    uint gfrom = end - gt[localid+1];

    // go thru data again writing elements to their correct position
    for (i = start + localid; i < end; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      tmp = s[i];
      // increment counts
      if (tmp < pivot) 
        sn[lfrom++] = tmp;

      if (tmp > pivot) 
        sn[gfrom++] = tmp;
    }
    __syncthreads();

    // Store the pivot value between the new sequences
    for (i = start + ltsum + localid;i < end - gtsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      d[i+d_offset] = pivot;
    }
    __syncthreads();

    // if the sequence is shorter than SORT_THRESHOLD
    // sort it using an alternative sort and place result in d
    if (ltsum <= SORT_THRESHOLD) {
      sort_threshold(sn, d+d_offset, start, start + ltsum, temp, localid);
    } else {
      PUSH(start, start + ltsum);
    }

    if (gtsum <= SORT_THRESHOLD) {
      sort_threshold(sn, d+d_offset, end - gtsum, end, temp, localid);
    } else {
      PUSH(end - gtsum, end);
    }
  }
}
