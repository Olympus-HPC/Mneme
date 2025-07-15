#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

  void
bucketcount (const float* input , 
            int* indice,
            unsigned int* prefixoffsets,
            const float* pivotpoints,
            const int listsize,
            sycl::nd_item<3> item_ct1,
            unsigned int *s_offset)
{

  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  const int lid = item_ct1.get_local_id(2);
  const int gsize =
      item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2);
  const int lsize = item_ct1.get_local_range().get(2);
  const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
  const int numThreads = gsize;

  for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
    s_offset[i] = 0;

  item_ct1.barrier();

  for (int tid = gid; tid < listsize; tid += numThreads) {
    float elem = input[tid];

    int idx  = DIVISIONS/2 - 1;
    int jump = DIVISIONS/4;
    float piv = pivotpoints[idx]; //s_pivotpoints[idx];

    while(jump >= 1){
      idx = (elem < piv) ? (idx - jump) : (idx + jump);
      piv = pivotpoints[idx]; //s_pivotpoints[idx];
      jump /= 2;
    }
    idx = (elem < piv) ? idx : (idx + 1);

    indice[tid] =
        (sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
             sycl::local_ptr<unsigned int>(&s_offset[warpBase + idx]))
             .fetch_add(1U)
         << LOG_DIVISIONS) +
        idx;
  }

  item_ct1.barrier();

  int prefixBase = item_ct1.get_group(2) * BUCKET_BLOCK_MEMORY;

  for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize)
    prefixoffsets[prefixBase + i] = s_offset[i] & 0x07FFFFFFU;

}

