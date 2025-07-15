#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void bucketsort(const float *input, const int *indice, float *output,
                const unsigned int *prefixoffsets, const unsigned int *offsets,
                const int listsize, sycl::nd_item<3> item_ct1,
                unsigned int *s_offset)
{
  const int grp_id = item_ct1.get_group(2);
  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  const int lid = item_ct1.get_local_id(2);
  const int gsize =
      item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2);
  const int lsize = item_ct1.get_local_range().get(2);

        int prefixBase = grp_id * BUCKET_BLOCK_MEMORY;
    const int warpBase = (lid >> BUCKET_WARP_LOG_SIZE) * DIVISIONS;
    const int numThreads = gsize;
    
	for (int i = lid; i < BUCKET_BLOCK_MEMORY; i += lsize){
		s_offset[i] = offsets[i & (DIVISIONS - 1)] + prefixoffsets[prefixBase + i];
	}

  item_ct1.barrier();

        for (int tid = gid; tid < listsize; tid += numThreads){
    float elem = input[tid];
    int id = indice[tid];
    output[s_offset[warpBase + (id & (DIVISIONS - 1))] + (id >>  LOG_DIVISIONS)] = elem;
  }
}

