#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
void bucketprefix(unsigned int *prefixoffsets, unsigned int *offsets,
                  int blocks, sycl::nd_item<3> item_ct1)
{

  const int tid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  const int size = blocks * BUCKET_BLOCK_MEMORY;
  int sum = 0;

  for (int i = tid; i < size; i += DIVISIONS) {
    int x = prefixoffsets[i];
    prefixoffsets[i] = sum;
    sum += x;
  }
  offsets[tid] = sum;
}

