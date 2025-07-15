#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute per-workgroup partial histograms
////////////////////////////////////////////////////////////////////////////////
  void
histogram1024 ( unsigned int* histoOutput, 
		const float* histoInput,
    const int listsize,
    const float minimum,
    const float maximum,
    sycl::nd_item<3> item_ct1,
    unsigned int *s_Hist)
{

  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  const int lid = item_ct1.get_local_id(2);
  const int gsize =
      item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2);
  const int lsize = item_ct1.get_local_range().get(2);

  //Per-warp substorage storage
  int mulBase = (lid >> BUCKET_WARP_LOG_SIZE);
  const int warpBase = IMUL(mulBase, HISTOGRAM_BIN_COUNT);

  //Clear shared memory storage for current threadblock before processing
  for(uint i = lid; i < HISTOGRAM_BLOCK_MEMORY; i+=lsize) {
    s_Hist[i] = 0;
  }


  //Read through the entire input buffer, build per-warp histograms
  item_ct1.barrier();
  for(int pos = gid; pos < listsize; pos += gsize) {
    uint data4 = ((histoInput[pos] - minimum)/(maximum - minimum)) * HISTOGRAM_BIN_COUNT;

    sycl::atomic<unsigned int, sycl::access::address_space::local_space>(
        sycl::local_ptr<unsigned int>(&s_Hist[warpBase + (data4 & 0x3FFU)]))
        .fetch_add(1U);
  }

  //Per-block histogram reduction
  item_ct1.barrier();

  for(int pos = lid; pos < HISTOGRAM_BIN_COUNT; pos += lsize){
    uint sum = 0;
    for(int i = 0; i < HISTOGRAM_BLOCK_MEMORY; i+= HISTOGRAM_BIN_COUNT){ 
      sum += s_Hist[pos + i] & 0x07FFFFFFU;
    }
    sycl::atomic<unsigned int>(
        sycl::global_ptr<unsigned int>(&histoOutput[pos]))
        .fetch_add(sum);
  }
}

