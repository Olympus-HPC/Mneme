#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
#include "bucketsort.h"

#include "kernel_bucketcount.h"
#include "kernel_bucketprefix.h"
#include "kernel_bucketsort.h"
#include "kernel_histogram.h"

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int histosize = 1024;
  //  ////////////////////////////////////////////////////////////////////////////
  //  // First pass - Create 1024 bin histogram
  //  ////////////////////////////////////////////////////////////////////////////
  unsigned int* h_offsets = (unsigned int *) malloc(DIVISIONS * sizeof(unsigned int));
  for(int i = 0; i < DIVISIONS; i++){
    h_offsets[i] = 0;
  }
  float* pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));
#ifdef DEBUG
  int* d_indice = (int *)malloc(listsize * sizeof(int));
  float* l_pivotpoints = (float *)malloc(DIVISIONS*sizeof(float));
#endif
  float* historesult = (float *)malloc(histosize * sizeof(float));

  int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
  //unsigned int* d_prefixoffsets = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));


  float *d_input_buff;
  unsigned int* d_offsets_buff;
  d_input_buff = sycl::malloc_device<float>((listsize + DIVISIONS * 4), q_ct1);
  q_ct1.memcpy(d_input_buff, d_input,
               sizeof(float) * (listsize + DIVISIONS * 4));
  d_offsets_buff = sycl::malloc_device<unsigned int>(DIVISIONS, q_ct1);

  size_t global_histogram = 6144;

#ifdef HISTO_WG_SIZE_0
  size_t local_histogram = HISTO_WG_SIZE_0;
#else
  size_t local_histogram = 96;
#endif

  q_ct1.memcpy(d_offsets_buff, h_offsets, sizeof(unsigned int) * DIVISIONS);
               
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        s_Hist_acc_ct1(sycl::range<1>(3072 /*HISTOGRAM_BLOCK_MEMORY*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, global_histogram / local_histogram) *
                sycl::range<3>(1, 1, local_histogram),
            sycl::range<3>(1, 1, local_histogram)),
        [=](sycl::nd_item<3> item_ct1) {
          histogram1024(d_offsets_buff, d_input_buff, listsize, minimum,
                        maximum, item_ct1, s_Hist_acc_ct1.get_pointer());
        });
  });
  q_ct1.memcpy(h_offsets, d_offsets_buff, sizeof(unsigned int) * histosize)
      .wait();

#ifdef DEBUG
  printf("h_offsets after histogram\n");
  for(int i=0; i<histosize; i++) {
    printf("%d %d\n", i, h_offsets[i]);
  }
#endif
  for(int i=0; i<histosize; i++) {
    historesult[i] = (float)h_offsets[i];
  }


  //  ///////////////////////////////////////////////////////////////////////////
  //  // Calculate pivot points (CPU algorithm)
  //  ///////////////////////////////////////////////////////////////////////////
  calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
      minimum, maximum, pivotPoints,
      (maximum - minimum)/(float)histosize);
  //
  //  ///////////////////////////////////////////////////////////////////////////
  //  // Count the bucket sizes in new divisions
  //  ///////////////////////////////////////////////////////////////////////////


  //buffer<float,1> l_pivotpoints_buff(pivotPoints, DIVISIONS, props);
  //buffer<int,1> d_indice_buff(listsize);
  //buffer<unsigned int,1> d_prefixoffsets_buff(blocks * BUCKET_BLOCK_MEMORY);

  float* d_pivotPoints_buff;
  int* d_indice_buff;
  unsigned int* d_prefixoffsets_buff;

  d_pivotPoints_buff = sycl::malloc_device<float>(DIVISIONS, q_ct1);
  q_ct1.memcpy(d_pivotPoints_buff, pivotPoints, sizeof(float) * DIVISIONS);

  d_indice_buff = sycl::malloc_device<int>(listsize, q_ct1);
  d_prefixoffsets_buff = (unsigned int *)sycl::malloc_device(
      sizeof(unsigned int) * blocks * BUCKET_BLOCK_MEMORY, q_ct1);

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        s_offset_acc_ct1(sycl::range<1>(1024 /*BUCKET_BLOCK_MEMORY*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                              sycl::range<3>(1, 1, BUCKET_THREAD_N),
                          sycl::range<3>(1, 1, BUCKET_THREAD_N)),
        [=](sycl::nd_item<3> item_ct1) {
          bucketcount(d_input_buff, d_indice_buff, d_prefixoffsets_buff,
                      d_pivotPoints_buff, listsize, item_ct1,
                      s_offset_acc_ct1.get_pointer());
        });
  });

#ifdef DEBUG
  printf("d_indice\n");
  for (int i = 0; i < listsize; i++) printf("%d %d\n", i, d_indice[i]);
  printf("d_prefixoffsets\n");
  for (int i = 0; i < blocks * BUCKET_BLOCK_MEMORY; i++) 
    printf("%d %d\n", i, d_prefixoffsets[i]);
#endif

  //
  //  ///////////////////////////////////////////////////////////////////////////
  //  // Prefix scan offsets and align each division to float4 (required by
  //  // mergesort)
  //  ///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
  size_t localpre = BUCKET_WG_SIZE_0;
#else
  size_t localpre = 128;
#endif
  size_t globalpre = DIVISIONS;

  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, globalpre / localpre) *
                              sycl::range<3>(1, 1, localpre),
                          sycl::range<3>(1, 1, localpre)),
        [=](sycl::nd_item<3> item_ct1) {
          bucketprefix(d_prefixoffsets_buff, d_offsets_buff, blocks, item_ct1);
        });
  });

  // copy the sizes from device to host
  q_ct1
      .memcpy(h_offsets, d_offsets_buff, sizeof(unsigned int) * DIVISIONS)
      .wait();

  origOffsets[0] = 0;
  for(int i=0; i<DIVISIONS; i++){
    origOffsets[i+1] = h_offsets[i] + origOffsets[i];
    if((h_offsets[i] % 4) != 0){
      nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i];
    }
    else nullElements[i] = 0;
  }
  for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4;
  for(int i=0; i<DIVISIONS; i++) {
    if((h_offsets[i] % 4) != 0)  h_offsets[i] = (h_offsets[i] & ~3) + 4;
  }
  for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i];
  for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1];
  h_offsets[0] = 0;

#ifdef DEBUG
  printf("h_offsets\n");
  for (int i = 0; i < DIVISIONS; i++) printf("%d %d\n", i, h_offsets[0]);
  printf("nullelements\n");
  for (int i = 0; i < DIVISIONS; i++) printf("%d %d\n", i, nullElements[0]);
  printf("origOffsets\n");
  for (int i = 0; i < DIVISIONS; i++) printf("%d %d\n", i, origOffsets[0]);
#endif

  //  ///////////////////////////////////////////////////////////////////////////
  //  // Finally, sort the lot
  //  ///////////////////////////////////////////////////////////////////////////

  // update the h_offsets on the device
  q_ct1.memcpy(d_offsets_buff, h_offsets, sizeof(unsigned int) * DIVISIONS);

  float* d_bucketOutput;
  d_bucketOutput =
      sycl::malloc_device<float>((listsize + DIVISIONS * 4), q_ct1);
  q_ct1.memcpy(d_bucketOutput, d_output,
               sizeof(float) * (listsize + DIVISIONS * 4));

  //buffer<float,1> d_bucketOutput(d_output, listsize + DIVISIONS*4, props);

  size_t localfinal = BUCKET_THREAD_N;
  size_t globalfinal = blocks*BUCKET_THREAD_N;

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        s_offset_acc_ct1(sycl::range<1>(1024 /*BUCKET_BLOCK_MEMORY*/), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                              sycl::range<3>(1, 1, BUCKET_THREAD_N),
                          sycl::range<3>(1, 1, BUCKET_THREAD_N)),
        [=](sycl::nd_item<3> item_ct1) {
          bucketsort(d_input_buff, d_indice_buff, d_bucketOutput,
                     d_prefixoffsets_buff, d_offsets_buff, listsize, item_ct1,
                     s_offset_acc_ct1.get_pointer());
        });
  });

  q_ct1
      .memcpy(d_output, d_bucketOutput,
              sizeof(float) * (listsize + DIVISIONS * 4))
      .wait();

  sycl::free(d_bucketOutput, q_ct1);
  sycl::free(d_input_buff, q_ct1);
  sycl::free(d_offsets_buff, q_ct1);
  sycl::free(d_pivotPoints_buff, q_ct1);
  sycl::free(d_indice_buff, q_ct1);
  sycl::free(d_prefixoffsets_buff, q_ct1);
  free(pivotPoints);
  free(historesult);
#ifdef DEBUG
  free(d_indice);
  free(d_prefixoffsets);
#endif
}
////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  float elemsPerSlice = listsize/(float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;
  for(int i=0; i<histosize; i++)
  {
    if(i == histosize - 1){
      if(!(p_idx < divisions)){
        pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      }
      break;
    }
    while(histogram[i] > we_need){
      if(!(p_idx < divisions)){
        printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
        exit(0);
      }
      pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      startsAt += (we_need/histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }
    // grab what we can from what remains of it
    we_need -= histogram[i];

    startsAt = endsAt;
    endsAt += histo_width;
  }
  while(p_idx < divisions){
    pivotPoints[p_idx] = pivotPoints[p_idx-1];
    p_idx++;
  }
}
