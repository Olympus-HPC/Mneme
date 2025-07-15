#ifndef __BUCKETSORT
#define __BUCKETSORT

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define LOG_DIVISIONS	10
#define DIVISIONS		(1 << LOG_DIVISIONS)

#define BUCKET_WARP_LOG_SIZE	5
#define BUCKET_WARP_N			1

#ifdef BUCKET_WG_SIZE_1
#define BUCKET_THREAD_N BUCKET_WG_SIZE_1
#else
#define BUCKET_THREAD_N			(BUCKET_WARP_N << BUCKET_WARP_LOG_SIZE)
#endif
#define BUCKET_BLOCK_MEMORY		(DIVISIONS * BUCKET_WARP_N)
#define BUCKET_BAND				128

#define HISTOGRAM_BIN_COUNT  1024
#define HISTOGRAM_BLOCK_MEMORY  (3 * HISTOGRAM_BIN_COUNT)
#define IMUL(a, b) sycl::mul24(a, b)

void bucketSort(float *d_input, float *d_output, int listsize,
				int *sizes, int *nullElements, float minimum, float maximum,
				unsigned int *origOffsets);
double getBucketTime();

#endif
