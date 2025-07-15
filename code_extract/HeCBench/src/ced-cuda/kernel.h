#ifndef __KERNEL
#define __KERNEL
#include <vector>
#include <algorithm>
#include <string.h>
#ifdef CHAI_OPENCV
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#endif

void run_cpu_threads(unsigned char *buffer0, unsigned char *buffer1, 
		unsigned char *theta, int rows, int cols, int num_threads, int t_index);

#endif
