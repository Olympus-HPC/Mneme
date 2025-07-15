/*
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include "haar.h"
#include "image.h"
#include "stdio-wrapper.h"

/* TODO: use matrices */
/* classifier parameters */
/************************************
 * Notes:
 * To paralleism the filter,
 * these monolithic arrays may
 * need to be splitted or duplicated
 ***********************************/
static int *stages_array;
static int *rectangles_array;
static int *weights_array;
static int *alpha1_array;
static int *alpha2_array;
static int *tree_thresh_array;
static int *stages_thresh_array;
static int **scaled_rectangles_array;

int clock_counter = 0;
float n_features = 0;

int iter_counter = 0;

/* compute integral images */
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum );

/* scale down the image */
void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighbor (MyImage *src, MyImage *dst);

/* rounding function */
inline int myRound( float value )
{
  return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
 * Function: detectObjects
 * Description: It calls all the major steps
 ******************************************************/

std::vector<MyRect> detectObjects(
  MyImage* _img, MySize minSize, MySize maxSize, myCascade* cascade,
  float scaleFactor, int minNeighbors, int total_nodes)
{
  /* group overlaping windows */
  const float GROUP_EPS = 0.4f;
  /* pointer to input image */
  MyImage *img = _img;
  /***********************************
   * create structs for images
   * see haar.h for details 
   * img1: normal image (unsigned char)
   * sum1: integral image (int)
   * sqsum1: square integral image (int)
   **********************************/
  MyImage image1Obj;
  MyIntImage sum1Obj;
  MyIntImage sqsum1Obj;
  /* pointers for the created structs */
  MyImage *img1 = &image1Obj;
  MyIntImage *sum1 = &sum1Obj;
  MyIntImage *sqsum1 = &sqsum1Obj;

  /********************************************************
   * allCandidates is the preliminaray face candidate,
   * which will be refined later.
   *
   * std::vector is a sequential container 
   * http://en.wikipedia.org/wiki/Sequence_container_(C++) 
   *
   * Each element of the std::vector is a "MyRect" struct 
   * MyRect struct keeps the info of a rectangle (see haar.h)
   * The rectangle contains one face candidate 
   *****************************************************/
  std::vector<MyRect> allCandidates;

  /* scaling factor */
  float factor;

  /* maxSize */
  if( maxSize.height == 0 || maxSize.width == 0 )
  {
    maxSize.height = img->height;
    maxSize.width = img->width;
  }

  /* window size of the training set */
  MySize winSize0 = cascade->orig_window_size;

  /* malloc for img1: unsigned char */
  createImage(img->width, img->height, img1);
  /* malloc for sum1: unsigned char */
  createSumImage(img->width, img->height, sum1);
  /* malloc for sqsum1: unsigned char */
  createSumImage(img->width, img->height, sqsum1);

  /* initial scaling factor */
  factor = 1;

#ifdef GPU
  int *d_rectangles_array;
  cudaMalloc((void**)&d_rectangles_array, sizeof(int)*total_nodes*12);
  cudaMemcpy(d_rectangles_array, rectangles_array, sizeof(int)*total_nodes*12, cudaMemcpyHostToDevice);
#endif

  /* iterate over the image pyramid */
  for( factor = 1; ; factor *= scaleFactor )
  {
    /* iteration counter */
    iter_counter++;

    /* size of the image scaled up */
    MySize winSize = { myRound(winSize0.width*factor), myRound(winSize0.height*factor) };

    /* size of the image scaled down (from bigger to smaller) */
    MySize sz = { static_cast<int>( img->width/factor ), static_cast<int>( img->height/factor ) };

    /* difference between sizes of the scaled image and the original detection window */
    MySize sz1 = { sz.width - winSize0.width, sz.height - winSize0.height };

    /* if the actual scaled image is smaller than the original detection window, break */
    if( sz1.width < 0 || sz1.height < 0 )
      break;

    /* if a minSize different from the original detection window is specified, continue to the next scaling */
    if( winSize.width < minSize.width || winSize.height < minSize.height )
      continue;

    /*************************************
     * Set the width and height of 
     * img1: normal image (unsigned char)
     * sum1: integral image (int)
     * sqsum1: squared integral image (int)
     * see image.c for details
     ************************************/
    setImage(sz.width, sz.height, img1);
    setSumImage(sz.width, sz.height, sum1);
    setSumImage(sz.width, sz.height, sqsum1);

    /***************************************
     * Compute-intensive step:
     * building image pyramid by downsampling
     * downsampling using nearest neighbor
     **************************************/
    nearestNeighbor(img, img1);

    /***************************************************
     * Compute-intensive step:
     * At each scale of the image pyramid,
     * compute a new integral and squared integral image
     ***************************************************/
    integralImages(img1, sum1, sqsum1);

    /* sets images for haar classifier cascade */
    /**************************************************
     * Note:
     * Summing pixels within a haar window is done by
     * using four corners of the integral image:
     * http://en.wikipedia.org/wiki/Summed_area_table
     * 
     * This function loads the four corners,
     * but does not do compuation based on four coners.
     * The computation is done next in ScaleImage_Invoker
     *************************************************/
    setImageForCascadeClassifier(cascade, sum1, sqsum1, 
                                 #ifdef GPU
                                 d_rectangles_array, 
                                 #endif
                                 total_nodes);

    /* print out for each scale of the image pyramid */
    printf("detecting faces, iter := %d\n", iter_counter);

    /****************************************************
     * Process the current scale with the cascaded fitler.
     * The main computations are invoked by this function.
     * Optimization oppurtunity:
     * the same cascade filter is invoked each time
     ***************************************************/
    ScaleImage_Invoker(cascade, factor, sum1->height, sum1->width,
        allCandidates);
  } /* end of the factor loop, finish all scales in pyramid*/

  if( minNeighbors != 0)
  {
    groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
  }

  freeImage(img1);
  freeSumImage(sum1);
  freeSumImage(sqsum1);
#ifdef GPU
  cudaFree(d_rectangles_array);
#endif
  return allCandidates;
}

/***********************************************
 * Note:
 * The int_sqrt is softwar integer squre root.
 * GPU has hardware for floating squre root (sqrtf).
 * In GPU, it is wise to convert an int variable
 * into floating point, and use HW sqrtf function.
 * More info:
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
 **********************************************/
/*****************************************************
 * The int_sqrt is only used in runCascadeClassifier
 * If you want to replace int_sqrt with HW sqrtf in GPU,
 * simple look into the runCascadeClassifier function.
 *****************************************************/
unsigned int int_sqrt (unsigned int value)
{
  int i;
  unsigned int a = 0, b = 0, c = 0;
  for (i=0; i < (32 >> 1); i++)
  {
    c<<= 2;
#define UPPERBITS(value) (value>>30)
    c += UPPERBITS(value);
#undef UPPERBITS
    value <<= 2;
    a <<= 1;
    b = (a<<1) | 1;
    if (c >= b)
    {
      c -= b;
      a++;
    }
  }
  return a;
}

#ifdef GPU
__global__ 
void filter_kernel (const int*__restrict__ d_rectangles_array, 
                    int**__restrict__ d_scaled_rectangles_array, 
                    int*__restrict__ data, int width, int total_nodes)
{
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid >= total_nodes) return;

  int idx = gid * 12;
  for (int k = 0; k < 3; k++)
  {
    int tr_x = d_rectangles_array[idx + k * 4];
    int tr_y = d_rectangles_array[idx + 1 + k * 4];
    int tr_width = d_rectangles_array[idx + 2 + k * 4];
    int tr_height = d_rectangles_array[idx + 3 + k * 4];
    int *p0 = data + width * (tr_y) + (tr_x);
    int *p1 = data + width * (tr_y) + (tr_x + tr_width);
    int *p2 = data + width * (tr_y + tr_height) + (tr_x);
    int *p3 = data + width * (tr_y + tr_height) + (tr_x + tr_width);
    if (k < 2)
    {
      d_scaled_rectangles_array[idx + k * 4]     = p0;
      d_scaled_rectangles_array[idx + k * 4 + 1] = p1; 
      d_scaled_rectangles_array[idx + k * 4 + 2] = p2; 
      d_scaled_rectangles_array[idx + k * 4 + 3] = p3; 
    }
    else
    {
      bool z = ((tr_x == 0) && (tr_y == 0) && (tr_width == 0) && (tr_height == 0));
      d_scaled_rectangles_array[idx + k * 4]     = z ? NULL : p0;
      d_scaled_rectangles_array[idx + k * 4 + 1] = z ? NULL : p1;
      d_scaled_rectangles_array[idx + k * 4 + 2] = z ? NULL : p2;
      d_scaled_rectangles_array[idx + k * 4 + 3] = z ? NULL : p3;
    } /* end of branch if(k<2) */
  }   /* end of k loop */
}
#endif

void setImageForCascadeClassifier( myCascade* _cascade, MyIntImage* _sum, MyIntImage* _sqsum, 
#ifdef GPU
    int* d_rectangles_array, 
#endif
    int total_nodes)
{
  MyIntImage *sum = _sum;
  MyIntImage *sqsum = _sqsum;
  myCascade* cascade = _cascade;
  MyRect equRect;

  cascade->sum = *sum;
  cascade->sqsum = *sqsum;

  equRect.x = equRect.y = 0;
  equRect.width = cascade->orig_window_size.width;
  equRect.height = cascade->orig_window_size.height;

  cascade->inv_window_area = equRect.width*equRect.height;

  cascade->p0 = (sum->data) ;
  cascade->p1 = (sum->data +  equRect.width - 1) ;
  cascade->p2 = (sum->data + sum->width*(equRect.height - 1));
  cascade->p3 = (sum->data + sum->width*(equRect.height - 1) + equRect.width - 1);
  cascade->pq0 = (sqsum->data);
  cascade->pq1 = (sqsum->data +  equRect.width - 1) ;
  cascade->pq2 = (sqsum->data + sqsum->width*(equRect.height - 1));
  cascade->pq3 = (sqsum->data + sqsum->width*(equRect.height - 1) + equRect.width - 1);

#ifdef GPU
  /****************************************
   * Load the index of the four corners 
   * of the filter rectangle
   **************************************/
  int **d_scaled_rectangles_array;
  cudaMalloc((void**)&d_scaled_rectangles_array, sizeof(int*)*total_nodes*12);

  dim3 grid ((total_nodes+255)/256);
  dim3 block (256);
  filter_kernel<<<grid, block>>>(d_rectangles_array, 
      d_scaled_rectangles_array, sum->data, sum->width, total_nodes);

  cudaMemcpy(scaled_rectangles_array, d_scaled_rectangles_array, 
      sizeof(int*)*total_nodes*12, cudaMemcpyDeviceToHost);

  cudaFree(d_scaled_rectangles_array);
#else
  /****************************************
   * Load the index of the four corners 
   * of the filter rectangle
   **************************************/

  int r_index = 0;

  // collapsed the nested loop on the cpu too
  for (int i = 0; i < total_nodes; i++) 
  {
    /* loop over the number of rectangles */
    for(int k = 0; k < 3; k++)
    {
      MyRect tr;
      tr.x = rectangles_array[r_index + k*4];
      tr.width = rectangles_array[r_index + 2 + k*4];
      tr.y = rectangles_array[r_index + 1 + k*4];
      tr.height = rectangles_array[r_index + 3 + k*4];
      if (k < 2)
      {
        scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
        scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
        scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
        scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
      }
      else
      {
        if ((tr.x == 0)&& (tr.y == 0) &&(tr.width == 0) &&(tr.height == 0))
        {
          scaled_rectangles_array[r_index + k*4] = NULL ;
          scaled_rectangles_array[r_index + k*4 + 1] = NULL ;
          scaled_rectangles_array[r_index + k*4 + 2] = NULL;
          scaled_rectangles_array[r_index + k*4 + 3] = NULL;
        }
        else
        {
          scaled_rectangles_array[r_index + k*4] = (sum->data + sum->width*(tr.y ) + (tr.x )) ;
          scaled_rectangles_array[r_index + k*4 + 1] = (sum->data + sum->width*(tr.y ) + (tr.x  + tr.width)) ;
          scaled_rectangles_array[r_index + k*4 + 2] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x ));
          scaled_rectangles_array[r_index + k*4 + 3] = (sum->data + sum->width*(tr.y  + tr.height) + (tr.x  + tr.width));
        }
      } /* end of branch if(k<2) */
    } /* end of k loop*/
    r_index+=12;
  } /* end i loop */
#endif
}

/****************************************************
 * evalWeakClassifier:
 * the actual computation of a haar filter.
 * More info:
 * http://en.wikipedia.org/wiki/Haar-like_features
 ***************************************************/
inline int evalWeakClassifier(int variance_norm_factor, int p_offset, int tree_index, int w_index, int r_index )
{
  /* the node threshold is multiplied by the standard deviation of the image */
  int t = tree_thresh_array[tree_index] * variance_norm_factor;

  int sum = (*(scaled_rectangles_array[r_index] + p_offset)
      - *(scaled_rectangles_array[r_index + 1] + p_offset)
      - *(scaled_rectangles_array[r_index + 2] + p_offset)
      + *(scaled_rectangles_array[r_index + 3] + p_offset))
    * weights_array[w_index];


  sum += (*(scaled_rectangles_array[r_index+4] + p_offset)
      - *(scaled_rectangles_array[r_index + 5] + p_offset)
      - *(scaled_rectangles_array[r_index + 6] + p_offset)
      + *(scaled_rectangles_array[r_index + 7] + p_offset))
    * weights_array[w_index + 1];

  if ((scaled_rectangles_array[r_index+8] != NULL))
    sum += (*(scaled_rectangles_array[r_index+8] + p_offset)
        - *(scaled_rectangles_array[r_index + 9] + p_offset)
        - *(scaled_rectangles_array[r_index + 10] + p_offset)
        + *(scaled_rectangles_array[r_index + 11] + p_offset))
      * weights_array[w_index + 2];

  if(sum >= t)
    return alpha2_array[tree_index];
  else
    return alpha1_array[tree_index];
}

int runCascadeClassifier( myCascade* _cascade, MyPoint pt, int start_stage )
{
  int p_offset, pq_offset;
  int i, j;
  unsigned int mean;
  unsigned int variance_norm_factor;
  int haar_counter = 0;
  int w_index = 0;
  int r_index = 0;
  int stage_sum;
  myCascade* cascade;
  cascade = _cascade;

  p_offset = pt.y * (cascade->sum.width) + pt.x;
  pq_offset = pt.y * (cascade->sqsum.width) + pt.x;

  /**************************************************************************
   * Image normalization
   * mean is the mean of the pixels in the detection window
   * cascade->pqi[pq_offset] are the squared pixel values (using the squared integral image)
   * inv_window_area is 1 over the total number of pixels in the detection window
   *************************************************************************/

  variance_norm_factor =  (cascade->pq0[pq_offset] - cascade->pq1[pq_offset] - cascade->pq2[pq_offset] + cascade->pq3[pq_offset]);
  mean = (cascade->p0[p_offset] - cascade->p1[p_offset] - cascade->p2[p_offset] + cascade->p3[p_offset]);

  variance_norm_factor = (variance_norm_factor*cascade->inv_window_area);
  variance_norm_factor =  variance_norm_factor - mean*mean;

  /***********************************************
   * Note:
   * The int_sqrt is softwar integer squre root.
   * GPU has hardware for floating squre root (sqrtf).
   * In GPU, it is wise to convert the variance norm
   * into floating point, and use HW sqrtf function.
   * More info:
   * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
   **********************************************/
  if( variance_norm_factor > 0 )
    variance_norm_factor = int_sqrt(variance_norm_factor);
  else
    variance_norm_factor = 1;

  /**************************************************
   * The major computation happens here.
   * For each scale in the image pyramid,
   * and for each shifted step of the filter,
   * send the shifted window through cascade filter.
   *
   * Note:
   *
   * Stages in the cascade filter are independent.
   * However, a face can be rejected by any stage.
   * Running stages in parallel delays the rejection,
   * which induces unnecessary computation.
   *
   * Filters in the same stage are also independent,
   * except that filter results need to be merged,
   * and compared with a per-stage threshold.
   *************************************************/
  for( i = start_stage; i < cascade->n_stages; i++ )
  {

    /****************************************************
     * A shared variable that induces false dependency
     * 
     * To avoid it from limiting parallelism,
     * we can duplicate it multiple times,
     * e.g., using stage_sum_array[number_of_threads].
     * Then threads only need to sync at the end
     ***************************************************/
    stage_sum = 0;

    for( j = 0; j < stages_array[i]; j++ )
    {
      /**************************************************
       * Send the shifted window to a haar filter.
       **************************************************/
      stage_sum += evalWeakClassifier(variance_norm_factor, p_offset, haar_counter, w_index, r_index);
      n_features++;
      haar_counter++;
      w_index+=3;
      r_index+=12;
    } /* end of j loop */

    /**************************************************************
     * threshold of the stage. 
     * If the sum is below the threshold, 
     * no faces are detected, 
     * and the search is abandoned at the i-th stage (-i).
     * Otherwise, a face is detected (1)
     **************************************************************/

    /* the number "0.4" is empirically chosen for 5kk73 */
    if( stage_sum < 0.4*stages_thresh_array[i] ){
      return -i;
    } /* end of the per-stage thresholding */
  } /* end of i loop */
  return 1;
}

void ScaleImage_Invoker( myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec)
{
  myCascade* cascade = _cascade;

  float factor = _factor;
  MyPoint p;
  int result;
  int y1, y2, x2, x, y, step;
  std::vector<MyRect> *vec = &_vec;

  MySize winSize0 = cascade->orig_window_size;
  MySize winSize;

  winSize.width =  myRound(winSize0.width*factor);
  winSize.height =  myRound(winSize0.height*factor);
  y1 = 0;

  /********************************************
   * When filter window shifts to image boarder,
   * some margin need to be kept
   *********************************************/
  y2 = sum_row - winSize0.height;
  x2 = sum_col - winSize0.width;

  /********************************************
   * Step size of filter window shifting
   * Reducing step makes program faster,
   * but decreases quality of detection.
   * example:
   * step = factor > 2 ? 1 : 2;
   * 
   * For 5kk73, 
   * the factor and step can be kept constant,
   * unless you want to change input image.
   *
   * The step size is set to 1 for 5kk73,
   * i.e., shift the filter window by 1 pixel.
   *******************************************/  
  step = 1;

  /**********************************************
   * Shift the filter window over the image.
   * Each shift step is independent.
   * Shared data structure may limit parallelism.
   *
   * Some random hints (may or may not work):
   * Split or duplicate data structure.
   * Merge functions/loops to increase locality
   * Tiling to increase computation-to-memory ratio
   *********************************************/
  for( x = 0; x <= x2; x += step )
    for( y = y1; y <= y2; y += step )
    {
      p.x = x;
      p.y = y;

      /*********************************************
       * Optimization Oppotunity:
       * The same cascade filter is used each time
       ********************************************/
      result = runCascadeClassifier( cascade, p, 0 );

      /*******************************************************
       * If a face is detected,
       * record the coordinates of the filter window
       * the "push_back" function is from std:vec, more info:
       * http://en.wikipedia.org/wiki/Sequence_container_(C++)
       *
       * Note that, if the filter runs on GPUs,
       * the push_back operation is not possible on GPUs.
       * The GPU may need to use a simpler data structure,
       * e.g., an array, to store the coordinates of face,
       * which can be later memcpy from GPU to CPU to do push_back
       *******************************************************/
      if( result > 0 )
      {
        MyRect r = {myRound(x*factor), myRound(y*factor), winSize.width, winSize.height};
        vec->push_back(r);
      }
    }
}

/*****************************************************
 * Compute the integral image (and squared integral)
 * Integral image helps quickly sum up an area.
 * More info:
 * http://en.wikipedia.org/wiki/Summed_area_table
 ****************************************************/
void integralImages( MyImage *src, MyIntImage *sum, MyIntImage *sqsum )
{
  int x, y, s, sq, t, tq;
  unsigned char it;
  int height = src->height;
  int width = src->width;
  unsigned char *data = src->data;
  int * sumData = sum->data;
  int * sqsumData = sqsum->data;
  for( y = 0; y < height; y++)
  {
    s = 0;
    sq = 0;
    /* loop over the number of columns */
    for( x = 0; x < width; x ++)
    {
      it = data[y*width+x];
      /* sum of the current row (integer)*/
      s += it; 
      sq += it*it;

      t = s;
      tq = sq;
      if (y != 0)
      {
        t += sumData[(y-1)*width+x];
        tq += sqsumData[(y-1)*width+x];
      }
      sumData[y*width+x]=t;
      sqsumData[y*width+x]=tq;
    }
  }
}

/***********************************************************
 * This function downsample an image using nearest neighbor
 * It is used to build the image pyramid
 **********************************************************/
void nearestNeighbor (MyImage *src, MyImage *dst)
{
  int y;
  int j;
  int x;
  int i;
  unsigned char* t;
  unsigned char* p;
  int w1 = src->width;
  int h1 = src->height;
  int w2 = dst->width;
  int h2 = dst->height;

  int rat = 0;

  unsigned char* src_data = src->data;
  unsigned char* dst_data = dst->data;


  int x_ratio = (int)((w1<<16)/w2) +1;
  int y_ratio = (int)((h1<<16)/h2) +1;

  for (i=0;i<h2;i++)
  {
    t = dst_data + i*w2;
    y = ((i*y_ratio)>>16);
    p = src_data + y*w1;
    rat = 0;
    for (j=0;j<w2;j++)
    {
      x = (rat>>16);
      *t++ = p[x];
      //printf("x_ratio=%d y_ratio=%d downsampling at src image: row=%d col=%d\n", x_ratio, y_ratio, y, x);
      rat += x_ratio;
    }
  }
}

int readTextClassifier(const char *info_file, const char *class_file)
{
  /*number of stages of the cascade classifier*/
  int stages = 0;
  /*total number of weak classifiers (one node each)*/
  int total_nodes = 0;
  int i, j, k, l;
  char mystring [12];
  int r_index = 0;
  int w_index = 0;
  int tree_index = 0;
  FILE *finfo = fopen(info_file, "r");
  if (finfo == NULL) {
    fprintf(stderr, "Failed to open file %s. Exit\n", info_file);
    return -1;
  }

  FILE *fp = fopen(class_file, "r");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open file %s. Exit\n", class_file);
    return -1;
  }

  /**************************************************
   * how many stages are in the cascaded filter? 
   * the first line of info.txt is the number of stages 
   **************************************************/
  if ( fgets (mystring , 12 , finfo) != NULL )
    stages = atoi(mystring);

  if (stages == 0) {
    printf("The number of stages in the cascaded filter must be postive\n");
    return -1;
  }

  stages_array = (int *)malloc(sizeof(int)*stages);

  /**************************************************
   * how many filters in each stage? 
   * They are specified in info.txt,
   * starting from second line.
   * (from line 2 to line 26)
   *************************************************/
  i = 0;
  while ( fgets (mystring , 12 , finfo) != NULL )
  {
    stages_array[i] = atoi(mystring);
    total_nodes += stages_array[i];
    i++;
  }
  fclose(finfo);


  /* TODO: use matrices where appropriate */
  /***********************************************
   * Allocate a lot of array structures
   * Note that, to increase parallelism,
   * some arrays need to be splitted or duplicated
   **********************************************/
  rectangles_array = (int*)malloc(sizeof(int)*total_nodes*12);
  scaled_rectangles_array = (int**)malloc(sizeof(int*)*total_nodes*12);
  weights_array = (int*)malloc(sizeof(int)*total_nodes*3);
  alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
  alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
  tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
  stages_thresh_array = (int*)malloc(sizeof(int)*stages);

  /******************************************
   * Read the filter parameters in class.txt
   *
   * Each stage of the cascaded filter has:
   * 18 parameter per filter x tilter per stage
   * + 1 threshold per stage
   *
   * For example, 
   * the first stage has 9 filters,
   * the first stage is specified using
   * 18 * 9 + 1 = 163 parameters
   * They are line 1 to 163 of class.txt
   *
   * The 18 parameters for each filter are:
   * 1 to 4: coordinates of rectangle 1
   * 5: weight of rectangle 1
   * 6 to 9: coordinates of rectangle 2
   * 10: weight of rectangle 2
   * 11 to 14: coordinates of rectangle 3
   * 15: weight of rectangle 3
   * 16: threshold of the filter
   * 17: alpha 1 of the filter
   * 18: alpha 2 of the filter
   ******************************************/

  /* loop over n of stages */
  for (i = 0; i < stages; i++)
  {    /* loop over n of trees */
    for (j = 0; j < stages_array[i]; j++)
    {  /* loop over n of rectangular features */
      for(k = 0; k < 3; k++)
      {  /* loop over the n of vertices */
        for (l = 0; l <4; l++)
        {
          if (fgets (mystring , 12 , fp) != NULL)
            rectangles_array[r_index] = atoi(mystring);
          else
            break;
          r_index++;
        } /* end of l loop */
        if (fgets (mystring , 12 , fp) != NULL)
        {
          weights_array[w_index] = atoi(mystring);
          /* Shift value to avoid overflow in the haar evaluation */
          /*TODO: make more general */
          /*weights_array[w_index]>>=8; */
        }
        else
          break;
        w_index++;
      } /* end of k loop */
      if (fgets (mystring , 12 , fp) != NULL)
        tree_thresh_array[tree_index]= atoi(mystring);
      else
        break;
      if (fgets (mystring , 12 , fp) != NULL)
        alpha1_array[tree_index]= atoi(mystring);
      else
        break;
      if (fgets (mystring , 12 , fp) != NULL)
        alpha2_array[tree_index]= atoi(mystring);
      else
        break;
      tree_index++;
      if (j == stages_array[i]-1)
      {
        if (fgets (mystring , 12 , fp) != NULL)
          stages_thresh_array[i] = atoi(mystring);
        else
          break;
      }
    } /* end of j loop */
  } /* end of i loop */
  fclose(fp);
  return total_nodes;
}

void releaseTextClassifier()
{
  free(stages_array);
  free(rectangles_array);
  free(scaled_rectangles_array);
  free(weights_array);
  free(tree_thresh_array);
  free(alpha1_array);
  free(alpha2_array);
  free(stages_thresh_array);
}
/* End of file. */
