/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#include <cmath>
#include <iostream>
#include <fstream>
#include <cstring>
#include "benchmark.h"
#include "datatypes.h"

// maximum allowed deviation from the reference results
#define MAX_EPS 0.001
// number of GPU threads
#define THREADS 256

class points2image : public kernel {
  private:
    // the number of testcases read
    int read_testcases = 0;
    // testcase and reference data streams
    std::ifstream input_file, output_file;
    // whether critical deviation from the reference data has been detected
    bool error_so_far = false;
    // deviation from the reference data
    double max_delta = 0.0;
  public:
    /*
     * Initializes the kernel. Must be called before run().
     */
    virtual void init();
    /**
     * Performs the kernel operations on all input and output data.
     * p: number of testcases to process in one step
     */
    virtual void run(int p = 1);
    /**
     * Finally checks whether all input data has been processed successfully.
     */
    virtual bool check_output();
    // the point clouds to process in one iteration
    PointCloud2* pointcloud2 = NULL;
    // the associated camera extrinsic matrices
    Mat44* cameraExtrinsicMat = NULL;
    // the associated camera intrinsic matrices
    Mat33* cameraMat = NULL;
    // distance coefficients for the current iteration
    Vec5* distCoeff = NULL;
    // image sizes for the current iteration
    ImageSize* imageSize = NULL;
    // Algorithm results for the current iteration
    PointsImage* results = NULL;
  protected:
    /**
     * Reads the next test cases.
     * count: the number of testcases to read
     * returns: the number of testcases actually read
     */
    virtual int read_next_testcases(int count);
    /**
     * Compares the results from the algorithm with the reference data.
     * count: the number of testcases processed 
     */
    virtual void check_next_outputs(int count);
    /**
     * Reads the number of testcases in the data set.
     */
    int  read_number_testcases(std::ifstream& input_file);
};

// image storage
// expected image size 800x600 pixels with 4 components per pixel
__device__ __managed__ float result_buffer[800*600*4];

int points2image::read_number_testcases(std::ifstream& input_file)
{
  // reads the number of testcases in the data stream
  int number;
  try {
    input_file.read((char*)&(number), sizeof(int));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the number of testcases.");
  }
  return number;
}
/**
 * Reads the next point cloud
 */
void  parsePointCloud(std::ifstream& input_file, PointCloud2* pointcloud2) {
  input_file.read((char*)&(pointcloud2->height), sizeof(int));
  input_file.read((char*)&(pointcloud2->width), sizeof(int));
  input_file.read((char*)&(pointcloud2->point_step), sizeof(uint));
#ifdef DEBUG
  printf("PointCloud: height=%d width=%d point_step=%d\n",
          pointcloud2->height , pointcloud2->width , pointcloud2->point_step);
#endif
  cudaMallocManaged(&pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
  input_file.read((char*)pointcloud2->data, pointcloud2->height * pointcloud2->width * pointcloud2->point_step);
}

/**
 * Parses the next camera extrinsic matrix.
 */
void  parseCameraExtrinsicMat(std::ifstream& input_file, Mat44* cameraExtrinsicMat) {
  try {
    for (int h = 0; h < 4; h++)
      for (int w = 0; w < 4; w++)
        input_file.read((char*)&(cameraExtrinsicMat->data[h][w]),sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next extrinsic matrix.");    
  }
}
/**
 * Parses the next camera matrix.
 */
void parseCameraMat(std::ifstream& input_file, Mat33* cameraMat ) {
  try {
    for (int h = 0; h < 3; h++)
      for (int w = 0; w < 3; w++)
        input_file.read((char*)&(cameraMat->data[h][w]), sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next camera matrix.");
  }
}

/**
 * Parses the next distance coefficients.
 */
void  parseDistCoeff(std::ifstream& input_file, Vec5* distCoeff) {
  try {
    for (int w = 0; w < 5; w++)
      input_file.read((char*)&(distCoeff->data[w]), sizeof(double));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next set of distance coefficients.");
  }
}

/**
 * Parses the next image sizes.
 */
void  parseImageSize(std::ifstream& input_file, ImageSize* imageSize) {
  try {
    input_file.read((char*)&(imageSize->width), sizeof(int));
    input_file.read((char*)&(imageSize->height), sizeof(int));
  } catch (std::ifstream::failure) {
    throw std::ios_base::failure("Error reading the next image size.");
  }
}

/**
 * Parses the next reference image.
 */
void parsePointsImage(std::ifstream& output_file, PointsImage* goldenResult) {
  try {
    // read data of static size
    output_file.read((char*)&(goldenResult->image_width), sizeof(int));
    output_file.read((char*)&(goldenResult->image_height), sizeof(int));
    output_file.read((char*)&(goldenResult->max_y), sizeof(int));
    output_file.read((char*)&(goldenResult->min_y), sizeof(int));
    int pos = 0;
    int elements = goldenResult->image_height * goldenResult->image_width;
    goldenResult->intensity = new float[elements];
    goldenResult->distance = new float[elements];
    goldenResult->min_height = new float[elements];
    goldenResult->max_height = new float[elements];
    // read data of variable size
    for (int h = 0; h < goldenResult->image_height; h++)
      for (int w = 0; w < goldenResult->image_width; w++)
      {
        output_file.read((char*)&(goldenResult->intensity[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->distance[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->min_height[pos]), sizeof(float));
        output_file.read((char*)&(goldenResult->max_height[pos]), sizeof(float));
        pos++;
      }
  } catch (std::ios_base::failure) {
    throw std::ios_base::failure("Error reading the next reference image.");
  }
}

// return how many could be read
int points2image::read_next_testcases(int count)
{
  int i;
  // free the memory that has been allocated in the previous iteration
  // and allocate new for the currently required data sizes
  delete [] pointcloud2;
  pointcloud2 = new PointCloud2[count];
  delete [] cameraExtrinsicMat;
  cameraExtrinsicMat = new Mat44[count];
  delete [] cameraMat;
  cameraMat = new Mat33[count];
  delete [] distCoeff;
  distCoeff = new Vec5[count];
  delete [] imageSize;
  imageSize = new ImageSize[count];
  delete [] results;
  results = new PointsImage[count];
  // read data from the next test case
  for (i = 0; (i < count) && (read_testcases < testcases); i++,read_testcases++)
  {
    try {
      parsePointCloud(input_file, pointcloud2 + i);
      parseCameraExtrinsicMat(input_file, cameraExtrinsicMat + i);
      parseCameraMat(input_file, cameraMat + i);
      parseDistCoeff(input_file, distCoeff + i);
      parseImageSize(input_file, imageSize + i);
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << std::endl;
      exit(-3);
    }
  }
  return i;
}

void points2image::init() {
  std::cout << "Open testcase and reference data streams\n";
  input_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  output_file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
  try {
    input_file.open("../../data/p2i_input.dat", std::ios::binary);
  } catch (std::ifstream::failure) {
    std::cerr << "Error opening the input data file" << std::endl;
    exit(-2);
  }
  try {
    output_file.open("../../data/p2i_output.dat", std::ios::binary);
  } catch (std::ifstream::failure) {
    std::cerr << "Error opening the output data file" << std::endl;
    exit(-2);
  }
  try {
    // consume the total number of testcases
    testcases = read_number_testcases(input_file);
    printf("the total number of testcases = %d\n", testcases);
  } catch (std::ios_base::failure& e) {
    std::cerr << e.what() << std::endl;
    exit(-3);
  }

  // prepare the first iteration
  error_so_far = false;
  max_delta = 0.0;
  pointcloud2 = NULL;
  cameraExtrinsicMat = NULL;
  cameraMat = NULL;
  distCoeff = NULL;
  imageSize = NULL;
  results = NULL;

  std::cout << "Done\n" << std::endl;
}

/**
 * Improvised atomic min function for floats
 * Does only work for positive values.
 */
__device__ __forceinline__ float atomicFloatMin(float * addr, float value) {
  return  __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
}

/** 
 * Performs the transformation for a single point.
 * cp: pointer to cloud memory
 * msg_distance: image distances
 * msg_intensity: image intensities
 * msg_min_height: image minimum heights
 * width: cloud width
 * height: cloud height
 * point_step: point stride used for indexing cloud memory
 * w: image width
 * h: image height
 * invR: matrix to apply to cloud points
 * invT: translation to apply to cloud points
 * distCoeff: distance coefficients to apply to cloud points
 * cameraMat: camera intrinsic matrix
 * min_y: lower image extend bound
 * max_y: higher image extend bound
 */
__global__ void compute_point_from_pointcloud(
    const float*  __restrict__ cp, 
          float* volatile msg_distance,
          float* volatile msg_intensity,
          float* __restrict__ msg_min_height,
    int width, int height, int point_step,
    int w, int h, 
    Mat33 invR,
    Mat13 invT,
    Vec5 distCoeff,
    Mat33 cameraMat,
    int* __restrict__ min_y,
    int* __restrict__ max_y) 
{

  // determine index in cloud memory
  int y = blockIdx.x;
  int x = blockIdx.y * THREADS + threadIdx.x;
  if (x >= width) return;

  const float* fp = (float *)((uintptr_t)cp + (x + y*width) * point_step);

  float intensity = fp[4];
  // first step of the transformation
  Mat13 point, point2;
  point2.data[0] = double(fp[0]);
  point2.data[1] = double(fp[1]);
  point2.data[2] = double(fp[2]);

  for (int row = 0; row < 3; row++) {
    point.data[row] = invT.data[row];
    for (int col = 0; col < 3; col++) 
      point.data[row] += point2.data[col] * invR.data[row][col];
  }

  // discard points of low depth
  if (point.data[2] <= 2.5) return;

  // second transformation step
  double tmpx = point.data[0] / point.data[2];
  double tmpy = point.data[1] / point.data[2];
  double r2 = tmpx * tmpx + tmpy * tmpy;
  double tmpdist = 1.0 + distCoeff.data[0] * r2 + distCoeff.data[1] * r2 * r2
                   + distCoeff.data[4] * r2 * r2 * r2;

  Point2d imagepoint;
  imagepoint.x = tmpx * tmpdist + 2.0 * distCoeff.data[2] * tmpx * tmpy
                 + distCoeff.data[3] * (r2 + 2.0 * tmpx * tmpx);
  imagepoint.y = tmpy * tmpdist + distCoeff.data[2] * (r2 + 2.0 * tmpy * tmpy)
                 + 2.0 * distCoeff.data[3] * tmpx * tmpy;

  // apply camera intrinsics to yield a point on the image
  imagepoint.x = cameraMat.data[0][0] * imagepoint.x + cameraMat.data[0][2];
  imagepoint.y = cameraMat.data[1][1] * imagepoint.y + cameraMat.data[1][2];
  int px = int(imagepoint.x + 0.5);
  int py = int(imagepoint.y + 0.5);

  float cm_point;
  int pid;
  // safe point characteristics in the image
  if (0 <= px && px < w && 0 <= py && py < h)
  {
    pid = py * w + px;
    cm_point = point.data[2] * 100.0;  // double precision multiply
    atomicCAS((int*)&msg_distance[pid], 0, __float_as_int(cm_point));
    atomicFloatMin(&msg_distance[pid], cm_point);
  }
  // synchronize required for deterministic intensity in the image
  __syncthreads();

  if (0 <= px && px < w && 0 <= py && py < h)
  {
    float newvalue = msg_distance[pid];

    // update intensity, height and image extends
    if ( newvalue>= cm_point)
    {
      msg_intensity[pid] = intensity;
      atomicMax(max_y, py);
      atomicMin(min_y, py);
    }
    msg_min_height[pid] = -1.25f;
  }
}

/**
 * This code is extracted from Autoware, file:
 * ~/Autoware/ros/src/sensing/fusion/packages/points2image/lib/points_image/points_image.cpp
 * It uses the test data that has been read before and applies the linked algorithm.
 * pointcloud2: cloud of points to transform
 * cameraExtrinsicMat: camera matrix used for transformation
 * cameraMat: camera matrix used for transformation
 * distCoeff: distance coefficients for cloud transformation
 * imageSize: the size of the resulting image
 * returns: the two dimensional image of transformed points
 */
PointsImage pointcloud2_to_image(
    const PointCloud2& pointcloud2,
    const Mat44& cameraExtrinsicMat,
    const Mat33& cameraMat, const Vec5& distCoeff,
    const ImageSize& imageSize)
{
  // initialize the resulting image data structure
  int w = imageSize.width;
  int h = imageSize.height;
  PointsImage msg;
  msg.max_y = -1;
  msg.min_y = h;
  msg.image_height = imageSize.height;
  msg.image_width = imageSize.width;
  msg.intensity = result_buffer;
  msg.distance = msg.intensity + h*w;
  msg.min_height = msg.distance + h*w;
  msg.max_height = msg.min_height + h*w;
  std::memset(msg.intensity, 0, sizeof(float)*w*h);
  std::memset(msg.distance, 0, sizeof(float)*w*h);
  std::memset(msg.min_height, 0, sizeof(float)*w*h);
  std::memset(msg.max_height, 0, sizeof(float)*w*h);

  // preprocess the given matrices
  Mat33 invR;
  Mat13 invT;
  // transposed 3x3 camera extrinsic matrix
  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++)
      invR.data[row][col] = cameraExtrinsicMat.data[col][row];
  // translation vector: (transposed camera extrinsic matrix)*(fourth column of camera extrinsic matrix)
  for (int row = 0; row < 3; row++) {
    invT.data[row] = 0.0;
    for (int col = 0; col < 3; col++)
      invT.data[row] -= invR.data[row][col] * cameraExtrinsicMat.data[col][3];
  }
  // allocate memory for additional information used by the kernel
  int *msg_min_y, *msg_max_y;
  cudaMalloc((void**)&msg_min_y, sizeof(int));
  cudaMalloc((void**)&msg_max_y, sizeof(int));
  cudaMemcpy(msg_min_y, &msg.min_y, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(msg_max_y, &msg.max_y, sizeof(int), cudaMemcpyHostToDevice);

  // call the kernel with enough threads to process the cloud in a single call
  dim3 threaddim(THREADS);
  dim3 blockdim(pointcloud2.height, (pointcloud2.width+THREADS-1)/THREADS);

  compute_point_from_pointcloud<<<blockdim, threaddim>>>(
    pointcloud2.data,
    msg.distance,
    msg.intensity,
    msg.min_height,
    pointcloud2.width,
    pointcloud2.height,
    pointcloud2.point_step,
    w, h,
    invR, invT, distCoeff, cameraMat,
    msg_min_y, msg_max_y);

  // wait for the result and read image extends
  cudaMemcpy(&msg.min_y, msg_min_y, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&msg.max_y, msg_max_y, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(msg_max_y);
  cudaFree(msg_min_y);
  cudaFree(pointcloud2.data);

  return msg;
}

void points2image::run(int p) {
  // pause while reading and comparing data
  // only run the timer when the algorithm is active
  pause_func();
  while (read_testcases < testcases)
  {
    int count = read_next_testcases(p);
    unpause_func();
    // run the algorithm for each input data set
    for (int i = 0; i < count; i++)
    {
      results[i] = pointcloud2_to_image(pointcloud2[i],
          cameraExtrinsicMat[i],
          cameraMat[i], distCoeff[i],
          imageSize[i]);
    }
    pause_func();
    // compare with the reference data
    check_next_outputs(count);
  }
}

void points2image::check_next_outputs(int count)
{
  PointsImage reference;
  // parse the next reference image
  // and compare it to the data generated by the algorithm
  for (int i = 0; i < count; i++)
  {
    try {
      parsePointsImage(output_file, &reference);
    } catch (std::ios_base::failure& e) {
      std::cerr << e.what() << std::endl;
      exit(-3);
    }
    // detect image size deviation
    if ((results[i].image_height != reference.image_height)
        || (results[i].image_width != reference.image_width))
    {
      error_so_far = true;
    }
    // detect image extend deviation
    if ((results[i].min_y != reference.min_y)
        || (results[i].max_y != reference.max_y))
    {
      error_so_far = true;
    }
    // compare all pixels
    int pos = 0;
    for (int h = 0; h < reference.image_height; h++)
      for (int w = 0; w < reference.image_width; w++)
      {
        // compare members individually and detect deviations
        if (std::fabs(reference.intensity[pos] - results[i].intensity[pos]) > max_delta)
          max_delta = fabs(reference.intensity[pos] - results[i].intensity[pos]);
        if (std::fabs(reference.distance[pos] - results[i].distance[pos]) > max_delta)
          max_delta = fabs(reference.distance[pos] - results[i].distance[pos]);
        if (std::fabs(reference.min_height[pos] - results[i].min_height[pos]) > max_delta)
          max_delta = fabs(reference.min_height[pos] - results[i].min_height[pos]);
        if (std::fabs(reference.max_height[pos] - results[i].max_height[pos]) > max_delta)
          max_delta = fabs(reference.max_height[pos] - results[i].max_height[pos]);
        pos++;
      }
    // free the memory allocated by the reference image read above
    delete [] reference.intensity;
    delete [] reference.distance;
    delete [] reference.min_height;
    delete [] reference.max_height;
  }
}

bool points2image::check_output() {
  std::cout << "checking output \n";
  input_file.close();
  output_file.close();
  std::cout << "max delta: " << max_delta << "\n";
  if ((max_delta > MAX_EPS) || error_so_far) {
    return false;
  } else {
    return true;
  }
}
// set the external kernel instance used in main()
points2image a = points2image();
kernel& myKernel = a;
