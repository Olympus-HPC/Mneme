#include "helper.h"
#include "track_ellipse.h"
#include "misc_math.h"

#define NCIRCLES 7
#define NPOINTS 150
#define RADIUS 10
#define MIN_RAD (RADIUS - 2)
#define MAX_RAD (RADIUS * 2)
#define MaxR  (MAX_RAD + 2)

// kernels
#include "kernel_GICOV.h"
#include "kernel_dilated.h"

int main(int argc, char ** argv) {

  // Make sure the command line arguments have been specified
  if (argc != 3)  {
    fprintf(stderr, "Usage: %s <input file> <number of frames to process>", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Keep track of the start time of the program
  long long program_start_time = get_time();

  // Open video file
  char *video_file_name = argv[1];

  // Specify the number of frames to process
  int num_frames = atoi(argv[2]);

  avi_t *cell_file = AVI_open_input_file(video_file_name, 1);
  if (cell_file == NULL)  {
    char* avi_err_msg = (char*)"Error with AVI_open_input_file";
    AVI_print_error(avi_err_msg);
    exit(EXIT_FAILURE);
  }

  // Compute the sine and cosine of the angle to each point in each sample circle
  //  (which are the same across all sample circles)
  float host_sin_angle[NPOINTS], host_cos_angle[NPOINTS], theta[NPOINTS];
  for(int n = 0; n < NPOINTS; n++) {
    theta[n] = (((double) n) * 2.0 * PI) / ((double) NPOINTS);
    host_sin_angle[n] = sin(theta[n]);
    host_cos_angle[n] = cos(theta[n]);
#ifdef DEBUG
    printf("n=%d theta: %lf sin: %lf cos: %lf\n", n,theta[n], host_sin_angle[n], host_cos_angle[n]);
#endif
  }

  // Compute the (x,y) pixel offsets of each sample point in each sample circle
  int host_tX[NCIRCLES * NPOINTS], host_tY[NCIRCLES * NPOINTS];
  for (int k = 0; k < NCIRCLES; k++) {
    double rad = (double) (MIN_RAD + (2 * k)); 
    for (int n = 0; n < NPOINTS; n++) {
      host_tX[(k * NPOINTS) + n] = (int)(cos(theta[n]) * rad);
      host_tY[(k * NPOINTS) + n] = (int)(sin(theta[n]) * rad);
#ifdef DEBUG
      printf("n=%d %lf tX: %d tY: %d\n", n, cos(theta[n]) * rad, host_tX[(k * NPOINTS) + n], host_tY[(k * NPOINTS) + n]);
#endif
    }
  }

  float *host_strel = structuring_element(12);

  int i, j, *crow, *ccol, pair_counter = 0, x_result_len = 0, Iter = 20, ns = 4, k_count = 0, n;
  MAT *cellx, *celly, *A;
  double *GICOV_spots, *t, *G, *x_result, *y_result, *V, *QAX_CENTERS, *QAY_CENTERS;
  double threshold = 1.8, radius = 10.0, delta = 3.0, dt = 0.01, b = 5.0;

  // Extract a cropped version of the first frame from the video file
  MAT *image_chopped = get_frame(cell_file, 0, 1, 0);
  printf("Detecting cells in frame 0\n");

  // Get gradient matrices in x and y directions
  MAT *grad_x = gradient_x(image_chopped);
  MAT *grad_y = gradient_y(image_chopped);

  m_free(image_chopped);

  // Get GICOV matrices corresponding to image gradients

  // Determine the dimensions of the frame
  int grad_m = grad_x->m;
  int grad_n = grad_y->n;

  // Allocate host memory for grad_x and grad_y
  unsigned int grad_mem_size = sizeof(float) * grad_m * grad_n;
  float *host_grad_x = (float*) malloc(grad_mem_size);
  float *host_grad_y = (float*) malloc(grad_mem_size);
  float *host_gicov = (float *) malloc(grad_mem_size);

  // initalize float versions of grad_x and grad_y
  for (int m = 0; m < grad_m; m++) {
    for (int n = 0; n < grad_n; n++) {
      host_grad_x[(n * grad_m) + m] = (float) m_get_val(grad_x, m, n);
      host_grad_y[(n * grad_m) + m] = (float) m_get_val(grad_y, m, n);
#ifdef DEBUG
      printf("grad_x: %f grad_y: %f\n", host_grad_x[(n * grad_m) + m], host_grad_y[(n * grad_m) + m]);
#endif
    }
  }

  memset(host_gicov, 0, grad_mem_size);

  // Offload the GICOV score computation to the GPU
  long long GICOV_start_time = get_time();

  // Setup execution parameters
  int local_work_size = grad_m - (2 * MaxR); 
  int num_work_groups = grad_n - (2 * MaxR);

  size_t work_group_size = 256;
  size_t global_work_size = num_work_groups * local_work_size;
  if(global_work_size % work_group_size > 0)
    global_work_size=(global_work_size / work_group_size+1)*work_group_size;

#ifdef DEBUG
  printf("Find: local_work_size = %zu, global_work_size = %zu \n" ,work_group_size, global_work_size);
#endif

  float* d_sin_angle;
  float* d_cos_angle;
  int* d_tX;
  int* d_tY;
  float* d_grad_x;
  float* d_grad_y;
  float* d_gicov;

  cudaMalloc((void**)&d_sin_angle, sizeof(float)*NPOINTS);
  cudaMemcpyAsync(d_sin_angle, host_sin_angle, sizeof(float)*NPOINTS, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_cos_angle, sizeof(float)*NPOINTS);
  cudaMemcpyAsync(d_cos_angle, host_cos_angle, sizeof(float)*NPOINTS, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_tX, sizeof(int)*NCIRCLES*NPOINTS);
  cudaMemcpyAsync(d_tX, host_tX, sizeof(int)*NCIRCLES*NPOINTS, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_tY, sizeof(int)*NCIRCLES*NPOINTS);
  cudaMemcpyAsync(d_tY, host_tY, sizeof(int)*NCIRCLES*NPOINTS, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_grad_x, sizeof(float)*grad_m*grad_n);
  cudaMemcpyAsync(d_grad_x, host_grad_x, sizeof(float)*grad_m*grad_n, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_grad_y, sizeof(float)*grad_m*grad_n);
  cudaMemcpyAsync(d_grad_y, host_grad_y, sizeof(float)*grad_m*grad_n, cudaMemcpyHostToDevice, 0);

  cudaMalloc((void**)&d_gicov, sizeof(float)*grad_m*grad_n);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  kernel_GICOV<<<global_work_size/work_group_size, work_group_size>>>(
    d_grad_x,
    d_grad_y,
    d_sin_angle,
    d_cos_angle,
    d_tX,
    d_tY,
    d_gicov,
    local_work_size,
    num_work_groups,
    grad_m);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (GICOV): %f (s)\n", time * 1e-9f);

  cudaMemcpy(host_gicov, d_gicov, sizeof(float)*grad_m*grad_n, cudaMemcpyDeviceToHost);

  long long GICOV_end_time = get_time();
  
  // Copy the results into a new host matrix
#ifdef DEBUG
  printf("grad_m=%d grad_n=%d\n", grad_m, grad_n);
#endif
  MAT *gicov = m_get(grad_m, grad_n);
  for (int m = 0; m < grad_m; m++)
    for (int n = 0; n < grad_n; n++) {
#ifdef DEBUG
      printf("host_gicov: %f\n", host_gicov[(n * grad_m) + m]);
#endif
      m_set_val(gicov, m, n, host_gicov[(n * grad_m) + m]);
    }

  // Dilate the GICOV matrices
  long long dilate_start_time = get_time();
  // Determine the dimensions of the frame
  int max_gicov_m = gicov->m;
  int max_gicov_n = gicov->n;

  // Determine the dimensions of the structuring element
  int strel_m = 12 * 2 + 1;
  int strel_n = 12 * 2 + 1;

  float* d_strel;
  cudaMalloc((void**)&d_strel, sizeof(float)*strel_m*strel_n);
  cudaMemcpyAsync(d_strel, host_strel, sizeof(float)*strel_m*strel_n, cudaMemcpyHostToDevice, 0);
  float *host_dilated = (float *) malloc(sizeof(float)*max_gicov_m * max_gicov_n);

  float* d_img_dilated;
  cudaMalloc((void**)&d_img_dilated, sizeof(float)*max_gicov_m*max_gicov_n);

  // Setup execution parameters
  global_work_size = max_gicov_m * max_gicov_n;
  local_work_size = 176;
  // Make sure the global work size is a multiple of the local work size
  if (global_work_size % local_work_size != 0) {
    global_work_size = ((global_work_size / local_work_size) + 1) * local_work_size;
  }
#ifdef DEBUG
  printf("image dilate: local_work_size = %zu, global_work_size = %zu \n", local_work_size, global_work_size);
#endif

  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  kernel_dilated<<<global_work_size/local_work_size, local_work_size>>>(
	  d_strel, d_gicov, d_img_dilated, strel_m, strel_n, max_gicov_m, max_gicov_n);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (dilated): %f (s)\n", time * 1e-9f);

  cudaMemcpy(host_dilated, d_img_dilated, sizeof(float)*grad_m*grad_n, cudaMemcpyDeviceToHost);

  long long dilate_end_time = get_time();

  // Copy results into a new host matrix
  MAT *img_dilated = m_get(max_gicov_m, max_gicov_n);
  for (int m = 0; m < max_gicov_m; m++)
    for (int n = 0; n < max_gicov_n; n++) {
      m_set_val(img_dilated, m, n, host_dilated[(m * max_gicov_n) + n]);
#ifdef DEBUG
      printf("host_img_dilated: %f\n", host_dilated[(m * max_gicov_n) + n]);
#endif
    }

  // Find possible matches for cell centers based on GICOV and record the rows/columns in which they are found
  pair_counter = 0;
  crow = (int *) malloc(gicov->m * gicov->n * sizeof(int));
  ccol = (int *) malloc(gicov->m * gicov->n * sizeof(int));
  for(i = 0; i < gicov->m; i++) {
    for(j = 0; j < gicov->n; j++) {
      if(!double_eq(m_get_val(gicov,i,j), 0.0) && 
          double_eq(m_get_val(img_dilated,i,j), 
            m_get_val(gicov,i,j)))
      {
        crow[pair_counter]=i;
        ccol[pair_counter]=j;
        pair_counter++;
      }
    }
  }

  GICOV_spots = (double *) malloc(sizeof(double) * pair_counter);
  for(i = 0; i < pair_counter; i++)
    GICOV_spots[i] = sqrt(m_get_val(gicov, crow[i], ccol[i]));

  G = (double *) calloc(pair_counter, sizeof(double));
  x_result = (double *) calloc(pair_counter, sizeof(double));
  y_result = (double *) calloc(pair_counter, sizeof(double));

  x_result_len = 0;
  for (i = 0; i < pair_counter; i++) {
    if ((crow[i] > 29) && (crow[i] < BOTTOM - TOP + 39)) {
      x_result[x_result_len] = ccol[i];
      y_result[x_result_len] = crow[i] - 40;
      G[x_result_len] = GICOV_spots[i];
      x_result_len++;
    }
  }

  // Make an array t which holds each "time step" for the possible cells
  t = (double *) malloc(sizeof(double) * 36);
  for (i = 0; i < 36; i++) {
    t[i] = (double)i * 2.0 * PI / 36.0;
  }

  // Store cell boundaries (as simple circles) for all cells
  cellx = m_get(x_result_len, 36);
  celly = m_get(x_result_len, 36);
  for(i = 0; i < x_result_len; i++) {
    for(j = 0; j < 36; j++) {
      m_set_val(cellx, i, j, x_result[i] + radius * cos(t[j]));
      m_set_val(celly, i, j, y_result[i] + radius * sin(t[j]));
    }
  }

  A = TMatrix(9,4);
  V = (double *) malloc(sizeof(double) * pair_counter);
  QAX_CENTERS = (double * )malloc(sizeof(double) * pair_counter);
  QAY_CENTERS = (double *) malloc(sizeof(double) * pair_counter);
  memset(V, 0, sizeof(double) * pair_counter);
  memset(QAX_CENTERS, 0, sizeof(double) * pair_counter);
  memset(QAY_CENTERS, 0, sizeof(double) * pair_counter);

  // For all possible results, find the ones that are feasibly leukocytes and store their centers
  k_count = 0;
  for (n = 0; n < x_result_len; n++) {
    if ((G[n] < -1 * threshold) || G[n] > threshold) {
      MAT * x, *y;
      VEC * x_row, * y_row;
      x = m_get(1, 36);
      y = m_get(1, 36);

      x_row = v_get(36);
      y_row = v_get(36);

      // Get current values of possible cells from cellx/celly matrices
      x_row = get_row(cellx, n, x_row);
      y_row = get_row(celly, n, y_row);
      uniformseg(x_row, y_row, x, y);

      // Make sure that the possible leukocytes are not too close to the edge of the frame
      if ((m_min(x) > b) && (m_min(y) > b) && (m_max(x) < cell_file->width - b) && (m_max(y) < cell_file->height - b)) {
        MAT * Cx, * Cy, *Cy_temp, * Ix1, * Iy1;
        VEC  *Xs, *Ys, *W, *Nx, *Ny, *X, *Y;
        Cx = m_get(1, 36);
        Cy = m_get(1, 36);
        Cx = mmtr_mlt(A, x, Cx);
        Cy = mmtr_mlt(A, y, Cy);

        Cy_temp = m_get(Cy->m, Cy->n);

        for (i = 0; i < 9; i++)
          m_set_val(Cy, i, 0, m_get_val(Cy, i, 0) + 40.0);

        // Iteratively refine the snake/spline
        for (i = 0; i < Iter; i++) {
          int typeofcell;

          if(G[n] > 0.0) typeofcell = 0;
          else typeofcell = 1;

          splineenergyform01(Cx, Cy, grad_x, grad_y, ns, delta, 2.0 * dt, typeofcell);
        }

        X = getsampling(Cx, ns);
        for (i = 0; i < Cy->m; i++)
          m_set_val(Cy_temp, i, 0, m_get_val(Cy, i, 0) - 40.0);
        Y = getsampling(Cy_temp, ns);

        Ix1 = linear_interp2(grad_x, X, Y);
        Iy1 = linear_interp2(grad_x, X, Y);
        Xs = getfdriv(Cx, ns);
        Ys = getfdriv(Cy, ns);

        Nx = v_get(Ys->dim);
        for (i = 0; i < Ys->dim; i++)
          v_set_val(Nx, i, v_get_val(Ys, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

        Ny = v_get(Xs->dim);
        for (i = 0; i < Xs->dim; i++)
          v_set_val(Ny, i, -1.0 * v_get_val(Xs, i) / sqrt(v_get_val(Xs, i)*v_get_val(Xs, i) + v_get_val(Ys, i)*v_get_val(Ys, i)));

        W = v_get(Nx->dim);
        for (i = 0; i < Nx->dim; i++)
          v_set_val(W, i, m_get_val(Ix1, 0, i) * v_get_val(Nx, i) + m_get_val(Iy1, 0, i) * v_get_val(Ny, i));

        V[n] = mean(W) / std_dev(W);

        // Find the cell centers by computing the means of X and Y values for all snaxels of the spline contour
        QAX_CENTERS[k_count] = mean(X);
        QAY_CENTERS[k_count] = mean(Y) + TOP;

        k_count++;

        // Free memory
        v_free(W);
        v_free(Ny);
        v_free(Nx);
        v_free(Ys);
        v_free(Xs);
        m_free(Iy1);
        m_free(Ix1);
        v_free(Y);
        v_free(X);
        m_free(Cy_temp);
        m_free(Cy);
        m_free(Cx);        
      }

      // Free memory
      v_free(y_row);
      v_free(x_row);
      m_free(y);
      m_free(x);
    }
  }

  // Free memory
  cudaFree(d_sin_angle);
  cudaFree(d_cos_angle);
  cudaFree(d_tX);
  cudaFree(d_tY);
  cudaFree(d_grad_x);
  cudaFree(d_grad_y);
  cudaFree(d_gicov);
  cudaFree(d_strel);
  cudaFree(d_img_dilated);

  free(host_grad_x);
  free(host_grad_y);
  free(host_gicov);
  free(host_dilated);

  free(V);
  free(ccol);
  free(crow);
  free(GICOV_spots);
  free(t);
  free(G);
  free(x_result);
  free(y_result);
  m_free(A);
  m_free(celly);
  m_free(cellx);
  m_free(img_dilated);
  m_free(gicov);
  m_free(grad_y);
  m_free(grad_x);

  // Report the total number of cells detected
  printf("Cells detected: %d\n\n", k_count);

  // Report the breakdown of the detection runtime
  printf("Detection runtime\n");
  printf("-----------------\n");
  printf("GICOV computation: %.5f seconds\n", ((float) (GICOV_end_time - GICOV_start_time)) / (1000*1000));
  printf("   GICOV dilation: %.5f seconds\n", ((float) (dilate_end_time - dilate_start_time)) / (1000*1000));
  printf("            Total: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));

  // Now that the cells have been detected in the first frame,
  //  track the ellipses through subsequent frames
  if (num_frames > 1) printf("\nTracking cells across %d frames\n", num_frames);
  else                printf("\nTracking cells across 1 frame\n");


  long long tracking_start_time = get_time();
  int num_snaxels = 20;
  ellipsetrack(cell_file, QAX_CENTERS, QAY_CENTERS, k_count, radius, num_snaxels, num_frames);
  printf("           Total: %.5f seconds\n", ((float) (get_time() - tracking_start_time)) / (float) (1000*1000*num_frames));  

  // Report total program execution time
  printf("\nTotal application run time: %.5f seconds\n", ((float) (get_time() - program_start_time)) / (1000*1000));

  return 0;
}
