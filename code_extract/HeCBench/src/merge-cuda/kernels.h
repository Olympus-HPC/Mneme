/* POSITIVEINFINITY
 * Returns maximum value of a type
 */
__host__ __device__ float positiveInfinity(float tmp) {
  return FLT_MAX;
}

__host__ __device__ double positiveInfinity(double tmp) {
  return DBL_MAX;
}

__host__ __device__ uint32_t positiveInfinity(uint32_t tmp) {
  return 0xFFFFFFFFUL;
}

__host__ __device__ uint64_t positiveInfinity(uint64_t tmp) {
  return 0xFFFFFFFFFFFFFFFFUL;
}

template<typename vec_t>
__host__ __device__ vec_t getPositiveInfinity() {
  vec_t tmp = 0;
  return positiveInfinity(tmp);
}

/* NEGATIVEINFINITY
 * Returns minimum value of a type
 */
__host__ __device__ float negativeInfinity(float tmp) {
  return FLT_MIN;
}

__host__ __device__ double negativeInfinity(double tmp) {
  return DBL_MIN;
}

__host__ __device__ uint32_t negativeInfinity(uint32_t tmp) {
  return 0;
}

__host__ __device__ uint64_t negativeInfinity(uint64_t tmp) {
  return 0;
}

template<typename vec_t>
__host__ __device__ vec_t getNegativeInfinity() {
  vec_t tmp = 0;
  return negativeInfinity(tmp);
}

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X,Y) (((X) < (Y)) ? (X) : (Y))

/*
 * Performs a 32-wide binary search on one global diagonal per block to find the intersection with the path.
 * This divides the workload into independent merges for the next step 
 */
template<typename vec_t>
__global__ void workloadDiagonals(
  const vec_t *__restrict__ A, uint32_t A_length, 
  const vec_t *__restrict__ B, uint32_t B_length, 
  uint32_t *__restrict__ diagonal_path_intersections) {

  // Calculate combined index around the MergePath "matrix"
  int32_t combinedIndex = (uint64_t)blockIdx.x * ((uint64_t)A_length + (uint64_t)B_length) / (uint64_t)gridDim.x;
  volatile __shared__ int32_t x_top, y_top, x_bottom, y_bottom,  found;
  __shared__ int32_t oneorzero[32];

  int threadOffset = threadIdx.x - 16;

  // Figure out the coordinates of our diagonal
  x_top = MIN(combinedIndex, A_length);
  y_top = combinedIndex > (A_length) ? combinedIndex - (A_length) : 0;
  x_bottom = y_top;
  y_bottom = x_top;

  found = 0;

  // Search the diagonal
  while(!found) {
    // Update our coordinates within the 32-wide section of the diagonal 
    int32_t current_x = x_top - ((x_top - x_bottom) >> 1) - threadOffset;
    int32_t current_y = y_top + ((y_bottom - y_top) >> 1) + threadOffset;

    // Are we a '1' or '0' with respect to A[x] <= B[x]
    if(current_x >= (int32_t)A_length || current_y < 0) {
      oneorzero[threadIdx.x] = 0;
    } else if(current_y >= (int32_t)B_length || current_x < 1) {
      oneorzero[threadIdx.x] = 1;
    } else {
      oneorzero[threadIdx.x] = (A[current_x-1] <= B[current_y]) ? 1 : 0;
    }

    __syncthreads();

    // If we find the meeting of the '1's and '0's, we found the 
    // intersection of the path and diagonal
    if(threadIdx.x > 0 && (oneorzero[threadIdx.x] != oneorzero[threadIdx.x-1])) {
      found = 1;
      diagonal_path_intersections[blockIdx.x] = current_x;
      diagonal_path_intersections[blockIdx.x + gridDim.x + 1] = current_y;
    }

    __syncthreads();

    // Adjust the search window on the diagonal
    if(threadIdx.x == 16) {
      if(oneorzero[31] != 0) {
        x_bottom = current_x;
        y_bottom = current_y;
      } else {
        x_top = current_x;
        y_top = current_y;
      }
    }
    __syncthreads();
  }

  // Set the boundary diagonals (through 0,0 and A_length,B_length)
  if(threadIdx.x == 0 && blockIdx.x == 0) {
    diagonal_path_intersections[0] = 0;
    diagonal_path_intersections[gridDim.x + 1] = 0;
    diagonal_path_intersections[gridDim.x] = A_length;
    diagonal_path_intersections[gridDim.x + gridDim.x + 1] = B_length;
  }
}

/*
 * Performs merge windows within a thread block from that block's global diagonal 
 * intersection to the next 
 */
#define K 512
template<typename vec_t, bool timesections, bool countloops>
__global__ void mergeSinglePath(
    const vec_t * __restrict__ A, uint32_t A_length,
    const vec_t * __restrict__ B, uint32_t B_length, 
    const uint32_t * __restrict__ diagonal_path_intersections,
    vec_t * __restrict__ C, uint32_t C_length)
{

  // Storage space for local merge window
  __shared__ vec_t A_shared[(K+2) << 1];
  vec_t* B_shared = A_shared + K+2;

  volatile __shared__ uint32_t x_block_top, y_block_top, x_block_stop, y_block_stop;

  // Pre-calculate reused indices
  uint32_t threadIdX4 = threadIdx.x + threadIdx.x;
  threadIdX4 = threadIdX4 + threadIdX4;
  uint32_t threadIdX4p1 = threadIdX4 + 1;
  uint32_t threadIdX4p2 = threadIdX4p1 + 1;
  uint32_t threadIdX4p3 = threadIdX4p2 + 1;

  // Define global window and create sentinels
  switch(threadIdx.x) {
    case 0:
      x_block_top = diagonal_path_intersections[blockIdx.x];
      A_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 64:
      y_block_top = diagonal_path_intersections[blockIdx.x + gridDim.x + 1];
      A_shared[K+1] = getPositiveInfinity<vec_t>();
      break;
    case 32:
      x_block_stop = diagonal_path_intersections[blockIdx.x + 1];
      B_shared[0] = getNegativeInfinity<vec_t>();
      break;
    case 96:
      y_block_stop = diagonal_path_intersections[blockIdx.x + gridDim.x + 2];
      B_shared[K+1] = getPositiveInfinity<vec_t>();
      break;
    default:
      break;
  }

  A--;
  B--;

  __syncthreads();

  // Construct and merge windows from diagonal_path_intersections[blockIdx.x] 
  // to diagonal_path_intersections[blockIdx.x+1]
  while(((x_block_top < x_block_stop) || (y_block_top < y_block_stop))) {

    // Load current local window
    const vec_t * Atemp = A + x_block_top;
    const vec_t * Btemp = B + y_block_top;
    uint32_t sharedX = threadIdx.x+1;

    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    sharedX += blockDim.x;
    A_shared[sharedX] = Atemp[sharedX];
    B_shared[sharedX] = Btemp[sharedX];
    
    // Make sure this is before the sync
    vec_t *Ctemp = C + x_block_top + y_block_top;

    __syncthreads();

    // Binary search diagonal in the local window for path
    int32_t offset = threadIdX4 >> 1;
    uint32_t Ax = offset + 1;
    vec_t * BSm1 = B_shared + threadIdX4p2;
    vec_t * BS = BSm1 + 1;
    while(true) {
      offset = ((offset+1) >> 1);
      if(A_shared[Ax] > BSm1[~Ax]) {
        if(A_shared[Ax-1] <= BS[~Ax]) {
          //Found it
          break;
        }
        Ax -= offset;
      } else {
        Ax += offset;
      }
    }

    uint32_t Bx = threadIdX4p2 - Ax;

    // Merge four elements starting at the found path intersection
    vec_t Ai, Bi, Ci;
    Ai = A_shared[Ax];
    Bi = B_shared[Bx];
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4] = Ci;
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4p1] = Ci;
    if(Ai > Bi) {Ci = Bi; Bx++; Bi = B_shared[Bx];} else {Ci = Ai; Ax++; Ai = A_shared[Ax];}
    Ctemp[threadIdX4p2] = Ci;
    Ctemp[threadIdX4p3] = Ai > Bi ? Bi : Ai;

    // Update for next window
    if(threadIdx.x == 127) {
      x_block_top += Ax - 1;
      y_block_top += Bx - 1;
    }
    __syncthreads();
  } // Go to next window
}
