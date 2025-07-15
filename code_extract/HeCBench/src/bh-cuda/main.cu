/*
   ECL-BH v4.5: Simulation of the gravitational forces in a star cluster using
   the Barnes-Hut n-body algorithm.

   Copyright (c) 2010-2020 Texas State University. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher and Sahar Azimi

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/ECL-BH/.

Publication: This work is described in detail in the following paper.
Martin Burtscher and Keshav Pingali. An Efficient CUDA Implementation of the
Tree-based Barnes Hut n-Body Algorithm. Chapter 6 in GPU Computing Gems
Emerald Edition, pp. 75-92. January 2011.
 */


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

// threads per block
#define THREADS1 256  // must be a power of 2
#define THREADS2 256
#define THREADS3 256 
#define THREADS4 256
#define THREADS5 256
#define THREADS6 256

// block count = factor * #SMs
#define FACTOR1 2
#define FACTOR2 2
#define FACTOR3 1  // must all be resident at the same time
#define FACTOR4 1  // must all be resident at the same time
#define FACTOR5 2
#define FACTOR6 2

#define WARPSIZE 32
#define MAXDEPTH 32

//
// compute center and radius
//
__global__
void BoundingBoxKernel(
    const int nnodesd,
    const int nbodiesd,
    int* const __restrict__ startd,
    int* const __restrict__ childd,
    float4* const __restrict__ posMassd,
    float3* const __restrict__ maxd,
    float3* const __restrict__ mind,
    float* const __restrict__ radiusd,
    int* const __restrict__ bottomd,
    int* const __restrict__ stepd,
    unsigned int* const __restrict__ blkcntd)
{
  int i, j, k;
  float val;
  float3 min, max;
  __shared__ float sminx[THREADS1], 
                   smaxx[THREADS1], 
                   sminy[THREADS1],
                   smaxy[THREADS1],
                   sminz[THREADS1],
                   smaxz[THREADS1];

  // initialize with valid data (in case #bodies < #threads)
  const float4 p0 = posMassd[0];
  min.x = max.x = p0.x;
  min.y = max.y = p0.y;
  min.z = max.z = p0.z;

  // scan all bodies
  i = threadIdx.x;
  int inc = THREADS1 * gridDim.x;
  for (j = i + blockIdx.x * THREADS1; j < nbodiesd; j += inc) {
    const float4 p = posMassd[j];
    val = p.x;
    min.x = fminf(min.x, val);
    max.x = fmaxf(max.x, val);
    val = p.y;
    min.y = fminf(min.y, val);
    max.y = fmaxf(max.y, val);
    val = p.z;
    min.z = fminf(min.z, val);
    max.z = fmaxf(max.z, val);
  }

  // reduction in shared memory
  sminx[i] = min.x;
  smaxx[i] = max.x;
  sminy[i] = min.y;
  smaxy[i] = max.y;
  sminz[i] = min.z;
  smaxz[i] = max.z;

  for (j = THREADS1 / 2; j > 0; j /= 2) {
    __syncthreads();
    if (i < j) {
      k = i + j;
      sminx[i] = min.x = fminf(min.x, sminx[k]);
      smaxx[i] = max.x = fmaxf(max.x, smaxx[k]);
      sminy[i] = min.y = fminf(min.y, sminy[k]);
      smaxy[i] = max.y = fmaxf(max.y, smaxy[k]);
      sminz[i] = min.z = fminf(min.z, sminz[k]);
      smaxz[i] = max.z = fmaxf(max.z, smaxz[k]);
    }
  }

  // write block result to global memory
  if (i == 0) {
    k = blockIdx.x;
    mind[k] = min;
    maxd[k] = max;
    __threadfence();

    inc = gridDim.x - 1;
    if (inc == atomicInc(blkcntd, inc)) {
      // I'm the last block, so combine all block results
      for (j = 0; j <= inc; j++) {
        float3 minp = mind[j];
        float3 maxp = maxd[j];
        min.x = fminf(min.x, minp.x);
        max.x = fmaxf(max.x, maxp.x);
        min.y = fminf(min.y, minp.y);
        max.y = fmaxf(max.y, maxp.y);
        min.z = fminf(min.z, minp.z);
        max.z = fmaxf(max.z, maxp.z);
      }

      // compute radius
      val = fmaxf(max.x - min.x, max.y - min.y);
      *radiusd = fmaxf(val, max.z - min.z) * 0.5f;

      // create root node
      k = nnodesd;
      *bottomd = k;

      startd[k] = 0;
      float4 p;
      p.x = (min.x + max.x) * 0.5f;
      p.y = (min.y + max.y) * 0.5f;
      p.z = (min.z + max.z) * 0.5f;
      p.w = -1.0f;
      posMassd[k] = p;
      k *= 8;
      for (i = 0; i < 8; i++) childd[k + i] = -1;
      (*stepd)++;
    }
  }
}


//
// build tree
//
__global__
void ClearKernel1(const int nnodesd, const int nbodiesd, int* const __restrict__ childd)
{
  int top = 8 * nnodesd;
  int bottom = 8 * nbodiesd;
  int inc = blockDim.x * gridDim.x;
  int k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < top) {
    childd[k] = -1;
    k += inc;
  }
}


__global__
void TreeBuildingKernel(
    const int nnodesd,
    const int nbodiesd,
    volatile int* const __restrict__ childd,
    const float4* const __restrict__ posMassd,
    const float* const __restrict radiusd,
            int* const __restrict bottomd
)
{
  int i, j, depth, skip, inc;
  float x, y, z, r;
  float dx, dy, dz;
  int ch, n, cell, locked, patch;
  float radius;

  // cache root data
  radius = *radiusd * 0.5f;
  const float4 root = posMassd[nnodesd];

  skip = 1;
  inc = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
    const float4 p = posMassd[i];
    if (skip != 0) {
      // new body, so start traversing at root
      skip = 0;
      n = nnodesd;
      depth = 1;
      r = radius;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (root.x < p.x) {j = 1; dx = r;}
      if (root.y < p.y) {j |= 2; dy = r;}
      if (root.z < p.z) {j |= 4; dz = r;}
      x = root.x + dx;
      y = root.y + dy;
      z = root.z + dz;
    }

    // follow path to leaf cell
    ch = childd[n*8+j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (x < p.x) {j = 1; dx = r;}
      if (y < p.y) {j |= 2; dy = r;}
      if (z < p.z) {j |= 4; dz = r;}
      x += dx;
      y += dy;
      z += dz;
      ch = childd[n*8+j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*8+j;
      if (ch == -1) {
        if (ch == atomicCAS((int*)&childd[locked], ch, i)) {  // if null, just insert the new body
          i += inc;  // move on to next body
          skip = 1;
        }
      } else {  // there already is a body at this position
        if (ch == atomicCAS((int*)&childd[locked], ch, -2)) {  // try to lock
          patch = -1;
          const float4 chp = posMassd[ch];
          // create new cell(s) and insert the old and new bodies
          do {
            depth++;
            cell = atomicSub(bottomd, 1) - 1;

            if (patch != -1) {
              childd[n*8+j] = cell;
            }
            patch = max(patch, cell);

            j = 0;
            if (x < chp.x) j = 1;
            if (y < chp.y) j |= 2;
            if (z < chp.z) j |= 4;
            childd[cell*8+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            if (x < p.x) {j = 1; dx = r;}
            if (y < p.y) {j |= 2; dy = r;}
            if (z < p.z) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n*8+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*8+j] = i;

          i += inc;  // move on to next body
          skip = 2;
        }
      }
    }
    __syncthreads();  // optional barrier for performance

    if (skip == 2) {
      childd[locked] = patch;
    }
  }
}


__global__
void ClearKernel2(
    const int nnodesd, 
    int* const __restrict__ startd, 
    float4* const __restrict__ posMassd,
    int* const __restrict__ bottomd)
{
  int k, inc, bottom;

  bottom = *bottomd;
  inc = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    posMassd[k].w = -1.0f;
    startd[k] = -1;
    k += inc;
  }
}


//
// compute center of mass
//

__global__
void SummarizationKernel(
    const int nnodesd, 
    const int nbodiesd,
    volatile int* const __restrict__ countd,
    const int* const __restrict__ childd,
    volatile float4* const __restrict__ posMassd, // will cause hanging for 2048 bodies without volatile
    int* const __restrict bottomd)
{
  __shared__ int child[THREADS3 * 8];
  __shared__ float mass[THREADS3 * 8];

  int i, j, ch, cnt;
  float cm, px, py, pz, m;
  int bottom = *bottomd;
  int inc = blockDim.x * gridDim.x;
  int k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += inc;

  int restart = k;
  for (j = 0; j < 3; j++) {  // wait-free pre-passes
    // iterate over all cells assigned to thread
    while (k <= nnodesd) {
      if (posMassd[k].w < 0.0f) {
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch >= nbodiesd) && ((mass[i*THREADS3+threadIdx.x] = posMassd[ch].w) < 0.0f)) {
            break;
          }
        }
        if (i == 8) {
          // all children are ready
          cm = 0.0f;
          px = 0.0f;
          py = 0.0f;
          pz = 0.0f;
          cnt = 0;
          for (i = 0; i < 8; i++) {
            ch = child[i*THREADS3+threadIdx.x];
            if (ch >= 0) {
              const float chx = posMassd[ch].x;
              const float chy = posMassd[ch].y;
              const float chz = posMassd[ch].z;
              const float chw = posMassd[ch].w;
              if (ch >= nbodiesd) {  // count bodies (needed later)
                m = mass[i*THREADS3+threadIdx.x];
                cnt += countd[ch];
              } else {
                m = chw;
                cnt++;
              }
              // add child's contribution
              cm += m;
              px += chx * m;
              py += chy * m;
              pz += chz * m;
            }
          }
          countd[k] = cnt;
          m = 1.0f / cm;
          posMassd[k].x = px * m;
          posMassd[k].y = py * m;
          posMassd[k].z = pz * m;
          posMassd[k].w = cm;
        }
      }
      k += inc;  // move on to next cell
    }
    k = restart;
  }

  j = 0;
  // iterate over all cells assigned to thread
  while (k <= nnodesd) {
    if (posMassd[k].w >= 0.0f) {
      k += inc;
    } else {
      if (j == 0) {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = childd[k*8+i];
          child[i*THREADS3+threadIdx.x] = ch;  // cache children
          if ((ch < nbodiesd) || ((mass[i*THREADS3+threadIdx.x] = posMassd[ch].w) >= 0.0f)) {
            j--;
          }
        }
      } else {
        j = 8;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if ((ch < nbodiesd) || (mass[i*THREADS3+threadIdx.x] >= 0.0f) || ((mass[i*THREADS3+threadIdx.x] = posMassd[ch].w) >= 0.0f)) {
            j--;
          }
        }
      }

      if (j == 0) {
        // all children are ready
        cm = 0.0f;
        px = 0.0f;
        py = 0.0f;
        pz = 0.0f;
        cnt = 0;
        for (i = 0; i < 8; i++) {
          ch = child[i*THREADS3+threadIdx.x];
          if (ch >= 0) {
            // four reads due to missing copy constructor for "volatile float4"
            const float chx = posMassd[ch].x;
            const float chy = posMassd[ch].y;
            const float chz = posMassd[ch].z;
            const float chw = posMassd[ch].w;
            if (ch >= nbodiesd) {  // count bodies (needed later)
              m = mass[i*THREADS3+threadIdx.x];
              cnt += countd[ch];
            } else {
              m = chw;
              cnt++;
            }
            // add child's contribution
            cm += m;
            px += chx * m;
            py += chy * m;
            pz += chz * m;
          }
        }
        countd[k] = cnt;
        m = 1.0f / cm;
        // four writes due to missing copy constructor for "volatile float4"
        posMassd[k].x = px * m;
        posMassd[k].y = py * m;
        posMassd[k].z = pz * m;
        posMassd[k].w = cm;
        k += inc;
      }
    }
  }
}


//
// sort bodies
//
__global__
void SortKernel(
    const int nnodesd,
    const int nbodiesd, 
    int* const __restrict__ sortd,
    const int* const __restrict__ countd,
    volatile int* const __restrict__ startd,
    int* const __restrict__ childd,
    int* const __restrict__ bottomd)
{
  int i, j;
  int bottom = *bottomd;
  int dec = blockDim.x * gridDim.x;
  int k = nnodesd + 1 - dec + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= bottom) {
    int start = startd[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 8; i++) {
        int ch = childd[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front (needed later for speed)
            childd[k*8+i] = -1;
            childd[k*8+j] = ch;
          }
          j++;
          if (ch >= nbodiesd) {
            // child is a cell
            startd[ch] = start;  // set start ID of child
            start += countd[ch];  // add #bodies in subtree
          } else {
            // child is a body
            sortd[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= dec;  // move on to next cell
    }
    __syncthreads();  // optional barrier for performance
  }
}


//
// compute force
//
__global__
void ForceCalculationKernel(
    const int nnodesd, 
    const int nbodiesd,
    const float dthfd,
    const float itolsqd,
    const float epssqd,
    const int* const __restrict__ sortd,
    const int* const __restrict__ childd,
    const float4* const __restrict__ posMassd,
    float2* const __restrict__ veld,
    float4* const __restrict__ accVeld,
    const float* const __restrict__ radiusd,
    const int* const __restrict__ stepd)
{
  int i, j, k, n, depth, base, sbase, diff, pd, nd;
  float ax, ay, az, dx, dy, dz, tmp;
  __shared__ int pos[THREADS5], node[THREADS5];
  __shared__ float dq[THREADS5];

  if (0 == threadIdx.x) {
    tmp = *radiusd * 2;
    // precompute values that depend only on tree level
    dq[0] = tmp * tmp * itolsqd;
    for (i = 1; i < MAXDEPTH; i++) {
      dq[i] = dq[i - 1] * 0.25f;
      dq[i - 1] += epssqd;
    }
    dq[i - 1] += epssqd;
  }
  __syncthreads();

  // figure out first thread in each warp (lane 0)
  base = threadIdx.x / WARPSIZE;
  sbase = base * WARPSIZE;
  j = base * MAXDEPTH;

  diff = threadIdx.x - sbase;
  // make multiple copies to avoid index calculations later
  if (diff < MAXDEPTH) {
    dq[diff+j] = dq[diff];
  }
  __syncthreads();

  // iterate over all bodies assigned to thread
  for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodiesd; k += blockDim.x * gridDim.x) {
    i = sortd[k];  // get permuted/sorted index
    // cache position info
    const float4 pi = posMassd[i];

    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;

    // initialize iteration stack, i.e., push root node onto stack
    depth = j;
    if (sbase == threadIdx.x) {
      pos[j] = 0;
      node[j] = nnodesd * 8;
    }

    do {
      // stack is not empty
      pd = pos[depth];
      nd = node[depth];
      while (pd < 8) {
        // node on top of stack has more children to process
        n = childd[nd + pd];  // load child pointer
        pd++;

        if (n >= 0) {
          const float4 pn = posMassd[n];
          dx = pn.x - pi.x;
          dy = pn.y - pi.y;
          dz = pn.z - pi.z;
          tmp = dx*dx + (dy*dy + (dz*dz + epssqd));  // compute distance squared (plus softening)
          if ((n < nbodiesd) || __all_sync(0xffffffff, tmp >= dq[depth])) {  
          // check if all threads agree that cell is far enough away (or is a body)
            tmp = rsqrtf(tmp);  // compute distance
            tmp = pn.w * tmp * tmp * tmp;
            ax += dx * tmp;
            ay += dy * tmp;
            az += dz * tmp;
          } else {
            // push cell onto stack
            if (sbase == threadIdx.x) {
              pos[depth] = pd;
              node[depth] = nd;
            }
            depth++;
            pd = 0;
            nd = n * 8;
          }
        } else {
          pd = 8;  // early out because all remaining children are also zero
        }
      }
      depth--;  // done with this level
    } while (depth >= j);

    float4 acc = accVeld[i];
    if (*stepd > 0) {
      // update velocity
      float2 v = veld[i];
      v.x += (ax - acc.x) * dthfd;
      v.y += (ay - acc.y) * dthfd;
      acc.w += (az - acc.z) * dthfd;
      veld[i] = v;
    }

    // save computed acceleration
    acc.x = ax;
    acc.y = ay;
    acc.z = az;
    accVeld[i] = acc;
  }
}


//
// advance bodies
//
__global__
void IntegrationKernel(
     const int nbodiesd,
     const float dtimed,
     const float dthfd,
     float4* const __restrict__ posMass,
     float2* const __restrict__ veld,
     float4* const __restrict__ accVeld)
{
  int i, inc;
  float dvelx, dvely, dvelz;
  float velhx, velhy, velhz;

  // iterate over all bodies assigned to thread
  inc = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < nbodiesd; i += inc) {
    // integrate
    float4 acc = accVeld[i];
    dvelx = acc.x * dthfd;
    dvely = acc.y * dthfd;
    dvelz = acc.z * dthfd;

    float2 v = veld[i];
    velhx = v.x + dvelx;
    velhy = v.y + dvely;
    velhz = acc.w + dvelz;

    float4 p = posMass[i];
    p.x += velhx * dtimed;
    p.y += velhy * dtimed;
    p.z += velhz * dtimed;
    posMass[i] = p;

    v.x = velhx + dvelx;
    v.y = velhy + dvely;
    acc.w = velhz + dvelz;
    veld[i] = v;
    accVeld[i] = acc;
  }
}

__global__
void InitializationKernel(int *step, unsigned int *blkcnt)
{
  *step = -1;
  *blkcnt = 0;
}


// random number generator (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/util.C)

static int randx = 7;
static double drnd()
{
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}


int main(int argc, char* argv[])
{
  // perform some checks

  printf("ECL-BH v4.5\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "\n");
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }

  const int nbodies = atoi(argv[1]);
  if (nbodies < 1) {
    fprintf(stderr, "nbodies is too small: %d\n", nbodies);
    exit(-1);
  }

  if (nbodies > (1 << 30)) {
    fprintf(stderr, "nbodies is too large: %d\n", nbodies);
    exit(-1);
  }

  int timesteps = atoi(argv[2]);
  if (timesteps < 0) {
    fprintf(stderr, "the number of steps should be positive: %d\n", timesteps);
    exit(-1);
  }

  int i;
  int nnodes, step;
  double runtime;
  float dtime, dthf, epssq, itolsq;

  float4 *accVel;
  float2 *vel;
  int *d_sort, *d_child, *d_count, *d_start;
  int *d_step, *d_bottom;
  unsigned int *d_blkcnt;
  float *d_radius;
  float4 *d_accVel;
  float2 *d_vel;
  float3 *d_max, *d_min;
  float4 *d_posMass;
  float4 *posMass;
  double rsc, vsc, r, v, x, y, z, sq, scale;

  // the number of thread blocks may be adjusted for higher performance
  const int blocks = 24;

  nnodes = nbodies * 2;
  if (nnodes < 1024*blocks) nnodes = 1024*blocks;
  while ((nnodes & (WARPSIZE-1)) != 0) nnodes++;
  nnodes--;

  dtime = 0.025f; 
  dthf = dtime * 0.5f;
  epssq = 0.05 * 0.05;
  itolsq = 1.0f / (0.5 * 0.5);

  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  // allocate host memory

  accVel = (float4*)malloc(sizeof(float4) * nbodies);
  if (accVel == NULL) fprintf(stderr, "cannot allocate accVel\n");

  vel = (float2*)malloc(sizeof(float2) * nbodies);
  if (vel == NULL) fprintf(stderr, "cannot allocate vel\n");

  posMass = (float4*)malloc(sizeof(float4) * nbodies);
  if (posMass == NULL) fprintf(stderr, "cannot allocate posMass\n");

  // initialize host memory (https://github.com/staceyson/splash2/blob/master/codes/apps/barnes/code.C)

  rsc = (3 * 3.1415926535897932384626433832795) / 16;
  vsc = sqrt(1.0 / rsc);
  for (i = 0; i < nbodies; i++) {
    float4 p;
    p.w = 1.f / nbodies;
    r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = rsc * r / sqrt(sq);
    p.x = x * scale;
    p.y = y * scale;
    p.z = z * scale;
    posMass[i] = p;

    do {
      x = drnd();
      y = drnd() * 0.1;
    } while (y > x*x * pow(1 - x*x, 3.5));
    v = x * sqrt(2.0 / sqrt(1 + r*r));
    do {
      x = drnd()*2.0 - 1.0;
      y = drnd()*2.0 - 1.0;
      z = drnd()*2.0 - 1.0;
      sq = x*x + y*y + z*z;
    } while (sq > 1.0);
    scale = vsc * v / sqrt(sq);
    float2 v;
    v.x = x * scale;
    v.y = y * scale;
    accVel[i].w = z * scale;
    vel[i] = v;
  }

  // allocate device memory

  if (cudaSuccess != cudaMalloc((void **)&d_child, sizeof(int) * (nnodes+1) * 8))
    fprintf(stderr, "could not allocate d_child\n");

  if (cudaSuccess != cudaMalloc((void **)&d_vel, sizeof(float2) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_vel\n");

  if (cudaSuccess != cudaMalloc((void **)&d_accVel, sizeof(float4) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_accVel\n");

  if (cudaSuccess != cudaMalloc((void **)&d_count, sizeof(int) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_count\n");

  if (cudaSuccess != cudaMalloc((void **)&d_start, sizeof(int) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_start\n");

  if (cudaSuccess != cudaMalloc((void **)&d_sort, sizeof(int) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_sort\n");

  if (cudaSuccess != cudaMalloc((void **)&d_posMass, sizeof(float4) * (nnodes+1)))
    fprintf(stderr, "could not allocate d_posMass\n");

  if (cudaSuccess != cudaMalloc((void **)&d_max, sizeof(float3) * blocks * FACTOR1))
    fprintf(stderr, "could not allocate d_max\n");

  if (cudaSuccess != cudaMalloc((void **)&d_min, sizeof(float3) * blocks * FACTOR1))
    fprintf(stderr, "could not allocate d_min\n");

  if (cudaSuccess != cudaMalloc((void **)&d_step, sizeof(int)))
    fprintf(stderr, "could not allocate d_step\n");

  if (cudaSuccess != cudaMalloc((void **)&d_bottom, sizeof(int)))
    fprintf(stderr, "could not allocate d_bottom\n");

  if (cudaSuccess != cudaMalloc((void **)&d_blkcnt, sizeof(unsigned int)))
    fprintf(stderr, "could not allocate d_blkcnt\n");

  if (cudaSuccess != cudaMalloc((void **)&d_radius, sizeof(float)))
    fprintf(stderr, "could not allocate d_radius\n");


  if (cudaSuccess != cudaMemcpy(d_accVel, accVel, sizeof(float4) * nbodies, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of vel to device failed\n");

  if (cudaSuccess != cudaMemcpy(d_vel, vel, sizeof(float2) * nbodies, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of vel to device failed\n");

  if (cudaSuccess != cudaMemcpy(d_posMass, posMass, sizeof(float4) * nbodies, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of posMass to device failed\n");

  cudaDeviceSynchronize();

  struct timeval starttime, endtime;
  gettimeofday(&starttime, NULL);

  // run timesteps (launch kernels on a device)
  InitializationKernel<<<1, 1>>>(d_step, d_blkcnt);

  for (step = 0; step < timesteps; step++) {
    BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(
        nnodes, nbodies, d_start, d_child, d_posMass, d_max, d_min, 
        d_radius, d_bottom, d_step, d_blkcnt );

    ClearKernel1<<<blocks, 256>>>(nnodes, nbodies, d_child);

    TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(
        nnodes, nbodies, d_child, d_posMass, d_radius, d_bottom);

    ClearKernel2<<<blocks, 256>>>(nnodes, d_start, d_posMass, d_bottom);

    SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(
        nnodes, nbodies, d_count, d_child, d_posMass, d_bottom);

    SortKernel<<<blocks * FACTOR4, THREADS4>>>(
        nnodes, nbodies, d_sort, d_count, d_start, d_child, d_bottom);

    ForceCalculationKernel<<<blocks * FACTOR5, THREADS5>>>(
        nnodes, nbodies, dthf, itolsq, epssq, d_sort, d_child, d_posMass, 
        d_vel, d_accVel, d_radius, d_step);

    IntegrationKernel<<<blocks * FACTOR6, THREADS6>>>(
        nbodies, dtime, dthf, d_posMass, d_vel, d_accVel);
  }
  cudaDeviceSynchronize();

  gettimeofday(&endtime, NULL);
  runtime = (endtime.tv_sec + endtime.tv_usec/1000000.0 - 
             starttime.tv_sec - starttime.tv_usec/1000000.0);

  printf("Total kernel execution time: %.4lf s\n", runtime);

  // transfer final results back to a host
  if (cudaSuccess != cudaMemcpy(accVel, d_accVel, sizeof(float4) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of accVel from device failed\n");

  if (cudaSuccess != cudaMemcpy(vel, d_vel, sizeof(float2) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of vel from device failed\n");

  if (cudaSuccess != cudaMemcpy(posMass, d_posMass, sizeof(float4) * nbodies, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of posMass from device failed\n");

#ifdef DEBUG
  // print output for verification
  for (i = 0; i < nbodies; i++) {
    printf("%d: %.2e %.2e %.2e\n", i, posMass[i].x, posMass[i].y, posMass[i].z);
    printf("%d: %.2e %.2e %.2e %.2e\n", i, accVel[i].x, accVel[i].y, accVel[i].z, accVel[i].w);
    printf("%d: %.2e %.2e\n", i, vel[i].x, vel[i].y);
  }
#endif

  free(vel);
  free(accVel);
  free(posMass);

  cudaFree(d_child);
  cudaFree(d_vel);
  cudaFree(d_accVel);
  cudaFree(d_count);
  cudaFree(d_start);
  cudaFree(d_sort);
  cudaFree(d_posMass);
  cudaFree(d_max);
  cudaFree(d_min);
  cudaFree(d_step);
  cudaFree(d_blkcnt);
  cudaFree(d_bottom);
  cudaFree(d_radius);

  return 0;
}
