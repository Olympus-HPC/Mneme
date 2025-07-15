/****************************************************************************
 *
 * kernels.cu, Version 1.0.0 Mon 09 Jan 2012
 *
 * ----------------------------------------------------------------------------
 *
 * Copyright (C) 2012 CancerCare Manitoba
 *
 * The latest version of CUDA EGS and additional information are available online at 
 * http://www.physics.umanitoba.ca/~elbakri/cuda_egs/ and http://www.lippuner.ca/cuda_egs
 *
 * CUDA EGS is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.                                       
 *                                                                           
 * CUDA EGS is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.                              
 *                                                                           
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * ----------------------------------------------------------------------------
 *
 *   Contact:
 *
 *   Jonas Lippuner
 *   Email: jonas@lippuner.ca 
 *
 ****************************************************************************/

#ifdef CUDA_EGS
#define MASK 0xFFFFFFFF

/* * * * * * * * * * * * * * *
 * General Helper Functions  *
 * * * * * * * * * * * * * * */

// the maximum number of ULPs that two floats may differ and still be considered "almost equal"
#define MAX_ULPS 20

// check whether two floats are almost equal
// taken from http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
__device__ bool almostEqual(float A, float B) {
  // Make sure maxUlps is non-negative and small enough that the
  // default NAN won't compare as equal to anything.

  // Make aInt lexicographically ordered as a twos-complement int
  int aInt = *(int*)&A;
  if (aInt < 0)
    aInt = 0x80000000 - aInt;

  // Make bInt lexicographically ordered as a twos-complement int
  int bInt = *(int*)&B;
  if (bInt < 0)
    bInt = 0x80000000 - bInt;

  int intDiff = abs(aInt - bInt);
  if (intDiff <= MAX_ULPS)
    return true;

  return false;
}

// calculate indices of this thread
__device__ indices get_indices() {
  indices idx;
  // index of the block in the grid
  idx.b = blockIdx.y * gridDim.x + blockIdx.x;

  // index of the particle on the stack
  idx.p = idx.b * blockDim.x + threadIdx.x;

  // index of the warp in the block
  idx.w = threadIdx.x / WARP_SIZE;

  // index of the thread in the warp
  idx.t = threadIdx.x % WARP_SIZE;

  return idx;
}



/* * * * * * * * * * * * * * * * * * *
 * Random Number Generator Functions *
 * * * * * * * * * * * * * * * * * * */

/****************************************************************************
 * ALL THESE FUNCTIONS MUST *ALWAYS* BE CALLED BY *ALL* THREADS IN THE WARP *
 ****************************************************************************/

// update the array with random numbers for one warp
// This implements the Mersenne Twister for Graphic Processors (MTGP) and the
// code is largely based on the code available at http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MTGP/index.html
__device__ __noinline__ void MT_generate_array() {
  indices idx = get_indices();

  volatile uint *status = MT_statuses_shared[idx.w];
  volatile float *random_array = random_array_shared[idx.w];

  uint M = MT_params_shared[idx.w].M;
  uint mask = MT_params_shared[idx.w].mask;
  uint sh1 = MT_params_shared[idx.w].sh1;
  uint sh2 = MT_params_shared[idx.w].sh2;

  int first_bound = MT_N - M;
  if (first_bound > WARP_SIZE * MT_NUM_PER_THREAD)
    first_bound = WARP_SIZE * MT_NUM_PER_THREAD;

  // update first min(N - M, MT_NUM_PER_WARP) elements
  for (uint i = idx.t; i < first_bound; i += WARP_SIZE) {
    // recursion
    uint x = (status[i] & mask) ^ status[i + 1];
    x ^= x << sh1;
    x ^= status[i + M] >> sh2;
    x ^= MT_tables_shared[idx.w].recursion[x & 0x0FU];

    // temper output and fill random array
    uint t = status[i + M - 1];
    t ^= t >> 16;
    t ^= t >> 8;
    // set the last bit to 1 to get a float in the range (1,2) (excluding endpoints)
    t = ((x >> 9) ^ MT_tables_shared[idx.w].tempering[t & 0x0FU]) | 0x01U;
    random_array[i] = *((float*)&t) - 1.0F;

    // update status
    status[i] = x;
  }

  // update remaining elements
  for (int i = first_bound + idx.t; i < MT_N; i += WARP_SIZE) {
    // recursion
    uint x = (status[i] & mask) ^ status[i + 1 - (i + 1 >= MT_N ? MT_N : 0)];
    x ^= x << sh1;
    x ^= status[i + M - (i + M >= MT_N ? MT_N : 0)] >> sh2;
    x ^= MT_tables_shared[idx.w].recursion[x & 0x0FU];

    // temper output and fill random array
    if (i < WARP_SIZE * MT_NUM_PER_THREAD) {
      uint t = status[i + M - 1 - (i + M - 1 >= MT_N ? MT_N : 0)];
      t ^= t >> 16;
      t ^= t >> 8;
      t = ((x >> 9) ^ MT_tables_shared[idx.w].tempering[t & 0x0FU]) | 0x01U;
      random_array[i] = *((float*)&t) - 1.0F;
    }

    // update status
    status[i] = x;
  }
}

// read the status of the MT for this warp from global memory
__device__ void MT_read_status(indices idx) {
  uint MT_idx = idx.b * SIMULATION_WARPS_PER_BLOCK + idx.w;

  volatile uchar *rand_idx = rand_idx_shared;
  if (idx.t == 0) {
    MT_params_shared[idx.w] = MT_params[MT_idx];
    rand_idx[idx.w] = 0;
  }

  for (uint i = idx.t; i < MT_N; i += WARP_SIZE)
    MT_statuses_shared[idx.w][i] = MT_statuses[MT_idx * MT_NUM_STATUS + i];

  ((uint*)&(MT_tables_shared[idx.w]))[idx.t] = ((uint*)&(MT_tables[MT_idx]))[idx.t];

  MT_generate_array();
}

// write the status of the MT for this warp to global memory
__device__ void MT_write_status(indices idx) {
  uint MT_idx = idx.b * SIMULATION_WARPS_PER_BLOCK + idx.w;

  for (uint i = idx.t; i < MT_N; i += WARP_SIZE)
    MT_statuses[MT_idx * MT_NUM_STATUS + i] = MT_statuses_shared[idx.w][i];
}

// get the next random number
__device__ float get_rand(indices idx) {
  volatile uchar *rand_idx = rand_idx_shared;
  int i = rand_idx[idx.w] + 1;

  // all random numbers int he current array have been used, update the array
  if (i >= MT_NUM_PER_THREAD) {
    MT_generate_array();
    i = 0;
  }

  if (idx.t == 0)
    rand_idx[idx.w] = i;

  // return the random number for this thread
  return random_array_shared[idx.w][i * WARP_SIZE + idx.t];
}



/* * * * * * * * * * * *
 * Geometry Functions  *
 * * * * * * * * * * * */

// Determine the index i such that the point p lies between bounds[i] and bounds[i+1].
// Code was taken from the function isWhere of the class EGS_PlanesT in the file
// egs_planes.h (v 1.17 2009/07/06) and the function findRegion of the class EGS_BaseGeometry
// in the file egs_base_geometry.h (v 1.26 2008/09/22) of the EGSnrc C++ Class Library.
__device__ int isWhere(float p, uint nreg, float *bounds) {
  if ((p < bounds[0]) || (p > bounds[nreg]))
    return -1;
  if (nreg == 1)
    return 0;

  int ml = 0;
  int mu = nreg;
  while (mu - ml > 1) {
    int mav = (ml + mu) / 2;
    if (p <= bounds[mav])
      mu = mav; 
    else 
      ml = mav;
  }
  return  mu - 1;
}

// Determine the distance t to the next voxel boundary for the particle p and return
// the region index that the particle will enter.
// Code was taken from the function howfar of the class EGS_XYZGeometry in the file
// egs_nd_geometry.h (v 1.26 2009/07/06) of the EGSnrc C++ Class Library.
// Our region indices are shifted by 1 because our outside region is 0, while
// the outside region in the EGSnrc C++ Class Library is -1.
__device__ uint howfar(indices idx, particle_t &p, float &t) {
  if (p.region > 0) {
    // because of the above mentioned shift, we have to substract 1
    uint ir = p.region - 1;

    int iz = ir / (phantom.N.x * phantom.N.y); 
    ir -= iz * phantom.N.x * phantom.N.y; 
    int iy = ir / phantom.N.x;
    int ix = ir - iy * phantom.N.x;
    uint inew = p.region;

    if (p.u > 0.0F) {
      float d = (phantom.x_bounds[ix + 1] - p.x) / p.u;
      if (d <= t) { 
        t = d; 
        if (ix + 1 < phantom.N.x) 
          inew = p.region + 1; 
        else 
          inew = 0;
      }
    }
    else if (p.u < 0.0F) {
      float d = (phantom.x_bounds[ix] - p.x) / p.u;
      if (d <= t) { 
        t = d; 
        if (ix - 1 >= 0) 
          inew = p.region - 1; 
        else 
          inew = 0;
      }
    }

    if (p.v > 0.0F) {
      float d = (phantom.y_bounds[iy + 1] - p.y) / p.v;
      if (d <= t) { 
        t = d; 
        if (iy + 1 < phantom.N.y) 
          inew = p.region + phantom.N.x; 
        else 
          inew = 0;
      }
    }
    else if (p.v < 0.0F) {
      float d = (phantom.y_bounds[iy] - p.y) / p.v;
      if (d <= t) { 
        t = d; 
        if (iy - 1 >= 0) 
          inew = p.region - phantom.N.x; 
        else 
          inew = 0;
      }
    }

    if (p.w > 0.0F) {
      float d = (phantom.z_bounds[iz + 1] - p.z) / p.w;
      if (d <= t) { 
        t = d; 
        if (iz + 1 < phantom.N.z) 
          inew = p.region + phantom.N.x * phantom.N.y; 
        else 
          inew = 0;
      }
    }
    else if (p.w < 0.0F) {
      float d = (phantom.z_bounds[iz] - p.z) / p.w;
      if (d <= t) { 
        t = d; 
        if (iz - 1 >= 0) 
          inew = p.region - phantom.N.x * phantom.N.y; 
        else 
          inew = 0;
      }
    }

    return inew;
  }
  // this part corresponds to the function howfarFromOut of the class EGS_XYZGeometry 
  // in the file egs_nd_geometry.h (v 1.26 2009/07/06) of the EGSnrc C++ Class Library
  else {
    int ix, iy, iz;
    float t1;

    ix = -1;
    if ((p.x <= phantom.x_bounds[0]) && (p.u > 0.0F)) {
      t1 = (phantom.x_bounds[0] - p.x) / p.u; 
      ix = 0;
    }
    else if ((p.x >= phantom.x_bounds[phantom.N.x]) && (p.u < 0.0F)) {
      t1 = (phantom.x_bounds[phantom.N.x] - p.x) / p.u; 
      ix = phantom.N.x - 1;
    }

    if ((ix >= 0) && (t1 <= t)) {
      float y1 = p.y + p.v * t1;
      iy = isWhere(y1, phantom.N.y, phantom.y_bounds);

      if (iy >= 0) {
        float z1 = p.z + p.w * t1;
        iz = isWhere(z1, phantom.N.z, phantom.z_bounds);

        if (iz >= 0) {
          t = t1; 
          return ix + iy * phantom.N.x + iz * phantom.N.x * phantom.N.y + 1;             
        }
      }
    }

    iy = -1;
    if ((p.y <= phantom.y_bounds[0]) && (p.v > 0.0F)) {
      t1 = (phantom.y_bounds[0] - p.y) / p.v; 
      iy = 0;
    }
    else if ((p.y >= phantom.y_bounds[phantom.N.y]) && (p.v < 0.0F)) {
      t1 = (phantom.y_bounds[phantom.N.y] - p.y) / p.v; 
      iy = phantom.N.y - 1;
    }

    if ((iy >= 0) && (t1 <= t)) {
      float x1 = p.x + p.u * t1;
      ix = isWhere(x1, phantom.N.x, phantom.x_bounds);

      if (ix >= 0) {
        float z1 = p.z + p.w * t1;
        iz = isWhere(z1, phantom.N.z, phantom.z_bounds);

        if (iz >= 0) {
          t = t1; 
          return ix + iy * phantom.N.x + iz * phantom.N.x * phantom.N.y + 1; 
        }
      }
    }

    iz = -1;
    if ((p.z <= phantom.z_bounds[0]) && (p.w > 0.0F)) {
      t1 = (phantom.z_bounds[0] - p.z) / p.w; 
      iz = 0;
    }
    else if ((p.z >= phantom.z_bounds[phantom.N.z]) && (p.w < 0.0F)) {
      t1 = (phantom.z_bounds[phantom.N.z] - p.z) / p.w; 
      iz = phantom.N.z - 1;
    }

    if ((iz >= 0) && (t1 <= t)) {
      float x1 = p.x + p.u * t1;
      ix = isWhere(x1, phantom.N.x, phantom.x_bounds);

      if (ix >= 0) {
        float y1 = p.y + p.v * t1;
        iy = isWhere(y1, phantom.N.y, phantom.y_bounds);

        if (iy >= 0) {
          t = t1; 
          return ix + iy * phantom.N.x + iz * phantom.N.x * phantom.N.y + 1; 
        }
      }
    }

    return 0;
  }
}

/**********************************************************************
 * THIS FUNCTION MUST *ALWAYS* BE CALLED BY *ALL* THREADS IN THE WARP *
 **********************************************************************/
// This is the subroutine UPHI(IENTRY,LVL) with IENTRY = 2 and LVL = 1 in the file 
// egsnrc.mortran (v 1.72 2011/05/05) of the EGSnrc code system.
// However, note that we are not using the box method implemented in the macro 
// $SELECT-AZIMUTHAL-ANGLE, because that involves a sampling loop and then all
// threads would have to wait until the last thread has finished the loop. Calculating
// sin and cos with __sincosf is not that expensive, so we use that instead.
// Note that this is based on assumption and was not experimentally verified.
__device__ void uphi21(indices idx, float costhe, float sinthe, particle_t &p) {
  float r1 = get_rand(idx);

  if (p.process) {
    float phi = 2.0F * 3.14159265F * r1;
    float cosphi, sinphi;
    __sincosf(phi, &sinphi, &cosphi);

    float sinps2 = p.u * p.u + p.v * p.v;
    // small polar angle
    if (sinps2 < SMALL_POLAR_ANGLE_THRESHOLD) {
      p.u = sinthe * cosphi;
      p.v = sinthe * sinphi;
      p.w = p.w * costhe;
    }
    else {
      float sinpsi = sqrt(sinps2);
      float us = sinthe * cosphi;
      float vs = sinthe * sinphi;
      float sindel = p.v / sinpsi;
      float cosdel = p.u / sinpsi;

      p.u = p.w * cosdel * us - sindel * vs + p.u * costhe; 
      p.v = p.w * sindel * us + cosdel * vs + p.v * costhe;
      p.w = -sinpsi * us + p.w * costhe;
    }
  }
}

// Add the weight wt and energy e to the pixel (x,y) in the category cat. atomicAdd is
// used to avoid data hazards if multiple threads try to update the same pixel at the 
// same time.
__device__ void score_pixel(indices idx, int x, int y, uchar cat, float wt, float e) {
  // score the particle if the pixel is on the detector
  if ((x >= 0) && (x < (int)detector.N.x) &&
      (y >= 0) && (y < (int)detector.N.y)) {
    atomicAdd(&detector_scores_count[idx.b][cat][y * detector.N.x + x], wt);
    atomicAdd(&detector_scores_energy[idx.b][cat][y * detector.N.x + x], e);
  }
}


/* * * * * * * * * * * * * * *
 * Simulation Step Functions *
 * * * * * * * * * * * * * * */

// Create a new particle.
// This is essentially the function getNextParticle of the class EGS_CollimatedSource in the
// EGSnrc C++ Class Library with a source shape EGS_PointShape, target shape EGS_RectangleShape
// and a spectrum EGS_MonoEnergy or EGS_TabulatedSpectrum (the underlying EGS_AliasTable has 
// type = 1, i.e. it is a histogram).
__device__ void new_particle(indices idx, particle_t &p, volatile float *weight_list) {
  float r1 = get_rand(idx);
  float r2 = get_rand(idx);

  // for spectrum
#ifdef USE_ENERGY_SPECTRUM
  float r3 = get_rand(idx);
  float r4 = get_rand(idx); 
#endif

  if (p.process) {
    // set charge and next step
    p.charge = 0;
    p.reserved = 0;
    p.status = p_photon_step;

    // set energy and region
    p.region = 0;

#ifdef USE_ENERGY_SPECTRUM
    float aj = r3 * (float)source.n; 
    int j = (int)aj; 
    aj -= j;
    if (aj > source.wi[j])
      j = source.bin[j];

    float x = source.xi[j]; 
    float dx = source.xi[j+1] - x;

    p.e = x + dx * r4;
#else
    p.e = source.energy;
#endif

    // set source point
    p.x = source.source_point.x;
    p.y = source.source_point.y;
    p.z = source.source_point.z;

    // target point is uniformly distributed on rectangle
    float3 target_point = make_float3(source.rectangle_min.x + source.rectangle_size.x * r1, 
        source.rectangle_min.y + source.rectangle_size.y * r2, 
        source.rectangle_z);

    // calculate direction
    p.u = target_point.x - p.x;
    p.v = target_point.y - p.y;
    p.w = target_point.z - p.z;

    // normalize direction
    float d2i = 1 / (p.u * p.u + p.v * p.v + p.w * p.w);
    float di = sqrtf(d2i);
    p.u *= di;
    p.v *= di;
    p.w *= di;

    // calculate weight and set latch
    p.wt = source.rectangle_area * fabsf(p.w) * d2i;
    weight_list[idx.t] = p.wt;
    p.latch = 0;
  }    
  else
    weight_list[idx.t] = 0.0F;
}

// Photon energy fell below the cutoff energy, destroy the photon.
__device__ void cutoff_discard(indices idx, particle_t &p) {
  if (p.process)
    p.status = p_empty;
}

// Propagate the photon to the detector.
__device__ void user_discard(indices idx, particle_t &p) {
  if (!p.process)
    return;

  p.status = p_empty;

  // do not score if...
  if ((p.charge != 0) ||   // it is not a photon
      (p.region != 0) ||   // it is not in region 0 (outside)
      (p.w == 0.0F)) {     // it is not going parallel to the z direction
    return;
  }

  // propagate to image plane
  float delta = (detector.center.z - p.z) / p.w;

  // photon does not hit the detector
  if (delta < 0.0F)
    return;

  float2 pos = make_float2(p.x + delta * p.u,
      p.y + delta * p.v);

  // find pixel where the photon hits the detector
  float2 pix = make_float2((pos.x - detector.center.x) / detector.d.x + (float)detector.N.x / 2.0F,
      (pos.y - detector.center.y) / detector.d.y + (float)detector.N.y / 2.0F);

  float2 lower = make_float2(floorf(pix.x), floorf(pix.y));
  float2 upper = make_float2(ceilf(pix.x), ceilf(pix.y));

  int split = 0;
  int2 pixel;

  // split pixels if photon hits detector very close to pixel boundaries
  if (almostEqual(pix.x, lower.x)) {
    split += 1;
    pixel.x = (int)lower.x;
  } else if (almostEqual(pix.x, upper.x)) {
    split += 1;
    pixel.x = (int)upper.x;
  }
  else
    pixel.x = (int)lower.x;

  if (almostEqual(pix.y, lower.y)) {
    split += 2;
    pixel.y = (int)lower.y;
  } else if (almostEqual(pix.y, upper.y)) {
    split += 2;
    pixel.y = (int)upper.y;
  }
  else
    pixel.y = (int)lower.y;

  ushort num_compton = p.latch & 0xFFFFU;
  ushort num_rayleigh = p.latch >> 16;
  uchar cat = 0;

  if ((num_compton == 0) && (num_rayleigh == 0))
    cat = 0;
  else if ((num_compton == 1) && (num_rayleigh == 0))
    cat = 1;
  else if ((num_compton == 0) && (num_rayleigh == 1))
    cat = 2;
  else
    cat = 3;

  switch (split) {
    case 0:
      score_pixel(idx, pixel.x, pixel.y, cat, p.wt, p.e * p.wt);
      break;

    case 1:
      p.wt /= 2.0F;
      p.e *= p.wt;
      score_pixel(idx, pixel.x, pixel.y, cat, p.wt, p.e);
      score_pixel(idx, pixel.x - 1, pixel.y, cat, p.wt, p.e);
      break;

    case 2:
      p.wt /= 2.0F;
      p.e *= p.wt;
      score_pixel(idx, pixel.x, pixel.y, cat, p.wt, p.e);
      score_pixel(idx, pixel.x, pixel.y - 1, cat, p.wt, p.e);
      break;

    case 3:
      p.wt /= 4.0F;
      p.e *= p.wt;
      score_pixel(idx, pixel.x, pixel.y, cat, p.wt, p.e);
      score_pixel(idx, pixel.x - 1, pixel.y, cat, p.wt, p.e);
      score_pixel(idx, pixel.x, pixel.y - 1, cat, p.wt, p.e);
      score_pixel(idx, pixel.x - 1, pixel.y - 1, cat, p.wt, p.e);
      break;
  }
}

// Transport the photon one step through the phantom and determine which (if any) interaction
// takes place next.
// This is the subroutine PHOTON in the file egsnrc.mortran (v 1.72 2011/05/05) of the EGSnrc 
// code system.
__device__ void photon_step(indices idx, particle_t &p) {
  region_data_t reg_dat = region_data[p.region];

  if (p.process) {
    if (p.e <= reg_dat.pcut)
      p.status = p_cutoff_discard;
    else if (p.wt <= 0.0F)
      p.status = p_user_discard;       
  }

  float r1 = get_rand(idx);
  float dpmfp = -logf(r1);

  float gle = logf(p.e);
  int lgle = 0;
  float gmfpr0 = 0.0F;
  float tstep = 0.0F;
  float gmfp_val = 0.0F;
  float cohfac = 0.0F;
  ushort old_medium = reg_dat.med;

  if (p.process && (p.status == p_photon_step)) {
    if (reg_dat.med == VACUUM)
      tstep = VACUUM_STEP;
    else {
      float2 ge_dat = ge[reg_dat.med];
      lgle = (int)(ge_dat.x + ge_dat.y * gle);
      float2 gmfp_dat = gmfp[reg_dat.med * MXGE + lgle];
      gmfpr0 = gmfp_dat.x + gmfp_dat.y * gle;

      gmfp_val = gmfpr0 / reg_dat.rhof;

      if ((reg_dat.flags & f_rayleigh) > 0) {
        float2 cohe_dat = cohe[reg_dat.med * MXGE + lgle];
        cohfac = cohe_dat.x + cohe_dat.y * gle; 
        gmfp_val *= cohfac;
      }
      tstep = gmfp_val * dpmfp;
    }

    // HOWFAR
    uint new_region = howfar(idx, p, tstep);

    char idisc = 0;
    if (new_region == 0) {
      if (reg_dat.med == VACUUM)
        idisc = 1;
      else
        idisc = -1;
    }

    if (idisc > 0) {
      p.region = 0;
      p.status = p_user_discard;
    }
    else {
      p.x += p.u * tstep;
      p.y += p.v * tstep;
      p.z += p.w * tstep;

      if (reg_dat.med != VACUUM) 
        dpmfp = fmax(0.0F, dpmfp - tstep / gmfp_val);

      old_medium = reg_dat.med;

      if (new_region != p.region) {
        p.region = new_region;
        reg_dat = region_data[new_region];
      }

      if (p.e <= reg_dat.pcut)
        p.status = p_cutoff_discard;
      else if (idisc < 0)
        p.status = p_user_discard;
    } 
  }

  // determine next step if not already discarded 

  bool process = p.process && (p.status == p_photon_step);

  if (process && ((reg_dat.med != old_medium) || (reg_dat.med == VACUUM) || (dpmfp >= EPSGMFP))) {
    p.status = p_photon_step;
    process = false;
  }

  float r2 = get_rand(idx);
  if (process && ((reg_dat.flags & f_rayleigh) > 0)) {
    if (r2 < 1.0F - cohfac) {
      p.status = p_rayleigh;
      process = false;
    }
  }

  float r3 = get_rand(idx);
  if (process) {
    float2 gbr1_dat = gbr1[reg_dat.med * MXGE + lgle];
    float gbr1_val = gbr1_dat.x + gbr1_dat.y * gle;
    float2 gbr2_dat = gbr2[reg_dat.med * MXGE + lgle];
    float gbr2_val = gbr2_dat.x + gbr2_dat.y * gle;

    if ((r3 <= gbr1_val) && (p.e > 2.0F * ELECTRON_REST_MASS_FLOAT))
      p.status = p_pair;
    else {
      if (r3 < gbr2_val)
        p.status = p_compton; 
      else
        p.status = p_photo;
    }
  }
}

// Perform a Rayleigh interaction.
// This is the subroutine egs_rayleigh_sampling in the file egsnrc.mortran (v 1.72 2011/05/05) 
// of the EGSnrc code system.
__device__ void rayleigh(indices idx, particle_t &p) {
  region_data_t reg_dat;

  if (p.process) {
    reg_dat = region_data[p.region];
    p.status = p_photon_step;

    // count scatter event
    p.latch += (1 << 16);
  }

  float xmax = 0.0F;
  float pmax_val = 0.0F;

  if (p.process) {
    float gle = logf(p.e);
    float2 ge_dat = ge[reg_dat.med];
    int lgle = (int)(ge_dat.x + ge_dat.y * gle);

    float2 pmax_dat = pmax[reg_dat.med * MXGE + lgle];
    pmax_val = pmax_dat.x + pmax_dat.y * gle; 
    xmax = HC_INVERSE * p.e;
  }

  int dwi = RAYCDFSIZE - 1;
  int ibin = 0;
  int ib = 0;

  float xv = 0.0F;
  float costhe = 0.0F;
  float costhe2 = 0.0F;
  float sinthe = 0.0F;

  bool loop_done = !p.process;

  do {
    bool inner_loop_done = loop_done;

    do {
      float r1 = get_rand(idx);

      if (!inner_loop_done) {
        float temp = r1 * pmax_val;
        // indexing in C starts at 0 and not 1 as in FORTRAN
        ibin = (int)(temp * (float)dwi);
        ib = i_array[reg_dat.med * RAYCDFSIZE + ibin] - 1;
        int next_ib = i_array[reg_dat.med * RAYCDFSIZE + ibin + 1] - 1;

        if (next_ib > ib) {
          do {
            rayleigh_data_t ray_dat = rayleigh_data[reg_dat.med * MXRAYFF + ib + 1];
            if ((temp < ray_dat.fcum) || (ib >= RAYCDFSIZE - 2))
              break;
            ib++;
          } while (true);
        }

        rayleigh_data_t ray_dat = rayleigh_data[reg_dat.med * MXRAYFF + ib];
        temp = (temp - ray_dat.fcum) * ray_dat.c_array;
        xv = ray_dat.xgrid * expf(logf(1.0F + temp) * ray_dat.b_array);

        if (xv < xmax)
          inner_loop_done = true;
      }

    } while (!__all_sync(MASK, inner_loop_done));

    float r2 = get_rand(idx);

    if (!loop_done) {
      xv = xv / p.e;
      costhe = 1.0F - TWICE_HC2 * xv * xv;
      costhe2 = costhe * costhe;

      if (2.0F * r2 < 1.0F + costhe2)
        loop_done = true;
    }

  } while (!__all_sync(MASK, loop_done));

  sinthe = sqrtf(1.0F - costhe2);
  uphi21(idx, costhe, sinthe, p);
}

// Perform a Compton interaction.
// This is a simplified version of the subroutine COMPT in the file egsnrc.mortran (v 1.72 2011/05/05) 
// of the EGSnrc code system. The simplification is that we do not consider bound compton scattering,
// that we always use Klein-Nishina and do not create an electron.
__device__ void compton(indices idx, particle_t &p) {
  if (p.process) {
    p.status = p_photon_step;

    // count scatter event
    p.latch += 1;
  }

  float ko = p.e / ELECTRON_REST_MASS_FLOAT;
  float broi = 1.0F + 2.0F * ko;
  float bro = 1.0F / broi;

  // sampling loop
  bool loop_done = !p.process;

  float sinthe = 0.0F;
  float costhe = 0.0F;
  float br = 0.0F;

  do {
    float r1 = get_rand(idx);
    float r2 = get_rand(idx);
    float r3 = get_rand(idx);

    if (!loop_done) {
      if (ko > 2.0F) {
        float broi2 = broi * broi;
        float alph1 = logf(broi);
        float alph2 = ko * (broi + 1.0F) * bro * bro;
        float alpha = alph1 + alph2;

        if (r1 * alpha < alph1)
          br = expf(alph1 * r2) * bro;
        else
          br = sqrtf(r2 * broi2 + 1.0F - r2) * bro;

        costhe = (1.0F - br) / (ko * br);
        sinthe = fmax(0.0F, costhe * (2.0F - costhe));
        float aux = 1.0F + br * br;
        float rejf3 = aux - br * sinthe;

        if (r3 * aux < rejf3)
          loop_done = true;
      }
      else {
        float bro1 = 1.0F - bro;
        float rejmax = broi + bro;

        br = bro + bro1 * r1;
        costhe = (1.0F - br) / (ko * br);
        sinthe = fmax(0.0F, costhe * (2.0F - costhe));
        float rejf3 = 1.0F + br * br - br * sinthe;
        if (r2 * br * rejmax < rejf3)
          loop_done = true;
      }
    }

  } while (!__all_sync(MASK, loop_done));

  costhe = 1.0F - costhe;
  sinthe = sqrtf(sinthe);

  if (p.process)
    p.e *= br;

  uphi21(idx, costhe, sinthe, p);
}

// Perform a pair production interaction. Since we do not consider electrons, we just destroy the photon.
__device__ void pair_production(indices idx, particle_t &p) {
  if (p.process)
    p.status = p_empty;
}

// Perform a photo electric interaction. Since we do not consider electrons, we just destroy the photon.
__device__ void photo(indices idx, particle_t &p) {
  if (p.process)
    p.status = p_empty;
}


/* * * * * *
 * Kernels *
 * * * * * */

// this is the simulation kernel
/* * * * * * * * * * * * * * * * * * * * * * * * * *
 * Number of blocks: SIMULATION_NUM_BLOCKS         *
 * Number of warps:  SIMULATION_WARPS_PER_BLOCK    *
 * * * * * * * * * * * * * * * * * * * * * * * * * */
__global__ void simulation_step_kernel(bool init, bool limit_reached) {
  indices idx = get_indices();

  volatile uint *step_counters = step_counters_shared[idx.w];
  volatile float *weight_list = weight_list_shared[idx.w];
  volatile double *combined_weight_list = combined_weight_list_shared;

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  volatile uint *list_depth = list_depth_shared;
  volatile uint *num_inner_iterations = num_inner_iterations_shared;
#endif

  // reset detector pixels
  for (uint i = 0; i < NUM_DETECTOR_CAT; i++) {
    for (uint j = threadIdx.x; j < detector.N.x * detector.N.y; j += blockDim.x) {
      detector_scores_count[idx.b][i][j] = 0.0F;
      detector_scores_energy[idx.b][i][j] = 0.0F;
    }
  }

  __syncthreads();

  // reset step counts
  if (idx.t < NUM_CAT)
    step_counters[idx.t] = 0;

  // reset weight counter
  if (idx.t == 0)
    combined_weight_list[idx.w] = 0.0F;

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  if (idx.t == 0) {
    list_depth[idx.w] = 0;
    num_inner_iterations[idx.w] = 0;
  }
#endif

  // read MT status
  MT_read_status(idx);

  particle_t p;

  // read particle from stack
  uint4 tmp = stack.a[idx.p];
  p.status = ((uchar*)&tmp.x)[0];
  p.reserved = ((uchar*)&tmp.x)[1];
  p.charge = ((uchar*)&tmp.x)[2];
  p.process = ((uchar*)&tmp.x)[3];
  p.e = *(float*)&tmp.y;
  p.wt = *(float*)&tmp.z;
  p.region = tmp.w;

  tmp = stack.b[idx.p];
  p.latch = tmp.x;
  p.x = *(float*)&tmp.y;
  p.y = *(float*)&tmp.z;
  p.z = *(float*)&tmp.w;

  tmp = stack.c[idx.p];
  p.u = *(float*)&tmp.x;
  p.v = *(float*)&tmp.y;
  p.w = *(float*)&tmp.z;

  if (init)
    p.status = p_new_particle;

  bool done;

  for (uint i = 0; i < SIMULATION_ITERATIONS; i++) {

    // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
    if (idx.t == 0)
      num_inner_iterations[idx.w] += 1;
#endif

    done = false;
    if ((p.status == p_empty) && (!limit_reached))
      p.status = p_new_particle;

    // photon step
    p.process = (p.status == p_photon_step);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      photon_step(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_photon_step] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // new particle
    p.process = (p.status == p_new_particle);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      new_particle(idx, p, weight_list);

      // add the weights together
      if (idx.t == 0) {
        double combined_weight = 0.0F;

        for (uchar j = 0; j < WARP_SIZE; j++)
          combined_weight += (double)weight_list[j];

        combined_weight_list[idx.w] += combined_weight;
      }

      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_new_particle] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // user discard
    p.process = (p.status == p_user_discard);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      user_discard(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_user_discard] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // compton
    p.process = (p.status == p_compton);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      compton(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_compton] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // photoelectric effect
    p.process = (p.status == p_photo);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      photo(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_photo] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // rayleigh
    p.process = (p.status == p_rayleigh);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      rayleigh(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_rayleigh] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // pair production
    p.process = (p.status == p_pair);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      pair_production(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_pair] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }
    if (__all_sync(MASK, done))
      continue;

    // cutoff discard
    p.process = (p.status == p_cutoff_discard);
    done |= p.process;
    if (__any_sync(MASK, p.process)) {
      cutoff_discard(idx, p);
      uint count_mask = __ballot_sync(MASK, p.process);
      if (idx.t == 0) {
        step_counters[p_cutoff_discard] += __popc(count_mask);
#ifdef DO_LIST_DEPTH_COUNT
        list_depth[idx.w] += 1;
#endif
      }
    }

    if ((__all_sync(MASK, !done)) && (limit_reached))
      break;

  }

  __syncthreads();

  // combine the counters in shared memory and write them to global memory
  if (threadIdx.x < NUM_CAT) {
    uint total_count = 0;

    // step through the warps
    for (uchar i = 0; i < SIMULATION_WARPS_PER_BLOCK; i++)
      total_count += step_counters_shared[i][threadIdx.x];

    (*total_step_counts)[idx.b][threadIdx.x] = total_count;
  }

  // combine the weights in shared memory and write them to global memory
  if (threadIdx.x == 0) {
    double total_weight = 0.0F;

    // step through the warps
    for (uchar i = 0; i < SIMULATION_WARPS_PER_BLOCK; i++)
      total_weight += combined_weight_list[i];

    if (total_weight > 0.0F)
      (*total_weights)[idx.b] += total_weight;
  }

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  if (threadIdx.x == 0) {
    uint tot_list_depth = 0;
    uint tot_it = 0;

    for (uchar i =0; i < SIMULATION_WARPS_PER_BLOCK; i++) {
      tot_list_depth += list_depth[i];
      tot_it += num_inner_iterations[i];
    }

    (*total_list_depth)[idx.b] = tot_list_depth;
    (*total_num_inner_iterations)[idx.b] = tot_it;
  }
#endif

  // write particle back to stack
  ((uchar*)&tmp.x)[0] = p.status;
  ((uchar*)&tmp.x)[1] = p.reserved;
  ((uchar*)&tmp.x)[2] = p.charge;
  ((uchar*)&tmp.x)[3] = p.process;
  tmp.y = *(uint*)&p.e;
  tmp.z = *(uint*)&p.wt;
  tmp.w = p.region;
  stack.a[idx.p] = tmp;

  tmp.x = p.latch;
  tmp.y = *(uint*)&p.x;
  tmp.z = *(uint*)&p.y;
  tmp.w = *(uint*)&p.z;
  stack.b[idx.p] = tmp;

  tmp.x = *(uint*)&p.u;
  tmp.y = *(uint*)&p.v;
  tmp.z = *(uint*)&p.w;
  stack.c[idx.p] = tmp;

  // write MT status
  MT_write_status(idx);
}

// this is the summing kernel
/* * * * * * * * * * * * * * * * * * * * * * * * * *
 * Number of blocks: SUM_DETECTOR_NUM_BLOCKS       *
 * Number of warps:  SUM_DETECTOR_WARPS_PER_BLOCK  *
 * * * * * * * * * * * * * * * * * * * * * * * * * */
__global__ void sum_detector_scores_kernel() {
  bool do_count = blockIdx.x / NUM_DETECTOR_CAT;
  uchar cat = blockIdx.x % NUM_DETECTOR_CAT;

  detector_scores_t *scores;
  double *totals;
  if (do_count) {
    scores = &detector_scores_count;
    totals = detector_totals_count[cat];
  }
  else {
    scores = &detector_scores_energy;
    totals = detector_totals_energy[cat];
  }

  for (uint i = threadIdx.x; i < detector.N.x * detector.N.y; i += blockDim.x) {
    double total = 0.0F;

    for (uint j = 0; j < SIMULATION_NUM_BLOCKS; j++)
      total += (double)((*scores)[j][cat][i]);

    totals[i] += total;
  }
}

#endif
