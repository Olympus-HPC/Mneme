/****************************************************************************
 *
 * init.cu, Version 1.0.0 Mon 09 Jan 2012
 *
 * ----------------------------------------------------------------------------
 *
 * CUDA EGS
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

// remove all control characters (possibly including spaces) from a string
string trim(string str, bool allowSpace) {
  string trimmed;
  for (uint i = 0; i < str.length(); i++) {
    char c = str[i];
    if ((c > 32) || (allowSpace && (c == 32)))
      trimmed += c;
  }
  return trimmed;
}

// read the Mersenne Twister (MT) parameters from the parameter file, initialize the MTs 
// with the given seed and copy the relevant data to the device
void init_MTs(uint seed) {
  // read MT parameters from file generated with MTGPDC
  FILE *params = fopen(MT_params_file, "rb");
  if (params == NULL)
    printf("Could not open the MT parameter file \"%s\".\n", MT_params_file);

  uint num = SIMULATION_WARPS_PER_BLOCK * SIMULATION_NUM_BLOCKS;
  MT_input_param *input_MTs = (MT_input_param*)malloc(num * sizeof(MT_input_param));
  fread(input_MTs, sizeof(MT_input_param), num, params);
  fclose(params);

  // initialize MTs that will run on GPU
  h_MT_params = (MT_param*)malloc(num * sizeof(MT_param));
  h_MT_statuses = (uint*)malloc(num * MT_NUM_STATUS * sizeof(uint));

  // create look-up tables and store them in textures
  uint size_tbl = num * sizeof(MT_tables_t);
  MT_tables_t *h_MT_tables = (MT_tables_t*)malloc(size_tbl);
  memset(h_MT_tables, 0, size_tbl);

  // calculate table entries, copy parameters and initialize the MT
  for (uint block = 0; block < SIMULATION_NUM_BLOCKS; block++) {
    for (uint warp = 0; warp < SIMULATION_WARPS_PER_BLOCK; warp++) {

      // set up indices for the table and the MT from the input array
      int MT_idx = block * SIMULATION_WARPS_PER_BLOCK + warp;

      // calculate table entries
      for (uint i = 1; i < MT_TABLE_SIZE; i++) {
        for (uint j = 1, k = 0; j <= i; j <<= 1, k++) {
          if (i & j) {
            h_MT_tables[MT_idx].recursion[i] ^= input_MTs[MT_idx].tbl[k];
            h_MT_tables[MT_idx].tempering[i] ^= input_MTs[MT_idx].tmp_tbl[k];
          }
        }
      }

      // change tempering table to produce float output
      for (uint i = 0; i < MT_TABLE_SIZE; i++)
        h_MT_tables[MT_idx].tempering[i] = (h_MT_tables[MT_idx].tempering[i] >> 9) | 0x3F800001U;

      // copy parameters
      h_MT_params[MT_idx].M = input_MTs[MT_idx].M;
      h_MT_params[MT_idx].sh1 = input_MTs[MT_idx].sh1;
      h_MT_params[MT_idx].sh2 = input_MTs[MT_idx].sh2;
      h_MT_params[MT_idx].mask = input_MTs[MT_idx].mask;

      // initilize the state using the given seed
      uint hidden_seed = h_MT_tables[MT_idx].recursion[4] ^ (h_MT_tables[MT_idx].recursion[8] << 16);
      uint tmp = hidden_seed;
      tmp += tmp >> 16;
      tmp += tmp >> 8;

      memset(&h_MT_statuses[MT_idx * MT_NUM_STATUS], tmp & 0xFF, sizeof(uint) * MT_N);
      h_MT_statuses[MT_idx * MT_NUM_STATUS] = seed;
      h_MT_statuses[MT_idx * MT_NUM_STATUS + 1] = hidden_seed;
      for (uint i = 1; i < MT_N; i++)
        h_MT_statuses[MT_idx * MT_NUM_STATUS + i] ^= 0x6C078965U * (h_MT_statuses[MT_idx * MT_NUM_STATUS + i-1] ^ (h_MT_statuses[MT_idx * MT_NUM_STATUS + i-1] >> 30)) + i;
    }
  }

  // allocate device memory for tables
  cudaMalloc(&d_MT_tables, size_tbl);

  // copy tables to device
  cudaMemcpy(d_MT_tables, h_MT_tables, size_tbl, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(MT_tables, &d_MT_tables, sizeof(MT_tables_t*));

  // allocate memory on device for MTs and copy data from host to device
  cudaMalloc(&d_MT_params, num * sizeof(MT_param));
  cudaMalloc(&d_MT_statuses, num * MT_NUM_STATUS * sizeof(uint));
  cudaMemcpy(d_MT_params, h_MT_params, num * sizeof(MT_param), cudaMemcpyHostToDevice);
  cudaMemcpy(d_MT_statuses, h_MT_statuses, num * MT_NUM_STATUS * sizeof(uint), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(MT_params, &d_MT_params, sizeof(MT_param*));
  cudaMemcpyToSymbol(MT_statuses, &d_MT_statuses, sizeof(uint*));

  // free host memory that's no longer used
  free(input_MTs);
  free(h_MT_params);
  free(h_MT_statuses);
  free(h_MT_tables);
}

// allocate memory for the stack and associated counters
void init_stack() {
  uint stack_size = SIMULATION_NUM_BLOCKS * SIMULATION_WARPS_PER_BLOCK * WARP_SIZE;

  cudaMalloc(&d_stack.a, stack_size * sizeof(uint4));
  cudaMalloc(&d_stack.b, stack_size * sizeof(uint4));
  cudaMalloc(&d_stack.c, stack_size * sizeof(uint4));
  cudaMemcpyToSymbol(stack, &d_stack, sizeof(stack_t));

  // allocate step counts on host
  h_total_step_counts = (total_step_counts_t*)malloc(sizeof(total_step_counts_t));

  // init step counts on the device
  cudaMalloc(&d_total_step_counts, sizeof(total_step_counts_t));
  cudaMemset(d_total_step_counts, 0, sizeof(total_step_counts_t));
  cudaMemcpyToSymbol(total_step_counts, &d_total_step_counts, sizeof(total_step_counts_t*));

  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  h_total_list_depth = (total_list_depth_t*)malloc(sizeof(total_list_depth_t));
  h_total_num_inner_iterations = (total_num_inner_iterations_t*)malloc(sizeof(total_num_inner_iterations_t));

  cudaMalloc(&d_total_list_depth, sizeof(total_list_depth_t));
  cudaMemset(d_total_list_depth, 0, sizeof(total_list_depth_t));
  cudaMemcpyToSymbol(total_list_depth, &d_total_list_depth, sizeof(total_list_depth_t*));

  cudaMalloc(&d_total_num_inner_iterations, sizeof(total_num_inner_iterations_t));
  cudaMemset(d_total_num_inner_iterations, 0, sizeof(total_num_inner_iterations_t));
  cudaMemcpyToSymbol(total_num_inner_iterations, &d_total_num_inner_iterations, sizeof(total_num_inner_iterations_t*));
#endif
}

// read the source parameters from the input file and copy the relevant data to the device
void init_source() {

  // we are using an energy spectrum
#ifdef USE_ENERGY_SPECTRUM

  // read spectrum file, copied from egs_spectra.cpp
  ifstream sdata(spec_file);
  if (!sdata)
    printf("Could not open spectrum file \"%s\".\n", spec_file);

  char title[1024];
  sdata.getline(title,1023);
  if( sdata.eof() || sdata.fail() || !sdata.good() ) {
    printf("Could not read the title of spectrum file \"%s\".\n", spec_file);
  }
  if( sdata.eof() || sdata.fail() || !sdata.good() ) {
    printf("Could not read the spectrum type and the number of bins in spectrum file \"%s\".\n", spec_file);
  }
  double dum; int nbin, mode;
  sdata >> nbin >> dum >> mode;
  if( sdata.eof() || sdata.fail() || !sdata.good() ) {
    printf("Could not read the spectrum type and the number of bins in spectrum file \"%s\".\n", spec_file);
  }
  if( nbin < 2 ) {
    printf("The number of bins in the spectrum must be at least 2, but found %d in the spectrum file \"%s\".\n", nbin, spec_file);
  }
  if( mode < 0 || mode > 2 ) {
    printf("Unknown spectrum type %d found in spectrum file \"%s\"\n", mode, spec_file);
  }

  double *en_array, *f_array; int ibin;
  f_array = new double [nbin];
  if( mode == 0 || mode == 1 ) {
    en_array = new double [nbin+1];
    en_array[0] = dum; ibin=1;
  }
  else {
    en_array = new double [nbin]; ibin=0;
  }
  for(int j=0; j<nbin; j++) {
    sdata >> en_array[ibin++] >> f_array[j];
    if( sdata.eof() || sdata.fail() || !sdata.good() ) {
      printf("Could not read line %d in spectrum file \"%s\"\n", j+2, spec_file);
      delete [] en_array; delete [] f_array;
    }
    if( mode != 2 && ibin > 1 ) {
      if( en_array[ibin-1] <= en_array[ibin-2] ) {
        printf("Energies are not in increasing order on lines %d and %d in the spectrum file \"%s\".\n", j+2, j+1, spec_file);
      }
    }
    if( mode == 0 ) 
      f_array[j]/=(en_array[ibin-1]-en_array[ibin-2]);
  }

  sdata.close();

  int itype = 1;
  if( mode == 2 ) itype = 0;
  else if( mode == 3 ) itype = 2;
  int nb = itype == 1 ? nbin+1 : nbin;

  int type = itype;
  int Type = itype;

  uint n = nb;
  uint np = n;
  h_source.n = n;

  double *xi = new double[n];
  double *fi = new double[n]; 
  double *wi = new double[n]; 
  int *bin = new int[n];

  double *x = en_array;
  double *f = f_array;

  for(int i=0; i<np; i++) { xi[i] = x[i]; fi[i] = f[i]; }
  if( Type ) xi[np] = x[np]; 
  if( Type == 2 ) fi[np] = f[np];

  double *fcum = new double[np]; bool *not_done = new bool[np];
  double sum = 0, sum1 = 0; int i;
  for(i=0; i<np; i++) {
    if( type == 0 ) fcum[i] = fi[i];
    else if( type == 1 ) fcum[i] = fi[i]*(xi[i+1]-xi[i]);
    else fcum[i] = 0.5*(fi[i]+fi[i+1])*(xi[i+1]-xi[i]);
    sum += fcum[i]; wi[i] = 1; bin[i] = 0; 
    not_done[i] = true; wi[i] = 1;
    if( type == 0 ) sum1 += fcum[i]*xi[i];
    else if( type == 1 ) sum1 += 0.5*fcum[i]*(xi[i+1]+xi[i]);
    else sum1 += fcum[i]*(fi[i]*(2*xi[i]+xi[i+1])+
        fi[i+1]*(xi[i]+2*xi[i+1]))/(3*(fi[i]+fi[i+1]));
  }

  for(i=0; i<np; i++) fi[i] /= sum;
  sum /= np;  int jh, jl; 
  for(i=0; i<np-1; i++) {
    for(jh=0; jh<np-1; jh++) {
      if(not_done[jh] && fcum[jh] > sum) break;
    }
    for(jl=0; jl<np-1; jl++) {
      if(not_done[jl] && fcum[jl] < sum) break;
    }
    double aux = sum - fcum[jl];
    fcum[jh] -= aux; not_done[jl] = false;
    wi[jl] = fcum[jl]/sum; bin[jl] = jh;
  }
  delete [] fcum; delete [] not_done;

  delete [] en_array; delete [] f_array;

  // convert to float
  float *xif = (float*)malloc(n * sizeof(float));
  float *wif = (float*)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    xif[i] = (float)xi[i];
    wif[i] = (float)wi[i];
  }

  // copy data to device
  cudaMalloc(&h_source.xi, n * sizeof(float));
  cudaMalloc(&h_source.wi, n * sizeof(float));
  cudaMalloc(&h_source.bin, n * sizeof(int));

  cudaMemcpy(h_source.xi, xif, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(h_source.wi, wif, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(h_source.bin, bin, n * sizeof(float), cudaMemcpyHostToDevice);

  free(xif);
  free(wif);
  delete[] xi;
  delete[] fi;
  delete[] wi;
  delete[] bin;

  // we are using a monoenergetic source
#else

  // read source energy from the input file
  float e = 0.08;
  h_source.energy = e;

#endif

  // read source position from the input file
  float p1 = 0, p2 = 0, p3 = -30;
  h_source.source_point = make_float3(p1, p2, p3);

  // read collimator rectangle from the input file
  float r1 = -6.4, r2 = -6.4, r3 = 6.4, r4 = 6.4;
  h_source.rectangle_min = make_float2(r1, r2);
  h_source.rectangle_max = make_float2(r3, r4);

  // read source collimator z coordinate from the input file
  float z = -6.4;
  h_source.rectangle_z = z;

  h_source.rectangle_size = make_float2(h_source.rectangle_max.x - h_source.rectangle_min.x, 
      h_source.rectangle_max.y - h_source.rectangle_min.y);
  h_source.rectangle_area = h_source.rectangle_size.x * h_source.rectangle_size.y;

  cudaMemcpyToSymbol(source, &h_source, sizeof(source));

  printf("\nSource\n");
#ifdef USE_ENERGY_SPECTRUM
  printf("  Spectrum file . . . . . . . %s\n", spec_file);
#else
  printf("  Energy  . . . . . . . . . . %f\n", h_source.energy);
#endif
  printf("  Position  . . . . . . . . . x = %f, y = %f, z = %f\n", h_source.source_point.x, h_source.source_point.y, h_source.source_point.z);
  printf("  Collimator\n");
  printf("    x . . . . . . . . . . . . min = %f, max = %f\n", h_source.rectangle_min.x, h_source.rectangle_max.x);
  printf("    y . . . . . . . . . . . . min = %f, max = %f\n", h_source.rectangle_min.y, h_source.rectangle_max.y);
  printf("    z . . . . . . . . . . . . %f\n", h_source.rectangle_z);
}

// read the detector parameters from the input file and copy the relevant data to the device
void init_detector() {
  float p1 = 0, p2 = 0, p3 = 30;

  h_detector.center = make_float3(p1, p2, p3);

  // read detector size from the input file
  int s1 = 512, s2 = 512;
  h_detector.N = make_uint2(s1, s2);

  // read detector pixel size from the input file
  float ps1 = 0.1, ps2 = 0.1;
  h_detector.d = make_float2(ps1, ps2);

  // copy detector parameters to the device
  cudaMemcpyToSymbol(detector, &h_detector, sizeof(detector));

  // init detector scores on the device
  uint size_det_f = h_detector.N.x * h_detector.N.y * sizeof(float);
  uint size_det_d = h_detector.N.x * h_detector.N.y * sizeof(double);

  for (uchar i = 0; i < SIMULATION_NUM_BLOCKS; i++) {
    for (uchar j = 0; j < NUM_DETECTOR_CAT; j++) {
      cudaMalloc(&d_detector_scores_count[i][j], size_det_f);
      cudaMemset(d_detector_scores_count[i][j], 0, size_det_f);
      cudaMalloc(&d_detector_scores_energy[i][j], size_det_f);
      cudaMemset(d_detector_scores_energy[i][j], 0, size_det_f);
    }
  }

  for (uchar i = 0; i < NUM_DETECTOR_CAT; i++) {
    cudaMalloc(&d_detector_totals_count[i], size_det_d);
    cudaMemset(d_detector_totals_count[i], 0, size_det_d);
    cudaMalloc(&d_detector_totals_energy[i], size_det_d);
    cudaMemset(d_detector_totals_energy[i], 0, size_det_d);
  }

  cudaMemcpyToSymbol(detector_scores_count, &d_detector_scores_count, sizeof(detector_scores_t));
  cudaMemcpyToSymbol(detector_scores_energy, &d_detector_scores_energy, sizeof(detector_scores_t));
  cudaMemcpyToSymbol(detector_totals_count, &d_detector_totals_count, sizeof(d_detector_totals_count));
  cudaMemcpyToSymbol(detector_totals_energy, &d_detector_totals_energy, sizeof(d_detector_totals_energy));

  // init total weights on the device
  cudaMalloc(&d_total_weights, sizeof(total_weights_t));
  cudaMemset(d_total_weights, 0, sizeof(total_weights_t));
  cudaMemcpyToSymbol(total_weights, &d_total_weights, sizeof(total_weights_t*));

  printf("\nDetector\n");
  printf("  Position  . . . . . . . . . x = %f, y = %f, z = %f\n", h_detector.center.x, h_detector.center.y, h_detector.center.z);
  printf("  Size (pixels) . . . . . . . x = %d, y = %d\n", h_detector.N.x, h_detector.N.y);
  printf("  Pixel size  . . . . . . . . x = %f, y = %f\n", h_detector.d.x, h_detector.d.y);
}

// populate the region data with the values read from the egsphant file and copy it to the device
void init_regions(uint nreg, medium_t **media, int *media_indices, float *densities) {
  uint size = (nreg + 1) * sizeof(region_data_t);
  region_data_t *h_region_data = (region_data_t*)malloc(size);

  // region 0 is outside of simulation geometry
  h_region_data[0].med = VACUUM;
  h_region_data[0].flags = 0;
  h_region_data[0].rhof = 0.0F;
  h_region_data[0].pcut = 0.0F;
  h_region_data[0].ecut = 0.0F;

  for (uint i = 1; i < nreg + 1; i++) {
    ushort med = media_indices[i - 1] - 1;
    h_region_data[i].med = med;
    if (med == VACUUM) {
      h_region_data[i].flags = 0;
      h_region_data[i].rhof = 0.0F;
      h_region_data[i].pcut = 0.0F;
      h_region_data[i].ecut = 0.0F;
    } 
    else {
      h_region_data[i].flags = f_rayleigh;
      if (densities[i - 1] == 0.0F)
        h_region_data[i].rhof = 1.0F;
      else
        h_region_data[i].rhof = densities[i - 1] / (float)media[med]->rho;
      h_region_data[i].pcut = (float)media[med]->ap;
      h_region_data[i].ecut = (float)media[med]->ae;
    }
  }

  // allocate device memory region data
  cudaMalloc(&d_region_data, size);

  // copy region data to device
  cudaMemcpy(d_region_data, h_region_data, size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(region_data, &d_region_data, sizeof(region_data_t*));

  // free host memory
  free(h_region_data);
}

// read the egsphant file and copy the phantom data to the device
void init_phantom() {
  // read egsphant file
  FILE *f = fopen(egsphant_file, "r");
  if (f == NULL)    printf("Could not open the phantom file \"%s\".\n", egsphant_file);

  int nmed = 0; // number of media
  if (fscanf(f, "%d\n", &nmed) != 1)
    printf("Could not read the number of media form the phantom file \"%s\".\n", egsphant_file);

  // read media names
  string *media_names = new string[nmed];
  for (int i = 0; i < nmed; i++) {
    char mBuf[1024];
    if (!fgets(mBuf, 1024, f))
      printf("Could not read medium name (i = %d) in the phantom file \"%s\".\n", i , egsphant_file);
    media_names[i] = trim(string(mBuf), true);
    if (media_names[i].compare("") == 0)
      printf("Medium name cannot be empty (i = %d) in the phantom file \"%s\".\n", i , egsphant_file);
  }

  // skip next line (contains dummy input)
  char dummyBuf[1024];
  fgets(dummyBuf, 1024, f);

  // read voxel numbers
  int nx, ny, nz;
  if (fscanf(f, "%d %d %d\n", &nx, &ny, &nz) != 3)
    printf("Could not read the voxel numbers in the phantom file \"%s\".\n", egsphant_file);

  // read voxel boundaries
  float *x_bounds = (float*)malloc((nx + 1) * sizeof(float));
  for (int i = 0; i <= nx; i++) {
    if (fscanf(f, "%f", x_bounds + i) != 1)
      printf("Could not read x coordinate of voxel boundary (i = %d) in the phantom file \"%s\".\n", i , egsphant_file);
  }
  float *y_bounds = (float*)malloc((ny + 1) * sizeof(float));
  for (int i = 0; i <= ny; i++) {
    if (fscanf(f, "%f", y_bounds + i) != 1)
      printf("Could not read y coordinate of voxel boundary (i = %d) in the phantom file \"%s\".\n", i , egsphant_file);
  }
  float *z_bounds = (float*)malloc((nz + 1) * sizeof(float));
  for (int i = 0; i <= nz; i++) {
    if (fscanf(f, "%f", z_bounds + i) != 1)
      printf("Could not read z coordinate of voxel boundary (i = %d) in the phantom file \"%s\".\n", i , egsphant_file);
  }

  // read media indices
  int *media_indices = (int*)malloc(nx * ny * nz * sizeof(int));
  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        if (fscanf(f, "%1d", media_indices + x + y * nx + z * nx * ny) != 1)
          printf("Could not read media index (x = %d, y = %d, z = %d) in the phantom file \"%s\".\n", x, y, z , egsphant_file);
      }
    }

    // skip blank line
    char dummyBuf[1024];
    fgets(dummyBuf, 1024, f);
    if (trim(string(dummyBuf), false).compare("") != 0)
      printf("Expected empty line but found \"%s\" (z = %d) in the phantom file \"%s\".\n", dummyBuf, z, egsphant_file);
  }

  // read densities
  float *densities = (float*)malloc(nx * ny * nz * sizeof(float));
  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        if (fscanf(f, "%f", densities + x + y * nx + z * nx * ny) != 1)
          printf("Could not read density (x = %d, y = %d, z = %d) in the phantom file \"%s\".\n", x, y, z , egsphant_file);
      }
    }

    // skip blank line
    char dummyBuf[1024];
    fgets(dummyBuf, 1024, f);
    if (trim(string(dummyBuf), false).compare("") != 0)
      printf("Expected empty line but found \"%s\" (z = %d) in the phantom file \"%s\".\n", dummyBuf, z, egsphant_file);
  }

  fclose(f);

  // copy phantom data to device
  h_phantom.N.x = nx;
  h_phantom.N.y = ny;
  h_phantom.N.z = nz;

  cudaMalloc(&h_phantom.x_bounds, (nx + 1) * sizeof(float));
  cudaMalloc(&h_phantom.y_bounds, (ny + 1) * sizeof(float));
  cudaMalloc(&h_phantom.z_bounds, (nz + 1) * sizeof(float));

  cudaMemcpy(h_phantom.x_bounds, x_bounds, (nx + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(h_phantom.y_bounds, y_bounds, (ny + 1) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(h_phantom.z_bounds, z_bounds, (nz + 1) * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(phantom, &h_phantom, sizeof(phantom));

  free(x_bounds);
  free(y_bounds);
  free(z_bounds);

  // init media
  medium_t **media = init_media(nmed, media_names);

  // init regions
  init_regions(nx * ny * nz, media, media_indices, densities);

  free(media);
  free(media_indices);
  free(densities);
}

// call the above functions to perfrom the initialization
void init(uint seed) {
  init_MTs(seed);
  init_stack();
  init_source();
  init_detector();
  init_phantom();
}

// free all allocated memory on the host and the device
void free_all() {
  cudaFree(d_MT_tables);
  cudaFree(d_MT_params);
  cudaFree(d_MT_statuses);

  cudaFree(d_stack.a);
  cudaFree(d_stack.b);
  cudaFree(d_stack.c);

  cudaFree(d_total_step_counts);
  free(h_total_step_counts);
  cudaFree(d_total_weights);

  for (uchar i = 0; i < SIMULATION_NUM_BLOCKS; i++) {
    for (uchar j = 0; j < NUM_DETECTOR_CAT; j++) {
      cudaFree(d_detector_scores_count[i][j]);
      cudaFree(d_detector_scores_energy[i][j]);
    }
  }

  for (uchar i = 0; i < NUM_DETECTOR_CAT; i++) {
    cudaFree(d_detector_totals_count[i]);
    cudaFree(d_detector_totals_energy[i]);
  }

#ifdef USE_ENERGY_SPECTRUM
  cudaFree(h_source.xi);
  cudaFree(h_source.wi);
  cudaFree(h_source.bin);
#endif

  cudaFree(h_phantom.x_bounds);
  cudaFree(h_phantom.y_bounds);
  cudaFree(h_phantom.z_bounds);

  cudaFree(d_region_data);
  cudaFree(d_ge);
  cudaFree(d_gmfp);
  cudaFree(d_gbr1);
  cudaFree(d_gbr2);
  cudaFree(d_cohe);
  cudaFree(d_pmax);
  cudaFree(d_rayleigh_data);
  cudaFree(d_i_array);


  // list depth counter
#ifdef DO_LIST_DEPTH_COUNT
  cudaFree(d_total_list_depth);
  cudaFree(d_total_num_inner_iterations);
  free(h_total_list_depth);
  free(h_total_num_inner_iterations);
#endif
}

#endif
