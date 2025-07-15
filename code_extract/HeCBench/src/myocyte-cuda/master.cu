//=====================================================================
//  MAIN FUNCTION
//=====================================================================

double master(fp timeinst,
    fp* initvalu,
    fp* parameter,
    fp* finavalu,
    fp* com,

    fp* d_initvalu,
    fp* d_finavalu,
    fp* d_params,
    fp* d_com)
{

  //=====================================================================
  //  VARIABLES
  //=====================================================================

  // counters
  int i;

  // offset pointers
  int initvalu_offset_ecc;
  int initvalu_offset_Dyad;
  int initvalu_offset_SL;
  int initvalu_offset_Cyt;

  dim3 threads;
  dim3 blocks;

  //=====================================================================
  //  execute ECC&CAM kernel - it runs ECC and CAMs in parallel
  //=====================================================================

  int d_initvalu_mem;
  d_initvalu_mem = EQUATIONS * sizeof(fp);
  int d_finavalu_mem;
  d_finavalu_mem = EQUATIONS * sizeof(fp);
  int d_params_mem;
  d_params_mem = PARAMETERS * sizeof(fp);
  int d_com_mem;
  d_com_mem = 3 * sizeof(fp);

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
    printf("initvalu %d %f\n", i, initvalu[i]);
  for (int i = 0; i < PARAMETERS; i++)
    printf("params %d %f\n", i, parameter[i]);
  printf("\n");
#endif

  cudaMemcpy(d_initvalu, initvalu, d_initvalu_mem, cudaMemcpyHostToDevice);
  cudaMemcpy(d_params, parameter, d_params_mem, cudaMemcpyHostToDevice);

  threads.x = NUMBER_THREADS;
  threads.y = 1;
  blocks.x = 2;
  blocks.y = 1;

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  kernel<<<blocks, threads>>>(
      timeinst,
      d_initvalu,
      d_finavalu,
      d_params,
      d_com);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  cudaMemcpy(finavalu, d_finavalu, d_finavalu_mem, cudaMemcpyDeviceToHost);
  cudaMemcpy(com, d_com, d_com_mem, cudaMemcpyDeviceToHost);

#ifdef DEBUG
  for (int i = 0; i < EQUATIONS; i++)
    printf("finavalu %d %f\n", i, finavalu[i]);
  for (int i = 0; i < 3; i++)
    printf("%f ", com[i]);
  printf("\n");

#endif

  //=====================================================================
  //  FINAL KERNEL
  //=====================================================================

  initvalu_offset_ecc = 0;
  initvalu_offset_Dyad = 46;
  initvalu_offset_SL = 61;
  initvalu_offset_Cyt = 76;

  kernel_fin(
      initvalu,
      initvalu_offset_ecc,
      initvalu_offset_Dyad,
      initvalu_offset_SL,
      initvalu_offset_Cyt,
      parameter,
      finavalu,
      com[0],
      com[1],
      com[2]);

  //=====================================================================
  //  COMPENSATION FOR NANs and INFs
  //=====================================================================

  for(i=0; i<EQUATIONS; i++){
    if (isnan(finavalu[i])){ 
      finavalu[i] = 0.0001;                        // for NAN set rate of change to 0.0001
    }
    else if (isinf(finavalu[i])){ 
      finavalu[i] = 0.0001;                        // for INF set rate of change to 0.0001
    }
  }

  return time;
}
