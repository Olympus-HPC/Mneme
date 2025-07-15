/* ################# wrappers ################### */

void compute_costs(
    int current_w, int w, int h, 
    uchar4* d_pixels, 
    short* d_costs_left,
    short* d_costs_up,
    short* d_costs_right)
{
#ifndef COMPUTE_COSTS_FULL

  dim3 threads_per_block(COSTS_BLOCKSIZE_X, COSTS_BLOCKSIZE_Y);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/(threads_per_block.x-2) + 1;
  num_blocks.y = (h-1)/(threads_per_block.y-1) + 1;    
  compute_costs_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs_left, d_costs_up, d_costs_right, 
      w, h, current_w);

#else

  dim3 threads_per_block(COSTS_BLOCKSIZE_X, COSTS_BLOCKSIZE_Y);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/threads_per_block.x + 1;
  num_blocks.y = (h-1)/threads_per_block.y + 1;    
  compute_costs_full_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_costs_left, d_costs_up, d_costs_right,
      w, h, current_w);

#endif
}

void compute_M(
    int current_w, int w, int h, 
    int* d_M, 
    short* d_costs_left,
    short* d_costs_up,
    short* d_costs_right)
{
#if !defined(COMPUTE_M_SINGLE) && !defined(COMPUTE_M_ITERATE)  

  if(current_w <= 256){
    dim3 threads_per_block(current_w);
    dim3 num_blocks(1);
    compute_M_kernel_small<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w);
  }
  else{
    dim3 threads_per_block(COMPUTE_M_BLOCKSIZE_X, 1);

    dim3 num_blocks;
    num_blocks.x = (current_w-1)/threads_per_block.x + 1;
    num_blocks.y = 1;

    dim3 num_blocks2;
    num_blocks2.x = (current_w-COMPUTE_M_BLOCKSIZE_X-1)/threads_per_block.x + 1; 
    num_blocks2.y = 1;  

    int num_iterations;
    num_iterations = (h-1)/(COMPUTE_M_BLOCKSIZE_X/2 - 1) + 1;

    int base_row = 0;
    for(int i = 0; i < num_iterations; i++){
      compute_M_kernel_step1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      compute_M_kernel_step2<<<num_blocks2, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, base_row);
      base_row = base_row + (COMPUTE_M_BLOCKSIZE_X/2) - 1;    
    }
  }

#endif
#ifdef COMPUTE_M_SINGLE    

  dim3 threads_per_block(min(256, next_pow2(current_w)), 1);   
  dim3 num_blocks(1,1);
  int num_el = (current_w-1)/threads_per_block.x + 1;
  compute_M_kernel_single<<<num_blocks, threads_per_block, 2*current_w*sizeof(int)>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, h, current_w, num_el);

#else
#ifdef COMPUTE_M_ITERATE

  dim3 threads_per_block(COMPUTE_M_BLOCKSIZE_X, 1);   
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/threads_per_block.x + 1;
  num_blocks.y = 1;
  compute_M_kernel_iterate0<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w);
  for(int row = 1; row < h; row++){
    compute_M_kernel_iterate1<<<num_blocks, threads_per_block>>>(d_costs_left, d_costs_up, d_costs_right, d_M, w, current_w, row);
  }

#endif
#endif
}

void find_min_index(
    int current_w,
    int* d_indices_ref,
    int* d_indices,
    int* reduce_row)
{
  //set the reference index array
  cudaMemcpy(d_indices, d_indices_ref, current_w*sizeof(int), cudaMemcpyDeviceToDevice);

  dim3 threads_per_block(REDUCE_BLOCKSIZE_X, 1);   
  dim3 num_blocks;
  num_blocks.y = 1; 
  int reduce_num_elements = current_w;
  do{
    num_blocks.x = (reduce_num_elements-1)/(threads_per_block.x*REDUCE_ELEMENTS_PER_THREAD) + 1;
    min_reduce<<<num_blocks, threads_per_block>>>(reduce_row, d_indices, reduce_num_elements); 
    reduce_num_elements = num_blocks.x;          
  }while(num_blocks.x > 1);    
}

void find_seam(
    int current_w, int w, int h, 
    int *d_M,
    int *d_indices,
    int *d_seam )
{
  find_seam_kernel<<<1, 1>>>(d_M, d_indices, d_seam, w, h, current_w);
}

void remove_seam(
    int current_w, int w, int h, 
    int *d_M,
    uchar4 *d_pixels,
    uchar4 *d_pixels_swap,
    int *d_seam )
{
  dim3 threads_per_block(REMOVE_BLOCKSIZE_X, REMOVE_BLOCKSIZE_Y);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/threads_per_block.x + 1;
  num_blocks.y = (h-1)/threads_per_block.y + 1;
  remove_seam_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_pixels_swap, d_seam, w, h, current_w);
}

void update_costs(
    int current_w, int w, int h, 
    int *d_M,
    uchar4 *d_pixels,
    short *d_costs_left,
    short *d_costs_up,
    short *d_costs_right,
    short *d_costs_swap_left,
    short *d_costs_swap_up,
    short *d_costs_swap_right,
    int *d_seam )
{
  dim3 threads_per_block(UPDATE_BLOCKSIZE_X, UPDATE_BLOCKSIZE_Y);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/threads_per_block.x + 1;
  num_blocks.y = (h-1)/threads_per_block.y + 1;    
  update_costs_kernel<<<num_blocks, threads_per_block>>>(
    d_pixels, d_costs_left, d_costs_up, d_costs_right, 
    d_costs_swap_left, d_costs_swap_up, d_costs_swap_right,
    d_seam, w, h, current_w);
}

void approx_setup(
    int current_w, int w, int h, 
    uchar4 *d_pixels,
    int *d_index_map,
    int *d_offset_map,
    int *d_M )
{
  dim3 threads_per_block(APPROX_SETUP_BLOCKSIZE_X, APPROX_SETUP_BLOCKSIZE_Y);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/(threads_per_block.x-4) + 1;
  num_blocks.y = (h-2)/(threads_per_block.y-1) + 1;    
  approx_setup_kernel<<<num_blocks, threads_per_block>>>(d_pixels, d_index_map, d_offset_map, d_M, w, h, current_w);
}

void approx_M(
    int current_w, int w, int h, 
    int *d_offset_map,
    int *d_M )
{
  dim3 threads_per_block(APPROX_M_BLOCKSIZE_X, 1);
  dim3 num_blocks;
  num_blocks.x = (current_w-1)/threads_per_block.x + 1;
  num_blocks.y = h/2;  
  int step = 1;
  while(num_blocks.y > 0){
    approx_M_kernel<<<num_blocks, threads_per_block>>>(d_offset_map, d_M, w, h, current_w, step);
    num_blocks.y = (int)num_blocks.y/2;
    step = step*2;
  }
}

void approx_seam(
    int w, int h, 
    int *d_index_map,
    int *d_indices,
    int *d_seam )
{
  approx_seam_kernel<<<1, 1>>>(d_index_map, d_indices, d_seam, w, h);
}
