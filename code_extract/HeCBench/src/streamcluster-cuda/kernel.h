__global__
void compute_cost(
    const Point_Struct *__restrict__ p_d_acc,       
    const float *__restrict__ coord_d_acc,
          float *__restrict__ work_mem_d_acc,      
    const int *__restrict__ center_table_d_acc,
    char *__restrict__ switch_membership_d_acc,      
    const int num,
    const int dim,
    const long x,
    const int K)
{  
  extern __shared__ float coord_s_acc[]; 
  /* block ID and global thread ID */
  const int local_id = threadIdx.x; 
  const int thread_id = blockDim.x*blockIdx.x+local_id;

  if(thread_id<num){
    // coordinate mapping of point[x] to shared mem
    if(local_id == 0)
      for(int i=0; i<dim; i++){ 
        coord_s_acc[i] = coord_d_acc[i*num + x];
      }
    __syncthreads();

    // cost between this point and point[x]: euclidean distance multiplied by weight
    float x_cost = 0.0;
    for(int i=0; i<dim; i++)
      x_cost += (coord_d_acc[(i*num)+thread_id]-coord_s_acc[i]) * (coord_d_acc[(i*num)+thread_id]-coord_s_acc[i]);
    x_cost = x_cost * p_d_acc[thread_id].weight;

    float current_cost = p_d_acc[thread_id].cost;

    int base = thread_id*(K+1);   
    // if computed cost is less then original (it saves), mark it as to reassign    
    if ( x_cost < current_cost ){
      switch_membership_d_acc[thread_id] = '1';
      int addr_1 = base + K;
      work_mem_d_acc[addr_1] = x_cost - current_cost;
    }
    // if computed cost is larger, save the difference
    else {
      int assign = p_d_acc[thread_id].assign;
      int addr_2 = base + center_table_d_acc[assign];
      work_mem_d_acc[addr_2] += current_cost - x_cost;
    }
  }
}
