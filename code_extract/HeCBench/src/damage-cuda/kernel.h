/* Calculate the damage of each node.
 *
 * nlist - An (n, local_size) array containing the neighbour lists,
 *         a value of -1 corresponds to a broken bond.
 * family - An array of the initial number of neighbours for each node.
 * n_neigh - An array of the number of neighbours (particles bound) for each node.
 * damage - An array of the damage for each node. 
 * local_cache - local (local_size) array to store the bond breakages.
 */
__global__ void damage_of_node(
  const int n,
  const int *__restrict__ nlist,
  const int *__restrict__ family,
        int *__restrict__ n_neigh,
     double *__restrict__ damage)
{
  extern __shared__ int local_cache[];

  const int local_id = threadIdx.x;
  const int local_size = blockDim.x;
  const int nid = blockIdx.x;
  const int global_id = nid * local_size + local_id;
  if (global_id >= n) return;

  //Copy values into local memory 
  local_cache[local_id] = nlist[global_id] != -1 ? 1 : 0; 

  //Wait for all threads
  __syncthreads();

  for (int i = local_size/2; i > 0; i /= 2) {
    if(local_id < i){
      local_cache[local_id] += local_cache[local_id + i];
    } 
    //Wait for all threads
    __syncthreads();
  }

  if (local_id == 0) {
    // Update damage and n_neigh
    int neighbours = local_cache[0];
    n_neigh[nid] = neighbours;
    damage[nid] = 1.0 - (double) neighbours / (double) (family[nid]);
  }
}
