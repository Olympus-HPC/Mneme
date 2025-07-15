__global__ void md ( const box_str* d_box_gpu,
    const FOUR_VECTOR* d_rv_gpu,
    const fp* d_qv_gpu,
    FOUR_VECTOR* d_fv_gpu,
    const fp alpha, 
    int dim_cpu_number_boxes) 
{

  __shared__ FOUR_VECTOR rA_shared[100];
  __shared__ FOUR_VECTOR rB_shared[100];
  __shared__ fp qB_shared[100];

  int bx = blockIdx.x; 
  int tx = threadIdx.x;
  int wtx = tx;

  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
  //  DO FOR THE NUMBER OF BOXES
  //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

  if(bx<dim_cpu_number_boxes){

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  Extract input parameters
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // parameters
    fp a2 = 2*alpha*alpha;

    // home box
    int first_i;
    // (enable the line below only if wanting to use shared memory)

    // nei box
    int pointer;
    int k = 0;
    int first_j;
    int j = 0;
    // (enable the two lines below only if wanting to use shared memory)

    // common
    fp r2;
    fp u2;
    fp vij;
    fp fs;
    fp fxij;
    fp fyij;
    fp fzij;
    THREE_VECTOR d;

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  Home box
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    //----------------------------------------------------------------------------------------------------------------------------------140
    //  Setup parameters
    //----------------------------------------------------------------------------------------------------------------------------------140

    // home box - box parameters
    first_i = d_box_gpu[bx].offset;

    //----------------------------------------------------------------------------------------------------------------------------------140
    //  Copy to shared memory
    //----------------------------------------------------------------------------------------------------------------------------------140

    // (enable the section below only if wanting to use shared memory)
    // home box - shared memory
    while(wtx<NUMBER_PAR_PER_BOX){
      rA_shared[wtx] = d_rv_gpu[first_i+wtx];
      wtx = wtx + NUMBER_THREADS;
    }
    wtx = tx;

    // (enable the section below only if wanting to use shared memory)
    // synchronize threads  - not needed, but just to be safe for now
    __syncthreads();

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  nei box loop
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

    // loop over nei boxes of home box
    for (k=0; k<(1+d_box_gpu[bx].nn); k++){

      //----------------------------------------50
      //  nei box - get pointer to the right box
      //----------------------------------------50

      if(k==0){
        pointer = bx;                          // set first box to be processed to home box
      }
      else{
        pointer = d_box_gpu[bx].nei[k-1].number;              // remaining boxes are nei boxes
      }

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // nei box - box parameters
      first_j = d_box_gpu[pointer].offset;

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Setup parameters
      //----------------------------------------------------------------------------------------------------------------------------------140

      // (enable the section below only if wanting to use shared memory)
      // nei box - shared memory
      while(wtx<NUMBER_PAR_PER_BOX){
        rB_shared[wtx] = d_rv_gpu[first_j+wtx];
        qB_shared[wtx] = d_qv_gpu[first_j+wtx];
        wtx = wtx + NUMBER_THREADS;
      }
      wtx = tx;

      // (enable the section below only if wanting to use shared memory)
      // synchronize threads because in next section each thread accesses data brought in by different threads here
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Calculation
      //----------------------------------------------------------------------------------------------------------------------------------140

      // loop for the number of particles in the home box
      while(wtx<NUMBER_PAR_PER_BOX){

        // loop for the number of particles in the current nei box
        for (j=0; j<NUMBER_PAR_PER_BOX; j++){

          r2 = rA_shared[wtx].v + rB_shared[j].v - DOT(rA_shared[wtx],rB_shared[j]); 
          u2 = a2*r2;
          vij= exp(-u2);
          fs = 2*vij;
          d.x = rA_shared[wtx].x  - rB_shared[j].x;
          fxij=fs*d.x;
          d.y = rA_shared[wtx].y  - rB_shared[j].y;
          fyij=fs*d.y;
          d.z = rA_shared[wtx].z  - rB_shared[j].z;
          fzij=fs*d.z;
          d_fv_gpu[first_i+wtx].v +=  qB_shared[j]*vij;
          d_fv_gpu[first_i+wtx].x +=  qB_shared[j]*fxij;
          d_fv_gpu[first_i+wtx].y +=  qB_shared[j]*fyij;
          d_fv_gpu[first_i+wtx].z +=  qB_shared[j]*fzij;

        }

        // increment work thread index
        wtx = wtx + NUMBER_THREADS;

      }

      // reset work index
      wtx = tx;

      // synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
      __syncthreads();

      //----------------------------------------------------------------------------------------------------------------------------------140
      //  Calculation END
      //----------------------------------------------------------------------------------------------------------------------------------140

    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------160
    //  nei box loop END
    //------------------------------------------------------------------------------------------------------------------------------------------------------160

  }

}
