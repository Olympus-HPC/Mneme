// srad kernel
__global__ void srad2(const  fp d_lambda, 
                    const int d_Nr, 
                    const int d_Nc, 
                    const long d_Ne, 
                    const int *d_iN, 
                    const int *d_iS, 
                    const int *d_jE, 
                    const int *d_jW,
                    const fp *d_dN, 
                    const fp *d_dS, 
                    const fp *d_dE, 
                    const fp *d_dW, 
                    const fp *d_c, 
                    fp *d_I)
{

  // indexes
    int bx = blockIdx.x;                  // get current horizontal block index
  int tx = threadIdx.x;                   // get current horizontal thread index
  int ei = bx*NUMBER_THREADS+tx;          // more threads than actual elements !!!
  int row;                                // column, x position
  int col;                                // row, y position

  // variables
  fp d_cN,d_cS,d_cW,d_cE;
  fp d_D;

  // figure out row/col location in new matrix
  row = (ei+1) % d_Nr - 1;                // (0-n) row
  col = (ei+1) / d_Nr + 1 - 1;            // (0-n) column
  if((ei+1) % d_Nr == 0){
    row = d_Nr - 1;
    col = col - 1;
  }

  if(ei<d_Ne){                            // make sure that only threads matching jobs run

    // diffusion coefficent
    d_cN = d_c[ei];                       // north diffusion coefficient
    d_cS = d_c[d_iS[row] + d_Nr*col];     // south diffusion coefficient
    d_cW = d_c[ei];                       // west diffusion coefficient
    d_cE = d_c[row + d_Nr * d_jE[col]];   // east diffusion coefficient

    // divergence (equ 58)
    d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];// divergence

    // image update (equ 61) (every element of IMAGE)
    d_I[ei] = d_I[ei] + (fp)0.25*d_lambda*d_D;// updates image (based on input time step and divergence)

  }

}
