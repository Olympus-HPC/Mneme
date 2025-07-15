
/**
 *
 *  @file cuThomasBatch.cu
 *
 *  @brief cuThomasBatch kernel implementaion.
 *
 *  cuThomasBatch is a software package provided by
 *  Barcelona Supercomputing Center - Centro Nacional de Supercomputacion
 *
 *  @author Ivan Martinez-Perez ivan.martinez@bsc.es
 *  @author Pedro Valero-Lara   pedro.valero@bsc.es
 *
 **/

/**
 *
 *  @ingroup cuThomasBatch
 *  
 *  Solve a set of Tridiagonal linear systems:
 *
 *      A_ix_i = RHS_i, for all i = 0, ..., N
 *
 *      N = BATCHCOUNT
 *
 *  where A is a MxM tridiagonal matrix:
 *
 *      A_i = [ D_i[0]     U_i[1]    .        .    .          .       
 *              L_i[0]     D_i[1]    U_i[2]   .    .          .          
 *              .          L_i[1]    D_i[2]   .    .          .     
 *              .          .         L_i[2]   .    .          U_i[M-1] 
 *              .          .         .        .    L_i[M-2]   D_i[M-1] ]
 *
 *  Note that the elements of the inputs must be interleaved by following the
 *  next pattern for N (BATCHCOUNT) tridiagonal systems and M elements each:
 *
 *      D_0[0], D_1[0], ..., D_N[0], ..., D_0[M-1], D_1[M-1], ..., D_N[M-1]
 *
**/

/**
 *  
 *  @param[in]
 *  L           double *.
 *              L is a pointer to the lower-diagonal vector
 *          
 *  @param[in]
 *  D           double *.
 *              D is a pointer to the diagonal vector
 *
 *  @param[in,out]
 *  U           double *.
 *              U is a pointer to the uper-diagonal vector
 *
 *  @param[in,out]
 *  RHS         double *.    
 *              RHS is a pointer to the Right Hand Side vector
 *   
 *   
 *  @param[in]
 *  M           int.
 *              M specifies the number of elemets of the systems 
 *
 *  @param[in]
 *  BATCHCOUNT  int.
 *              BATCHCOUNT specifies to number of systems to be procesed
 **/
#include "cuThomasBatch.h"

__global__ void cuThomasBatch(
            const double *L, const double *D, double *U, double *RHS,
            const int M,
            const int BATCHCOUNT
    ) {

        int tid = threadIdx.x + blockDim.x*blockIdx.x;

        if(tid < BATCHCOUNT) {

            int first = tid;
            int last  = BATCHCOUNT*(M-1)+tid;

            U[first] /= D[first];
            RHS[first] /= D[first];

            for (int i = first + BATCHCOUNT; i < last; i+=BATCHCOUNT) {
                U[i] /= D[i] - L[i] * U[i-BATCHCOUNT];
                RHS[i] = ( RHS[i] - L[i] * RHS[i-BATCHCOUNT] ) / 
							( D[i] - L[i] * U[i-BATCHCOUNT] );
            }

            RHS[last] = ( RHS[last] - L[last] * RHS[last-BATCHCOUNT] ) / 
							( D[last] - L[last] * U[last-BATCHCOUNT] );

            for (int i = last-BATCHCOUNT; i >= first; i-=BATCHCOUNT) {
                RHS[i] -= U[i] * RHS[i+BATCHCOUNT];
            }
       }
        
}
