// statistical kernel
__global__ void extract(const  long d_Ne, fp *d_I)
{ 

  // indexes
  int bx = blockIdx.x;                      // get current horizontal block index
  int tx = threadIdx.x;                     // get current horizontal thread index
  int ei = (bx*NUMBER_THREADS)+tx;          // unique thread id, more threads than actual elements !!!

  // copy input to output & log uncompress
  if(ei<d_Ne){                              // do only for the number of elements, omit extra threads

    d_I[ei] = exp(d_I[ei]/(fp)255);             // exponentiate input IMAGE and copy to output image

  }

}
