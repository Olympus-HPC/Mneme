__global__ void solver(
    int number_bands, int ngpown, int ncouls,
    const int *__restrict__ inv_igp_index,
    const int *__restrict__ indinv,
    const dataType *__restrict__ wx_array,
    const CustomComplex<dataType> *__restrict__ wtilde_array,
    const CustomComplex<dataType> *__restrict__ aqsmtemp,
    const CustomComplex<dataType> *__restrict__ aqsntemp,
    const CustomComplex<dataType> *__restrict__ I_eps_array,
    const dataType *__restrict__ vcoul,
    dataType *__restrict__ achtemp_re,
    dataType *__restrict__ achtemp_im) 
{
  dataType achtemp_re_loc[nend - nstart], achtemp_im_loc[nend - nstart];
  for (int iw = nstart; iw < nend; ++iw) {
    achtemp_re_loc[iw] = 0.00;
    achtemp_im_loc[iw] = 0.00;
  }

  for (int n1 = blockIdx.x; n1 < number_bands; n1 += gridDim.x) // 512 iterations
  {
    for (int my_igp = blockIdx.y; my_igp < ngpown; my_igp += gridDim.y) // 1634 iterations
    {
      int indigp = inv_igp_index[my_igp];
      int igp = indinv[indigp];
      CustomComplex<dataType> sch_store1 =
          CustomComplex_conj(aqsmtemp(n1, igp)) * aqsntemp(n1, igp) * 0.5 *
          vcoul[igp];

      for (int ig = threadIdx.x; ig < ncouls; ig += blockDim.x) {
        #pragma unroll
        for (int iw = nstart; iw < nend; ++iw) // 3 iterations
        {
          CustomComplex<dataType> wdiff =
              wx_array[iw] - wtilde_array(my_igp, ig);
          CustomComplex<dataType> delw =
              wtilde_array(my_igp, ig) * CustomComplex_conj(wdiff) *
              (1 / CustomComplex_real((wdiff * CustomComplex_conj(wdiff))));
          CustomComplex<dataType> sch_array =
              delw * I_eps_array(my_igp, ig) * sch_store1;

          achtemp_re_loc[iw] += CustomComplex_real(sch_array);
          achtemp_im_loc[iw] += CustomComplex_imag(sch_array);
        }
      }
    } // ngpown
  }   // number_bands

  // Add the final results here
  for (int iw = nstart; iw < nend; ++iw) {
    atomicAdd(&achtemp_re[iw], achtemp_re_loc[iw]);
    atomicAdd(&achtemp_im[iw], achtemp_im_loc[iw]);
  }
}
