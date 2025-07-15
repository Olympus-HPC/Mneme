template<int R>
void reference(
    float *__restrict in,
    float *__restrict out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  #pragma omp parallel for collapse(2)
  for (int idx = 0; idx < w; idx++)
    for (int idy = 0; idy < h; idy++) {

      int id = idy*w + idx;
      float I = in[id];
      float res = 0;
      float normalization = 0;

      // window centered at the coordinate (idx, idy)
      #pragma unroll
      for(int i = -R; i <= R; i++)
      #pragma unroll
        for(int j = -R; j <= R; j++) {

          int idk = idx+i;
          int idl = idy+j;

          // mirror edges
          if( idk < 0) idk = -idk;
          if( idl < 0) idl = -idl;
          if( idk > w - 1) idk = w - 1 - i;
          if( idl > h - 1) idl = h - 1 - j;

          int id_w = idl*w + idk;
          float I_w = in[id_w];

          // range kernel for smoothing differences in intensities
          float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

          // spatial (or domain) kernel for smoothing differences in coordinates
          float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                          (2.f * variance_spatial);

          // the weight is assigned using the spatial closeness (using the spatial kernel) 
          // and the intensity difference (using the range kernel)
          float weight = a_square * expf(spatial + range);
              normalization += weight;
              res += (I_w * weight);
        }
      out[id] = res/normalization;
    }
}


