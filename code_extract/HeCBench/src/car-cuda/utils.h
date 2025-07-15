typedef struct {
  int output_dim_b;
  int output_dim_c;
  int output_dim_h;
  int output_dim_w;
  int kernel_size;
  int image_w;
  int image_h;
} params;

// accesses to flattened 4D arrays
#define img(b,c,y,x) \
        img[(b)*dim_c*img_w*img_h + (c)*img_w*img_h + (y)*img_w + (x)]

#define offsets_h(b,c,y,x) \
        offsets_h[(b)*k_size*k_size*dim_w*dim_h + (c)*(dim_w*dim_h) + (y)*dim_w + (x)]

#define offsets_v(b,c,y,x) \
        offsets_v[(b)*k_size*k_size*dim_w*dim_h + (c)*(dim_w*dim_h) + (y)*dim_w + (x)]

#define kernels(b,c,y,x) \
        kernels[(b)*k_size*k_size*dim_w*dim_h + (c)*(dim_w*dim_h) + (y)*dim_w + (x)]

#define output(b,c,y,x) \
        output[(b)*dim_c*dim_w*dim_h + (c)*dim_w*dim_h + (y)*dim_w + (x)]

// rng
double LCG_random_double(unsigned long long * seed)
{
  const unsigned long long m = 9223372036854775808ULL; // 2^63
  const unsigned long long a = 2806196910506780709ULL;
  const unsigned long long c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}


