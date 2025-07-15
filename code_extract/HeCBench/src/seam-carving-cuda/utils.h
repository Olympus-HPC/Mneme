typedef enum {
    SEAM_CARVER_STANDARD_MODE,
    SEAM_CARVER_UPDATE_MODE,
    SEAM_CARVER_APPROX_MODE
} seam_carver_mode;

typedef struct { int r; int g; int b; } pixel;

int next_pow2(int n){
  int res = 1;
  while(res < n)
    res = res*2;
  return res;
}

uchar4 *build_pixels(const unsigned char *imgv, int w, int h){
  uchar4 *pixels = (uchar4*)malloc(w*h*sizeof(uchar4));
  uchar4 pix;
  for(int i = 0; i < h; i++){
    for(int j = 0; j < w; j++){
      pix.x = imgv[i*3*w + 3*j];
      pix.y = imgv[i*3*w + 3*j + 1];
      pix.z = imgv[i*3*w + 3*j + 2]; 
      pixels[i*w + j] = pix;            
    }
  }
  return pixels;
}

unsigned char *flatten_pixels(uchar4 *pixels, int w, int h, int new_w){
  unsigned char *flattened = (unsigned char*)malloc(3*new_w*h*sizeof(unsigned char));
  for(int i = 0; i < h; i++){
    for(int j = 0; j < new_w; j++){ 
      uchar4 pix = pixels[i*w + j];
      flattened[3*i*new_w + 3*j] = pix.x;
      flattened[3*i*new_w + 3*j + 1] = pix.y;
      flattened[3*i*new_w + 3*j + 2] = pix.z;
    }
  }
  return flattened;
}

