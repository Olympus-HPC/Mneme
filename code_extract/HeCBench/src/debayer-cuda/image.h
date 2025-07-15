#ifndef IMAGE_H
#define IMAGE_H

typedef unsigned char uchar;

enum pattern_t {
  RGGB = 0,
  GRBG = 1,
  GBRG = 2,
  BGGR = 3
};

enum {
  ADDRESS_CLAMP = 0, //repeat border
  ADDRESS_ZERO = 1, //returns 0
  ADDRESS_REFLECT_BORDER_EXCLUSIVE = 2, //reflects at boundary and will not duplicate boundary elements
  ADDRESS_REFLECT_BORDER_INCLUSIVE = 3, //reflects at boundary and will duplicate boundary elements,
  ADDRESS_NOOP = 4 //programmer guarantees no reflection necessary
};

//coordinate is c, r for compatibility with climage and CUDA
INLINE __device__
uint2 tex2D(const int rows, const int cols, const int _c, const int _r,
            const uint sample_method) {
  int c = _c;
  int r = _r;
  if (sample_method == ADDRESS_REFLECT_BORDER_EXCLUSIVE) {
    c = c < 0 ? -c : c;
    c = c >= cols ? cols - (c - cols) - 2 : c;
    r = r < 0 ? -r : r;
    r = r >= rows ? rows - (r - rows) - 2 : r;
  } else if (sample_method == ADDRESS_CLAMP) {
    c = c < 0 ? 0 : c;
    c = c > cols - 1 ? cols - 1 : c;
    r = r < 0 ? 0 : r;
    r = r > rows - 1 ? rows - 1 : r;
  } else if (sample_method == ADDRESS_REFLECT_BORDER_INCLUSIVE) {
    c = c < 0 ? -c - 1 : c;
    c = c >= cols ? cols - (c - cols) - 1 : c;
    r = r < 0 ? -r - 1 : r;
    r = r >= rows ? rows - (r - rows) - 1 : r;
  } else if (sample_method == ADDRESS_ZERO) {
  } else if (sample_method == ADDRESS_NOOP) {
  } else {
    assert(false);
  }
  assert_val(r >= 0 && r < rows, r);
  assert_val(c >= 0 && c < cols, c);
  uint2 result; 
  result.x = r;
  result.y = c;
  return result;
}

INLINE uchar* image_line_at_(uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch_p, const uint r) {
  assert_val(r >= 0 && r < im_rows, r);
  (void) im_cols;
  return im_p + r * image_pitch_p;
}
#define image_line_at(PixelT, im_p, im_rows, im_cols, image_pitch, r) ((PixelT *) image_line_at_((uchar *) (im_p), (im_rows), (im_cols), (image_pitch), (r)))

INLINE __device__
uchar* image_pixel_at_(uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch_p, 
                       const uint r, const uint c, const uint sizeof_pixel) {
  assert_val(r >= 0 && r < im_rows, r);
  assert_val(c >= 0 && c < im_cols, c);
  return im_p + r * image_pitch_p + c * sizeof_pixel;
}
#define image_pixel_at(PixelT, im_p, im_rows, im_cols, image_pitch, r, c) (*((PixelT *) image_pixel_at_((uchar *)(im_p), (im_rows), (im_cols), (image_pitch), (r), (c), sizeof(PixelT))))

INLINE __device__
uchar* image_tex2D_(uchar *im_p, const uint im_rows, const uint im_cols, const uint image_pitch, 
                    const int r, const int c, const uint sizeof_pixel, const uint sample_method) {
  const uint2 p2 = tex2D((int) im_rows, (int) im_cols, c, r, sample_method);
  return image_pixel_at_(im_p, im_rows, im_cols, image_pitch, p2.x, p2.y, sizeof_pixel);
}

#define image_tex2D(PixelT, im_p, im_rows, im_cols, image_pitch, r, c, sample_method) \
  (((sample_method) == ADDRESS_ZERO) & (((r) < 0) | ((r) >= (im_rows)) | ((c) < 0) | ((c) >= (im_cols))) ? 0 : \
   *(PixelT *) image_tex2D_((uchar *)(im_p), (im_rows), (im_cols), (image_pitch), (r), (c), sizeof(PixelT), (sample_method)))

#ifndef OUTPUT_CHANNELS
#define OUTPUT_CHANNELS 3
#endif

#ifndef ALPHA_VALUE
#define ALPHA_VALUE UCHAR_MAX
#endif

#ifndef PIXELT
#define PIXELT uchar
#endif

#ifndef RGBPIXELBASET
#define RGBPIXELBASET PIXELT
#endif

#ifndef RGBPIXELT
#define RGBPIXELT PASTE(RGBPIXELBASET, OUTPUT_CHANNELS)
#endif
#ifndef LDSPIXELT
#define LDSPIXELT int
#endif

typedef PIXELT PixelT;
typedef RGBPIXELBASET RGBPixelBaseT;
typedef RGBPIXELT RGBPixelT;
typedef LDSPIXELT LDSPixelT;// for LDS's, having this large enough to prevent bank conflicts make's a large difference

#define kernel_size 5
#define tile_rows 5
#define tile_cols 32
#define apron_rows (tile_rows + kernel_size - 1)
#define apron_cols (tile_cols + kernel_size - 1)

#define half_ksize  (kernel_size/2)
#define shalf_ksize ((int) half_ksize)
#define half_ksize_rem (kernel_size - half_ksize)
#define n_apron_fill_tasks (apron_rows * apron_cols)
#define n_tile_pixels  (tile_rows * tile_cols)

#define pixel_at(type, basename, r, c) image_pixel_at(type, PASTE_2(basename, _p), height, width, PASTE_2(basename, _pitch), (r), (c))
#define tex2D_at(type, basename, r, c) image_tex2D(type, PASTE_2(basename, _p), height, width, PASTE_2(basename, _pitch), (r), (c), ADDRESS_REFLECT_BORDER_EXCLUSIVE)
#define apron_pixel(_t_r, _t_c) apron[(_t_r) * apron_cols + (_t_c)]

#define output_pixel_cast(x) PASTE3(convert_,RGBPIXELBASET,_sat)((x))

#endif
