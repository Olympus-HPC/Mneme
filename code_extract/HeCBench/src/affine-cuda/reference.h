#define X_SIZE 512
#define Y_SIZE 512
#define PI     3.14159265359f
#define WHITE  (short)(1)
#define BLACK  (short)(0)

// reference implementation for verification
void affine_reference(const unsigned short *src, unsigned short *dst) 
{
  for (int y = 0; y < Y_SIZE; y++) {
    for (int x = 0; x < X_SIZE; x++) {
      const float    lx_rot   = 30.0f;
      const float    ly_rot   = 0.0f; 
      const float    lx_expan = 0.5f;
      const float    ly_expan = 0.5f; 
      int      lx_move  = 0;
      int      ly_move  = 0;
      float    affine[2][2];   // coefficients
      float    i_affine[2][2];
      float    beta[2];
      float    i_beta[2];
      float    det;
      float    x_new, y_new;
      float    x_frac, y_frac;
      float    gray_new;
      int      m, n;

      unsigned short    output_buffer[X_SIZE];


      // forward affine transformation 
      affine[0][0] = lx_expan * std::cos((float)(lx_rot*PI/180.0f));
      affine[0][1] = ly_expan * std::sin((float)(ly_rot*PI/180.0f));
      affine[1][0] = lx_expan * std::sin((float)(lx_rot*PI/180.0f));
      affine[1][1] = ly_expan * std::cos((float)(ly_rot*PI/180.0f));
      beta[0]      = lx_move;
      beta[1]      = ly_move;

      // determination of inverse affine transformation
      det = (affine[0][0] * affine[1][1]) - (affine[0][1] * affine[1][0]);
      if (det == 0.0f)
      {
        i_affine[0][0]   = 1.0f;
        i_affine[0][1]   = 0.0f;
        i_affine[1][0]   = 0.0f;
        i_affine[1][1]   = 1.0f;
        i_beta[0]        = -beta[0];
        i_beta[1]        = -beta[1];
      } 
      else 
      {
        i_affine[0][0]   =  affine[1][1]/det;
        i_affine[0][1]   = -affine[0][1]/det;
        i_affine[1][0]   = -affine[1][0]/det;
        i_affine[1][1]   =  affine[0][0]/det;
        i_beta[0]        = -i_affine[0][0]*beta[0]-i_affine[0][1]*beta[1];
        i_beta[1]        = -i_affine[1][0]*beta[0]-i_affine[1][1]*beta[1];
      }

      // Output image generation by inverse affine transformation and bilinear transformation

      x_new    = i_beta[0] + i_affine[0][0]*(x-X_SIZE/2.0f) + i_affine[0][1]*(y-Y_SIZE/2.0f) + X_SIZE/2.0f;
      y_new    = i_beta[1] + i_affine[1][0]*(x-X_SIZE/2.0f) + i_affine[1][1]*(y-Y_SIZE/2.0f) + Y_SIZE/2.0f;

      m        = (int)std::floor(x_new);
      n        = (int)std::floor(y_new);

      x_frac   = x_new - m;
      y_frac   = y_new - n;

      if ((m >= 0) && (m + 1 < X_SIZE) && (n >= 0) && (n+1 < Y_SIZE))
      {
        gray_new = (1.0f - y_frac) * ((1.0f - x_frac) * (src[(n * X_SIZE) + m])       + x_frac * (src[(n * X_SIZE) + m + 1])) + 
          y_frac  * ((1.0f - x_frac) * (src[((n + 1) * X_SIZE) + m]) + x_frac * (src[((n + 1) * X_SIZE) + m + 1]));

        output_buffer[x] = (unsigned short)gray_new;
      } 
      else if (((m + 1 == X_SIZE) && (n >= 0) && (n < Y_SIZE)) || ((n + 1 == Y_SIZE) && (m >= 0) && (m < X_SIZE))) 
      {
        output_buffer[x] = src[(n * X_SIZE) + m];
      } 
      else 
      {
        output_buffer[x] = WHITE;
      }

      dst[(y * X_SIZE)+x] = output_buffer[x];
    }
  }
}

