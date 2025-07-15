void reference (
    float *__restrict entropy,
    const char*__restrict val, 
    int height, int width)
{
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      // value of matrix element ranges from 0 inclusive to 16 exclusive
      char count[16];
      for (int i = 0; i < 16; i++) count[i] = 0;

      int total = 0;

      // 5x5 window
      for(int dx = -2; dx <= 2; dx++) {
        for(int dy = -2; dy <= 2; dy++) {
          int xx = x + dx,
              yy = y + dy;

          if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
            count[val[yy * width + xx]]++;
            total++;
          }
        }
      }

      if (total < 1) total = 1;

      float s = 0;
      for(int k = 0; k < 16; k++) {
        float p = (float)count[k] / total;
        s -= p * log2f(p);
      }

      entropy[y * width + x] = s;
    }
}

