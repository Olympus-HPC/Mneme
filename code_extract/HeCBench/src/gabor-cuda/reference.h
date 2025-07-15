double* generateGaborKernelHost(
  const unsigned int height,
  const unsigned int width,
  const unsigned int par_T,
  const double par_L,
  const double theta)
{
  const double sx = (double)par_T / (2.0*sqrt(2.0*log(2.0)));
  const double sy = par_L * sx;
  const double sx_2 = sx*sx;
  const double sy_2 = sy*sy;
  const double fx = 1.0 / (double)par_T;
  const double ctheta = cos(theta);
  const double stheta = sin(theta);
  const double center_y = (double)height / 2.0;
  const double center_x = (double)width / 2.0;
  const double scale = 1.0/(2.0*M_PI*sx*sy);

  double *gabor_spatial = (double*) malloc (height * width * sizeof(double));

  for (unsigned int y = 0; y < height; y++)
  {
    double centered_y = (double)y - center_y;
    for (unsigned int x = 0; x < width; x++)
    {
      double centered_x = (double)x - center_x;
      double u = ctheta * centered_x - stheta * centered_y;
      double v = ctheta * centered_y + stheta * centered_x;
      *(gabor_spatial + y*width + x) = scale * exp(-0.5*(u*u/sx_2 + v*v/sy_2)) * cos(2.0*M_PI*fx*u);
    }
  }
  return gabor_spatial;
}

