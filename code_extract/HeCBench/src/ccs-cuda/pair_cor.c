#include "ccs.h"

struct pair_r comput_r(char *sample,int wid,int k,int i,int D,struct gn *gene)
{
  float sx, sxx, sy, sxy, syy;
  float sx_n, sxx_n, sy_n, sxy_n, syy_n;
  int j;
  struct pair_r rval;

  rval.r=0.0;
  rval.n_r=0.0; 

  sx = 0; sxx = 0; sy = 0; sxy = 0; syy = 0;

  sx_n = 0; sxx_n = 0;  sy_n = 0;  sxy_n = 0; syy_n = 0;

  for (j = 0; j < D; j++) {
    if(sample[j]=='1')
      sx +=  gene[k].x[j];
    else
      sx_n +=  gene[k].x[j];
  }
  sx /= wid;

  sx_n/=(D-wid);

  for (j = 0; j < D; j++) {
    if(sample[j]=='1')
      sxx += (sx-gene[k].x[j]) * (sx-gene[k].x[j]);
    else
      sxx_n += (sx_n-gene[k].x[j]) * (sx_n-gene[k].x[j]);

  }
  sxx = ( float)sqrt(sxx);
  sxx_n = ( float)sqrt(sxx_n);

  for (j = 0; j < D; j++) {
    if(sample[j]=='1')
      sy +=  gene[i].x[j];
    else
      sy_n +=  gene[i].x[j];
  }

  sy /= wid; 
  sy_n /= (D-wid); 

  for (j = 0; j < D; j++)
  {
    if(sample[j]=='1') {
      sxy += (sx - gene[k].x[j]) * (sy - gene[i].x[j]);
      syy += (sy - gene[i].x[j]) * (sy - gene[i].x[j]);
    }
    else {
      sxy_n += (sx_n - gene[k].x[j]) * (sy_n - gene[i].x[j]);
      syy_n += (sy_n - gene[i].x[j]) * (sy_n - gene[i].x[j]);
    }
  }

  syy = ( float)sqrt(syy);

  syy_n = ( float)sqrt(syy_n);

  rval.r =  fabsf(sxy/(sxx * syy));

  rval.n_r =  fabsf(sxy_n/(sxx_n * syy_n));

  return rval;
}


