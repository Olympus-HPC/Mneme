#include "ccs.h"

double between_bicluster_correlation(struct gn *gene,struct bicl *bicluster, int nw,int old,int n,int D,double thr)
{
  int i,l;
  float score;
  char *g,*x;
  struct pair_r rval;
  int tc,dif,un;
  int total=0,mintotal=0;

  x = (char *)calloc(D,sizeof(char));
  g = (char *)calloc(n,sizeof(char));

  for (i=0;i<n;i++) {
    if (bicluster[nw].data[i]=='1' || bicluster[old].data[i]=='1') {
      g[i]='1';
    }
    else
      g[i]='0';
  }

  tc=0;

  for (i=0;i<D;i++) {
    if (bicluster[nw].sample[i]=='1' || bicluster[old].sample[i]=='1') {
      x[i] = '1';
      tc++;
    } 
    else
      x[i] = '0';
  }

  //Compute number of correlation for other samples//

  dif=0;
  un=0;
  for(i=0;i<n;i++) {
    if(g[i]=='1')
    {
      for(l=i+1;l<n;l++) {
        if(g[l]=='1')
        {
          rval=comput_r(x,tc, i, l, D, gene);
          if(rval.r>thr)
            total++; //number of correlation for combined bicluster sample
          if(rval.n_r>thr)
            mintotal++; //number of correlation for samples not in the combined bicluster

          if(rval.r>thr && rval.n_r>thr)
            dif++;
          if (rval.r>thr || rval.n_r>thr)
            un++;

        }
      }
    }
  }

  if (un>0 && total>mintotal)
    score=(float)dif/un;   
  else 
    score=1.0; 

  free(x);
  free(g);

  return(score);
}
