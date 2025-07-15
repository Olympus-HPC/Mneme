#include "ccs.h"

void mergebcl(struct bicl *bicluster, int nw,int old,int n,int D,double score)
{
  int i;
  for (i = 0; i < n; i++) {
    if (bicluster[old].data[i]=='1' && bicluster[nw].data[i]=='0') {
      bicluster[nw].data[i]='1';
      bicluster[nw].datacount+=1;
    }
  }

  for (i = 0; i < D; i++) {
    if (bicluster[old].sample[i]=='1' &&  bicluster[nw].sample[i]=='0') {
      bicluster[nw].sample[i]='1';
      bicluster[nw].samplecount+=1;
    }
  } 

  bicluster[old].score=1.0; //Delete the merged bicluster
  bicluster[nw].score=score; //update merged bicluster score
  printf("Bicluster %d and %d are merged. New score=%lf ...\n",nw,old,score); 
}
