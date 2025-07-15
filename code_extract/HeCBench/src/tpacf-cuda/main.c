/*
 Illinois Open Source License
 
 University of Illinois/NCSA
 Open Source License
 
 Copyright © 2009,    University of Illinois.  All rights reserved.
 
 Developed by: 
 Innovative Systems Lab
 National Center for Supercomputing Applications
 http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
 
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
 
 * Neither the names of Innovative Systems Lab and National Center for Supercomputing Applications, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
 
 THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 */

#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "args.h"
#include "kernel.h"

#define TDIFF(ts, te) (te.tv_sec - ts.tv_sec + (te.tv_usec - ts.tv_usec) * 1e-6)

#define d2r M_PI/180.0

double *init_bins(int bins_per_dec, float min_angle, float max_angle, int angle_units, int *nbins);
void calculatejkSizes(int njk, int ndPoints, int** jkSizes);
void write_results(long long **DD, long long **RRS, long long **DRS, int njk, int nbins, int bins_per_dec, float min_angle, int random_count, char *fname);

void writeBoundaries(double *binbs);
void doComputeGPU(char* dataName, char* randomNames, int nr, int dataSize, int randomSize, int njk, int* jkSizes, int nBins,
    int zeroBin, long long** DDs, long long** DRs, long long** RRs);
void compileHistograms(long long* DDs, long long* DRs, long long* RRs, long long*** DD, long long*** DR, long long*** RR, options *args);

int main(int argc, char* argv[])
{
  struct timeval T0, T1;

  gettimeofday(&T0, NULL);

  options args;
  parse_args(argc, argv, &args);

  printf("\ndata file: %s \n", args.data_name);
  printf("number of data points: %d\n", args.ndpoints);
  printf("random data files: %s.1-%i\n", args.random_name, args.random_count);
  printf("number of random points: %d \n", args.nrpoints);
  printf("output file: %s \n", args.output_name);
  printf("njk: %i \n", args.njk);
  printf("Min angular distance: %f %s\n", args.min_angle, (args.angle_units) ? "arcmin" : "degrees");
  printf("Max angular distance: %f %s\n", args.max_angle, (args.angle_units) ? "arcmin" : "degrees");
  printf("Bins per dec: %i\n", args.bins_per_dec);
  int nbins = floor(args.bins_per_dec * (log10(args.max_angle) - log10(args.min_angle)));
  printf("Total bins  : %i\n\n", nbins);

  args.ndpoints -= (args.ndpoints % 4);
  args.nrpoints -= (args.nrpoints % 4);

  int* jksizes = NULL;
  if(args.njk < 1) args.njk = 1;  
  // CPU version uses 0 jackknives to indicate no jackknife resampling; because of the different manner in which jackknife 
  // resampling is handled that would lead to  errors in the GPU version.          
  calculatejkSizes(args.njk, args.ndpoints, &jksizes);

  int tempnbins = 0;
  double* binbs = NULL;
  binbs = init_bins(args.bins_per_dec, args.min_angle, args.max_angle, args.angle_units, &tempnbins);
  writeBoundaries(binbs);

  int zeroBin = 0;
  int i;
  for(i=0; i<NUMBINS-1; i++) {
    if(0.0f > binbs[i]) {
      zeroBin = i;
      break;
    }
  }

  long long* DDs = NULL;
  long long* DRs = NULL;
  long long* RRs = NULL;

  doComputeGPU(args.data_name, args.random_name, args.random_count, args.ndpoints, args.nrpoints, args.njk, jksizes, NUMBINS, zeroBin, &DDs, &DRs, &RRs);

  long long** DD = NULL;
  long long** DR = NULL;
  long long** RR = NULL;

  compileHistograms(DDs, DRs, RRs, &DD, &DR, &RR, &args);

  gettimeofday(&T1, NULL);
  float timetemp = TDIFF(T0, T1);
  printf("DONE! after %f\n", timetemp);

  if(args.njk == 1) args.njk = 0; // # of jackknives should only be 1 if jackknife resampling is not to be used.

  write_results(DD, RR, DR, args.njk, tempnbins, args.bins_per_dec, args.min_angle, args.random_count, args.output_name);

  return 0;
}

void calculatejkSizes(int njk, int ndPoints, int** jkSizes) {
  *jkSizes = (int*)malloc(njk*sizeof(int));
  int pointsperjk = ndPoints / njk;
  pointsperjk -= (pointsperjk % 4);
  int i;
  for(i=0; i<njk-1; i++) {
    (*jkSizes)[i] = pointsperjk;
  }
  (*jkSizes)[njk-1] = ndPoints - (njk - 1)*pointsperjk;
}

double *init_bins(int bins_per_dec, float min_angle, float max_angle, int angle_units, int *nbins)
{
  int k;

  *nbins = floor(bins_per_dec * (log10(max_angle) - log10(min_angle)));

  // memory for bin boundaries
  double *binb = (double *)malloc(31*sizeof(double));
  if (binb == NULL)
  {
    fprintf(stderr, "Unable to allocate memory for bin boundaries.\n");
    exit(0);
  }

  printf("\nBin boundaries:\n");
  printf("#  degrees          arcminutes        dot_product\n");

  int binoffset = 30 - (*nbins);

  for(k = 0; k < (*nbins)+1; k++)
  {
    double bb = pow(10, log10(min_angle) + (k)*1.0/bins_per_dec);
    binb[k + binoffset] = cos(bb / ((angle_units) ? 60.0 : 1.0) * d2r);

    printf("%i %.15f %.15f ", k, (angle_units) ? bb/60.0f : bb, (angle_units) ? bb : bb * 60);
    printf("%.15f\n", binb[k + binoffset]);
  }

  for(k = 0; k<binoffset; k++) {
    binb[k] = -5.0;
  }

  binb[30] = -5.0;

  return binb;
}

void write_results(long long **DD, long long **RRS, long long **DRS, int njk, int nbins, int bins_per_dec, float min_angle, int random_count, char *fname)
{
  // compute and output results
  FILE *outfile;
  int k, l;

  if ((outfile = fopen(fname, "w")) == NULL)
  {
    fprintf(stderr, "Unable to open output file %s for writing, assuming stdout\n", fname);
    outfile = stdout;
  }

  fprintf(outfile, "boundary DD DRS RRS w ");

  for (l = 1; l < njk+1; l++)
    fprintf(outfile, "DD[%i] DRS[%i] RRS w[%i]  ", l, l, l);

  fprintf(outfile, "\n");

  double bb = pow(10, log10(min_angle) + 0*1.0/bins_per_dec);
  fprintf(outfile, "<%f %lld %lld %lld -\n", bb, DD[0][0], DRS[0][0], RRS[0][0]);

  int binoffset = 30 - nbins;

  for (k = 1; k < nbins+1; k++)
  {
    double bb = pow(10, log10(min_angle) + k*1.0/bins_per_dec);
    fprintf(outfile, "%f ", bb);

    for (l = 0; l < njk+1; l++)
    {
      if (RRS[0][k+binoffset] != 0)
      {
        double norm_dd = (double)DD[l][k+binoffset]; // (args.ndpoints * (args.ndpoints - 1) / 2.0);
        double norm_rrs = (double)RRS[0][k+binoffset]; // (args.random_count * args.nrpoints * (args.nrpoints - 1) / 2.0);
        double norm_drs = (double)DRS[l][k+binoffset]; // (args.random_count * args.ndpoints * args.nrpoints);
        double w = (random_count * norm_dd - norm_drs) / norm_rrs + 1.0;

        fprintf(outfile, "%lld %lld %lld %f ", DD[l][k+binoffset], DRS[l][k+binoffset], RRS[0][k+binoffset], w);
      }
      else
        fprintf(outfile, "%lld %lld %lld - ", DD[l][k+binoffset], DRS[l][k+binoffset], RRS[0][k+binoffset]);
    }

    fprintf(outfile, "\n");
  }

  bb = pow(10, log10(min_angle) + nbins*1.0/bins_per_dec);
  fprintf(outfile, ">%f %lld %lld %lld\n", bb, DD[0][nbins+1], DRS[0][nbins+1], RRS[0][nbins+1]);

  fclose(outfile);
}
