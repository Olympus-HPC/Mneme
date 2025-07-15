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

#ifndef MODEL_IO_CU
#define MODEL_IO_CU

#define d2r M_PI/180.0 
#define r2d 180.0/M_PI
#define r2am 60.0*180.0/M_PI

int readdatafile(char *fname, cartesian data, int npoints)
{
  FILE *infile;
  int lcount = 0;
  float ra, dec;
  double x, y, z;
  char buf[256];

  if ((infile = fopen(fname, "r")) == NULL)
  {
    fprintf(stderr, "Unable to open data file %s for reading\n", fname);
    return lcount;
  }

  for (lcount = 0; lcount < npoints; lcount++)
  {
    if (fgets(buf, 256, infile) == NULL)
      break;

    if (buf[0] == '#') { --lcount;  continue; } // skip comment line line

    if (sscanf(buf, "%f %f %lf %lf %lf", &ra, &dec, &x, &y, &z) != 5)
    {
      // data conversion
      double rarad = d2r * ra;
      double decrad = d2r * dec;
      double cd = cos(decrad);

      x = cos(rarad) * cd;
      y = sin(rarad) * cd;
      z = sin(decrad);
    }

    data.x[lcount] = x;
    data.y[lcount] = y;
    data.z[lcount] = z;
    data.jk[lcount] = 0;
  }

  fclose(infile);

  return lcount;
}

void assign_jk(struct cartesian data, int npoints, int njk)
{
  int i;

  if (njk < 2) return;

  int nppjk = 1 + npoints / njk; // number of points per jk

  for (i = 0; i < npoints; i++)
  {
    data.jk[i] = 1 + i / nppjk;  // assign it to a particular jackknife
  }
}

#endif
