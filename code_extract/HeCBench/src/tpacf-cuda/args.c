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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "args.h"

extern char *optarg;

void usage(char *name)
{
  printf("Options: <-d data_file_name> <-p data points count> \n");
  printf("         <-r rnd_file_name> <-n rnd_count> <-q random points count> \n");
  printf("         <-o file_name>\n");
  printf("         <-b bins_per_decade> <-l min_angle (degrees)> <-u max angle> [-a] [-j #]\n");

  exit(0);
}

void parse_args(int argc, char **argv, options* args)
{
  int c;

  args->data_name = NULL;
  args->random_name = NULL;
  args->random_count = 0;
  args->ndpoints = 0;
  args->nrpoints = 0;
  args->output_name = NULL;
  args->bins_per_dec = -1;
  args->min_angle = -1.0f;
  args->max_angle = -1.0f;
  args->angle_units = 0;  // degrees
  args->njk = 0;  // by default; can be either 0, or 2,3,..., but not 1

  while ((c = getopt(argc, argv, "d:n:r:p:q:o:j:b:l:u:a")) != EOF)
  {
    switch (c)
    {
      case 'a':
        args->angle_units = 1;  // arcminutes
        break;
      case 'd':
        args->data_name = optarg;
        break;
      case 'r':
        args->random_name = optarg;
        break;
      case 'n':
        args->random_count = atoi(optarg);
        break;
      case 'j':
        args->njk = atoi(optarg);
        break;
      case 'o':
        args->output_name = optarg;
        break;
      case 'p':
        args->ndpoints = atol(optarg);
        break;
      case 'q':
        args->nrpoints = atol(optarg);
        break;
      case 'b':
        args->bins_per_dec = atoi(optarg);
        break;
      case 'l':
        args->min_angle = atof(optarg);
        break;
      case 'u':
        args->max_angle = atof(optarg);
        break;
      default:
        usage(argv[0]);
    }
  }

  if (args->data_name == NULL || args->random_name == NULL ||
      args->random_count < 0 || args->output_name == NULL ||
      args->ndpoints == 0 || args->nrpoints == 0 ||
      args->bins_per_dec < 0 || args->min_angle < 0 ||
      args->max_angle < 0 || args->max_angle <= args->min_angle ||
      args->njk < 0 || args->njk == 1)
    usage(argv[0]);
}
