/****************************************************************************
 *
 * output.c, Version 1.0.0 Mon 09 Jan 2012
 *
 * ----------------------------------------------------------------------------
 *
 * CUDA EGS
 * Copyright (C) 2012 CancerCare Manitoba
 *
 * The latest version of CUDA EGS and additional information are available online at 
 * http://www.physics.umanitoba.ca/~elbakri/cuda_egs/ and http://www.lippuner.ca/cuda_egs
 *
 * CUDA EGS is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License as published by the Free Software 
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.                                       
 *                                                                           
 * CUDA EGS is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
 * details.                              
 *                                                                           
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * ----------------------------------------------------------------------------
 *
 *   Contact:
 *
 *   Jonas Lippuner
 *   Email: jonas@lippuner.ca 
 *
 ****************************************************************************/

#ifdef CUDA_EGS

#include "bmp.c"

// get time
void getTime() {
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(charBuffer, CHARLEN, "%I:%M:%S %p on %A, %B %d, %Y", timeinfo);
}

// check output prefix
void checkOutputPrefix(int ret, string output_prefix) {

  // get the absolute path of the output prefix

  string testFile = "test";
  uint i = 0;
  char buf[10];
  FILE *t;

  while (true) {
    sprintf(buf, "%d", i);
    t = fopen((testFile + string(buf)).c_str(), "r");
    // the test file already exists, don't change it
    if (t) {
      fclose(t);
      t = NULL;
      i++;
    }
    // the test file does not exist
    else {
      bool success = true;

      // try creating the file
      t = fopen((testFile + string(buf)).c_str(), "w");

      // file successfully created
      if (t) {
        // try writing to the file
        if (fprintf(t, "test") != 4)
          success = false;
        if (fclose(t))
          success = false;
        if (remove((testFile + string(buf)).c_str()))
          success = false;

      }
      // file could not be created
      else
        success = false;

      if (!success) {
        printf("ERROR (%d): Could not create output files with output prefix \"%s\". Please make sure the specified path exists and is writable.\n", ret, output_prefix.c_str());
        exit(ret);
      }
      else
        break;
    }
  }
}

// create a bitmap file of the detector data with a value of 0 being black and the largest data value being white
void saveBitmap(const char *filePath, float *data, int num) {
  // find highest photon count
  double highestCount = data[0];
  int i;
  for (i = 0; i < num; i++) {
    if (data[i] > highestCount)
      highestCount = data[i];
  }

  // calculate rgb values
  char *rgb = (char*)malloc(3 * num);
  for (i = 0; i < num; i++) {
    char val = char (data[i] / highestCount * 255);
    rgb[3 * i] = val;
    rgb[3 * i + 1] = val;
    rgb[3 * i + 2] = val;
  }

  // write bitmap
  write_bmp(filePath, h_detector.N.x, h_detector.N.y, rgb);

  free(rgb);
}

// write the detector data as into a binary file
void write_output(string baseName, string type, double *data[NUM_DETECTOR_CAT + 1]) {
  for (uchar i = 0; i <= NUM_DETECTOR_CAT; i++) {
    string fname = baseName + type + "_" + categories[i];

    FILE *out = fopen((fname + ".bin").c_str(), "wb");
    fwrite(&h_detector.N, sizeof(uint2), 1, out);

    // convert to float
    uint num_pixels = h_detector.N.x * h_detector.N.y;
    float *data_f = (float*)malloc(num_pixels * sizeof(float));
    for (uint j = 0; j < num_pixels; j++)
      data_f[j] = (float)data[i][j];

    // write data output
    fwrite(data_f, sizeof(float), num_pixels, out);
    fclose(out);

    // write bitmap output
    saveBitmap((fname + ".bmp").c_str(), data_f, num_pixels);

    free(data_f);
  }
}

#endif
