/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   faceDetection.cpp
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Main function for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "image.h"
#include "stdio-wrapper.h"
#include "haar.h"

using namespace std;

int main (int argc, char *argv[]) 
{
  if (argc != 5) {
    printf("Usage: %s <input image file> <classifier information> ", argv[0]);
    printf("<class information> <output image file>\n");
    exit(-1);
  }

  /* detection parameters */
  float scaleFactor = 1.2;
  int minNeighbours = 1;

  printf("-- entering main function --\r\n");
  printf("-- loading image --\r\n");

  MyImage imageObj;
  MyImage *image = &imageObj;

  int flag = readPgm(argv[1], image);
  if (flag == -1)
  {
    printf( "Unable to open input image\n");
    return 1;
  }

  /*return total number of weak classifiers (one node each)*/
  int total_nodes = readTextClassifier(argv[2], argv[3]);
  if (total_nodes == -1) return 1;

  printf("-- loading cascade classifier --\r\n");

  myCascade cascadeObj;
  myCascade *cascade = &cascadeObj;
  MySize minSize = {20, 20};
  MySize maxSize = {0, 0};

  /* classifier properties */
  cascade->n_stages = 25;
  cascade->total_nodes = 2913;
  cascade->orig_window_size.height = 24;
  cascade->orig_window_size.width = 24;

  printf("-- detecting faces --\r\n");
  std::vector<MyRect> result;

  auto start = std::chrono::steady_clock::now();

  result = detectObjects(image, minSize, maxSize, cascade, scaleFactor, minNeighbours, total_nodes);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Object detection time %f (s)\n", time * 1e-9f);

  for(unsigned int i = 0; i < result.size(); i++ )
  {
    MyRect r = result[i];
    drawRectangle(image, r);
  }

  printf("-- saving output --\r\n"); 
  flag = writePgm(argv[4], image); 

  printf("-- image saved --\r\n");

  /* delete image and free classifier */
  releaseTextClassifier();
  freeImage(image);

  return 0;
}
