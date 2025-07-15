/**
 * Author:  Florian Stock, Technische Universität Darmstadt,
 * Embedded Systems & Applications Group 2018
 * License: Apache 2.0 (see attachached File)
 */
#ifndef DATATYPES_H
#define DATATYPES_H

typedef struct Mat44 {
  double data[4][4];
} Mat44;

typedef struct Mat33 {
  double data[3][3];
} Mat33;

typedef struct Mat13 {
  double data[3];
} Mat13;

typedef struct Vec5 {
  double data[5];
} Vec5;

typedef struct Point2d {
  double x,y;
} Point2d;

typedef struct ImageSize {
  int32_t width, height;
} ImageSize;

typedef struct PointCloud2 {
  int32_t height;
  int32_t width;
  int32_t point_step;
  float* data;
} PointCloud2;


typedef struct PointsImage {
  // arrays of size image_heigt*image_width
  float* intensity;
  float* distance;
  float* min_height;
  float* max_height;
  int32_t max_y;
  int32_t min_y;
  int32_t image_height;
  int32_t image_width;
} PointsImage;

#endif
