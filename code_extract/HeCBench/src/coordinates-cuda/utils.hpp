/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#include <math.h>

constexpr double DEGREE_TO_RADIAN = M_PI / 180.0;
constexpr double RADIAN_TO_DEGREE = 180.0 / M_PI;

constexpr double EARTH_RADIUS_KM                = 6371.0;
constexpr double EARTH_CIRCUMFERENCE_EQUATOR_KM = 40000.0;
constexpr double EARTH_CIRCUMFERENCE_KM_PER_DEGREE = EARTH_CIRCUMFERENCE_EQUATOR_KM / 360.0;

/**
 * @brief A generic 2D vector type.
 *
 * This is the base type used in cuspatial for both Longitude/Latitude (LonLat) coordinate pairs and
 * Cartesian (X/Y) coordinate pairs. For LonLat pairs, the `x` member represents Longitude, and `y`
 * represents Latitude.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) vec_2d {
  using value_type = T;
  value_type x;
  value_type y;
};

/**
 * @brief A geographical Longitude/Latitude (LonLat) coordinate pair
 *
 * `x` is the longitude coordinate, `y` is the latitude coordinate.
 *
 * @tparam T the base type for the coordinates
 */
template <typename T>
struct alignas(2 * sizeof(T)) lonlat_2d : vec_2d<T> {
};

/**
 * @brief A Cartesian (x/y) coordinate pair.
 *
 * @tparam T the base type for the coordinates.
 */
template <typename T>
struct alignas(2 * sizeof(T)) cartesian_2d : vec_2d<T> {
};

/**
 * @brief Compare two 2D vectors for equality.
 */
template <typename T>
bool operator==(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
{
  //return (lhs.x == rhs.x) && (lhs.y == rhs.y);
  return fabs(lhs.x - rhs.x) < 1e-3 && fabs(lhs.y - rhs.y) < 1e-3;
}

/**
 * @brief Element-wise addition of two 2D vectors.
 */
template <typename T>
vec_2d<T> HOST_DEVICE operator+(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x + b.x, a.y + b.y};
}

/**
 * @brief Element-wise subtraction of two 2D vectors.
 */
template <typename T>
vec_2d<T> HOST_DEVICE operator-(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return vec_2d<T>{a.x - b.x, a.y - b.y};
}

/**
 * @brief Scale a 2D vector by a factor @p r.
 */
template <typename T>
vec_2d<T> HOST_DEVICE operator*(vec_2d<T> vec, T const& r)
{
  return vec_2d<T>{vec.x * r, vec.y * r};
}

/**
 * @brief Scale a 2d vector by ratio @p r.
 */
template <typename T>
vec_2d<T> HOST_DEVICE operator*(T const& r, vec_2d<T> vec)
{
  return vec * r;
}

/**
 * @brief Compute dot product of two 2D vectors.
 */
template <typename T>
T HOST_DEVICE dot(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.x + a.y * b.y;
}

/**
 * @brief Compute 2D determinant of a 2x2 matrix with column vectors @p a and @p b.
 */
template <typename T>
T HOST_DEVICE det(vec_2d<T> const& a, vec_2d<T> const& b)
{
  return a.x * b.y - a.y * b.x;
}

template <typename T>
HOST_DEVICE inline T midpoint(T a, T b)
{
  return (a + b) / (T)2;
}

template <typename T>
HOST_DEVICE inline T lon_to_x(T lon, T lat)
{
  return lon * EARTH_CIRCUMFERENCE_KM_PER_DEGREE * cos(lat * DEGREE_TO_RADIAN);
};

template <typename T>
HOST_DEVICE inline T lat_to_y(T lat)
{
  return lat * EARTH_CIRCUMFERENCE_KM_PER_DEGREE;
};

template <typename T>
struct to_cartesian_functor {
  to_cartesian_functor(lonlat_2d<T> origin) : _origin(origin) {}

  cartesian_2d<T> HOST_DEVICE operator()(lonlat_2d<T> loc)
  {
    cartesian_2d<T> t;
    t.x = lon_to_x(_origin.x - loc.x, midpoint(loc.y, _origin.y));
    t.y = lat_to_y(_origin.y - loc.y);
    return t;
  }

 private:
   lonlat_2d<T> _origin{};
};

#endif
