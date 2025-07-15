////////////////////////////////////////////////////////////////////////////////
/**
 * @file utils.hpp
 * @date 2017-03-04
 * @author Tiago Lobato Gimenes    (tlgimenes@gmail.com)
 *
 * @copyright
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILS_HPP
#define UTILS_HPP

////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////

//#define DOUBLE

////////////////////////////////////////////////////////////////////////////////

#ifdef DOUBLE
#define real double
#else
#define real float
#endif /* Real */

#define MAX_W 16

#define EPSILON 1e-13

#define FACTOR 1e6

#define NTHREADS 128

void
h_init_c(int nc, real *c, real inc, real c0) ;
void
h_init_half(int ttraces, real* scalco, real* gx, real* gy, real* sx, real* sy, real* h) ;
void
h_compute_semblances(const real* __restrict h, 
                     const real* __restrict c, 
                     const real* __restrict samples, 
                     real* __restrict num,
                     real* __restrict stt,
                     int t_id0, 
                     int t_idf,
                     real _idt,
                     real _dt,
                     int _tau,
                     int _w,
                     int nc,
                     int ns);
void
h_redux_semblances(const real* __restrict num, 
                   const real* __restrict  stt, 
                   int*  __restrict ctr, 
                   real* __restrict str, 
                   real* __restrict stk,
                   const int nc, 
                   const int cdp_id,
                   const int ns) ;

#endif /*! UTILS_HPP */

////////////////////////////////////////////////////////////////////////////////
