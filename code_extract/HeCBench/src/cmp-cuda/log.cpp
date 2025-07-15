////////////////////////////////////////////////////////////////////////////////
/**
 * @file log.cpp
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

#include "log.hpp"

#include <iostream>
#include <cstdlib>
#include <string>

////////////////////////////////////////////////////////////////////////////////

enum log_level_t logger::_verbosity_level = FAIL;

////////////////////////////////////////////////////////////////////////////////

#define CASE(c, out) \
  case c: if(logger::verbosity_level() >= level) { \
            out << "[" << #c << "]: " << msg << std::endl;\
          } break

void logger::log(enum log_level_t level, const std::string& msg) {
  switch(level) {
    CASE(WARNING, std::cerr);
    CASE(INFO, std::cout);
    default:
      std::cerr << "[FAIL]: " << msg << std::endl;
      exit(EXIT_FAILURE);
  }
}

////////////////////////////////////////////////////////////////////////////////

void logger::verbosity_level(int level) {
  switch(level) {
    case 0:  logger ::verbosity_level() = FAIL;    break ;
    case 1:  logger ::verbosity_level() = WARNING; break ;
    case 2:  logger ::verbosity_level() = INFO;    break ;
    default: logger ::verbosity_level() = INFO;
  }
}

////////////////////////////////////////////////////////////////////////////////
