////////////////////////////////////////////////////////////////////////////////
/**
 * @file log.hpp
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

#ifndef LOG_HPP
#define LOG_HPP

////////////////////////////////////////////////////////////////////////////////

#include <string>

////////////////////////////////////////////////////////////////////////////////

enum log_level_t {
  FAIL,
  WARNING,
  INFO
};

////////////////////////////////////////////////////////////////////////////////

class logger {
  private:
    static enum log_level_t _verbosity_level;

  public:
    static void log(enum log_level_t level, const std::string& msg);

    static void verbosity_level(int level);
    static enum log_level_t& verbosity_level() { return logger::_verbosity_level; }
};

////////////////////////////////////////////////////////////////////////////////

#define LOG(level, msg) \
  logger::log(level, std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " + msg)

////////////////////////////////////////////////////////////////////////////////

#endif /*! LOG_HPP */

////////////////////////////////////////////////////////////////////////////////
