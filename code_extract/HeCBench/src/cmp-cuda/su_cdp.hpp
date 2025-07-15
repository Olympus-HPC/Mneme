////////////////////////////////////////////////////////////////////////////////
/**
 * @file su_cdp.hpp
 * @date 2017-03-05
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

#ifndef SU_CDP_HPP
#define SU_CDP_HPP

////////////////////////////////////////////////////////////////////////////////

#include "su_trace.hpp"

#include <vector>

////////////////////////////////////////////////////////////////////////////////

class su_cdp {
  private:
    std::vector<su_trace> _traces;
    int _cdp;

  public:
    su_cdp();

    void push_back(const su_trace& trace);
    inline size_t size() { return _traces.size(); }

    inline const std::vector<su_trace>& traces() const { return _traces; }

    inline int& cdp() { return _cdp; }

    inline bool operator==(const su_cdp& other) const {
      return this->_traces.size() == other.traces().size();
    }

    inline bool operator<(const su_cdp& other) const {
      return _traces.size() > other.traces().size();
    }
};

////////////////////////////////////////////////////////////////////////////////

#endif /*! SU_CDP_HPP */

////////////////////////////////////////////////////////////////////////////////
