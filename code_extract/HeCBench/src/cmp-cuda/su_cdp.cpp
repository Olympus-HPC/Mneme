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

#include "su_cdp.hpp"

#include "log.hpp"

#include <climits>

////////////////////////////////////////////////////////////////////////////////

su_cdp::su_cdp() : _cdp(INT_MIN), _traces(0) {

}

////////////////////////////////////////////////////////////////////////////////

void su_cdp::push_back(const su_trace& trace) {
  if(_cdp == INT_MIN) _cdp = trace.cdp();
  else if(_cdp != trace.cdp()) LOG(FAIL, "CDPs do not match");

  _cdp = trace.cdp();
  _traces.push_back(trace);
}

////////////////////////////////////////////////////////////////////////////////
