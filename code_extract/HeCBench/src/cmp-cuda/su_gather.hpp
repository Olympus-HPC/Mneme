////////////////////////////////////////////////////////////////////////////////
/**
 * @file su_gather.hpp
 * @date 2017-03-06
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

#ifndef SU_GATHER_HPP
#define SU_GATHER_HPP

////////////////////////////////////////////////////////////////////////////////

#include "su_cdp.hpp"

#include "utils.hpp"

#include <map>

////////////////////////////////////////////////////////////////////////////////

class su_gather {
  private:
    std::vector<su_cdp> _cdps; // traces grouped by cdp, sorted by traces size
    int _ns;                   // total number of samples
    int _ntrs;
    int _nos;                  // total number of semblances
    int _ttraces;              // total number of traces

  public:
    su_gather(std::string& bin_file_path, int aph, int nc);

    inline int ns()      { return _ns;                }
    inline int ntrs()    { return _ntrs;              }
    inline int nos()     { return _nos;               }
    inline int ttraces() { return _ttraces;           }
    inline int ncdps()   { return _cdps.size();       }

    inline const su_cdp& operator[](int i) const { return _cdps[i]; }
    inline const std::vector<su_cdp>& operator()() const { return _cdps; }

    void linearize(int* &ntraces_by_cdp_id ,real* & samples, real &dt, real *&gx, real *&gy, real *&sx, real *&sy, real *&scalco, int nc);
};

////////////////////////////////////////////////////////////////////////////////

#endif /*! SU_GATHER_HPP */

////////////////////////////////////////////////////////////////////////////////
