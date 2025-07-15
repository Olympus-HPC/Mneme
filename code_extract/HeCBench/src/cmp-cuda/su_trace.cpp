////////////////////////////////////////////////////////////////////////////////
/**
 * @file su.cpp
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

#include "su_trace.hpp"
#include "log.hpp"

#include <cassert>
#include <cstdlib>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////

su_trace::su_trace(int ns): _ns(ns), _data(ns) {
}

////////////////////////////////////////////////////////////////////////////////

bool su_trace::fgettr(std::ifstream& file) {
  assert(file);

  if(file.read((char*)this, SU_HEADER_SIZE).eof())
    return false;

  this->_data.resize(this->_ns);

  if(file.read((char*)this->_data.data(), this->_ns * sizeof(float)).eof()) {
    LOG(FAIL, "Bad input data");
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

void su_trace::fputtr(std::ofstream& file) {
  assert(file);

  file.exceptions(std::ofstream::failbit | std::ofstream::badbit);

  try {
    file.write((char*)this, SU_HEADER_SIZE);
    file.write((char*)this->_data.data(), this->_ns * sizeof(float));
  } catch(const std::ios_base::failure& fail) {
    LOG(FAIL, fail.what());
  }
}

////////////////////////////////////////////////////////////////////////////////

float su_trace::halfoffset() {
  float hx = (float)(this->_gx - this->_sx) * 0.5;
  float hy = (float)(this->_gy - this->_sy) * 0.5;
  return std::sqrt(hx * hx + hy * hy);
}

////////////////////////////////////////////////////////////////////////////////

float su_trace::halfoffset_x() {
  return this->fscalco() * (this->_gx - this->_sx) * 0.5;
}

////////////////////////////////////////////////////////////////////////////////

float su_trace::halfoffset_y() {
  return this->fscalco() * (this->_gy - this->_sy) * 0.5;
}

////////////////////////////////////////////////////////////////////////////////

float su_trace::fscalco() const {
	if (this->_scalco == 0) return 1;
	if (this->_scalco > 0 ) return this->_scalco;

	return 1.0f / this->_scalco;
}

////////////////////////////////////////////////////////////////////////////////

su_trace& su_trace::operator=(const su_trace& other) {
  _tracl =    other.tracl();
  _tracr =    other.tracr();
  _fldr =     other.fldr();
  _tracf =    other.tracf();
  _ep =       other.ep();
  _cdp =      other.cdp();
  _cdpt =     other.cdpt();
  _trid =     other.trid();
  _nvs =      other.nvs();
  _nhs =      other.nhs();
  _duse =     other.duse();
  _offset =   other.offset();
  _gelev =    other.gelev();
  _selev =    other.selev();
  _sdepth =   other.sdepth();
  _gdel =     other.gdel();
  _sdel =     other.sdel();
  _swdep =    other.swdep();
  _gwdep =    other.gwdep();
  _scalel =   other.scalel();
  _sx =       other.sx();
  _sy =       other.sy();
  _gx =       other.gx();
  _gy =       other.gy();
  _counit =   other.counit();
  _wevel =    other.wevel();
  _swevel =   other.swevel();
  _sut =      other.sut();
  _gut =      other.gut();
  _sstat =    other.sstat();
  _gstat =    other.gstat();
  _tstat =    other.tstat();
  _laga =     other.laga();
  _lagb =     other.lagb();
  _delrt =    other.delrt();
  _muts =     other.muts();
  _mute =     other.mute();
  _ns =       other.ns();
  _dt =       other.dt();
  _gain =     other.gain();
  _igc =      other.igc();
  _igi =      other.igi();
  _corr =     other.corr();
  _sfs =      other.sfs();
  _sfe =      other.sfe();
  _slen =     other.slen();
  _styp =     other.styp();
  _stas =     other.stas();
  _stae =     other.stae();
  _tatyp =    other.tatyp();
  _afilf =    other.afilf();
  _afils =    other.afils();
  _nofilf =   other.nofilf();
  _nofils =   other.nofils();
  _lcf =      other.lcf();
  _hcf =      other.hcf();
  _lcs =      other.lcs();
  _hcs =      other.hcs();
  _year =     other.year();
  _day =      other.day();
  _hour =     other.hour();
  _minute =   other.minute();
  _sec =      other.sec();
  _timbas =   other.timbas();
  _trwf =     other.trwf();
  _grnors =   other.grnors();
  _grnofr =   other.grnofr();
  _grnlof =   other.grnlof();
  _gaps =     other.gaps();
  _otrav =    other.otrav();
  _d1 =       other.d1();
  _f1 =       other.f1();
  _d2 =       other.d2();
  _f2 =       other.f2();
  _ungpow =   other.ungpow();
  _unscale =  other.unscale();
  _ntr =      other.ntr();
  _mark =     other.mark();
  _shortpad = other.shortpad();

  this->_data.resize(this->_ns);

  return *(this);
}

////////////////////////////////////////////////////////////////////////////////
