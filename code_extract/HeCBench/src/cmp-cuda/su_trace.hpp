///////////////////////////////////////////////////////////////////////////////
/**
 * @file su.hpp
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

#ifndef SU_HPP
#define SU_HPP

////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////

class su_trace;

////////////////////////////////////////////////////////////////////////////////

#define SU_HEADER_SIZE ((unsigned long)&(((su_trace*)0)->_data))

////////////////////////////////////////////////////////////////////////////////

class su_trace {
  private:
    int _tracl;
    int _tracr;
    int _fldr;
    int _tracf;
    int _ep;
    int _cdp;
    int _cdpt;
    short _trid;
    short _nvs;
    short _nhs;
    short _duse;
    int _offset;
    int _gelev;
    int _selev;
    int _sdepth;
    int _gdel;
    int _sdel;
    int _swdep;
    int _gwdep;
    short _scalel;
    short _scalco;
    int _sx;
    int _sy;
    int _gx;
    int _gy;
    short _counit;
    short _wevel;
    short _swevel;
    short _sut;
    short _gut;
    short _sstat;
    short _gstat;
    short _tstat;
    short _laga;
    short _lagb;
    short _delrt;
    short _muts;
    short _mute;
    unsigned short _ns;
    unsigned short _dt;
    short _gain;
    short _igc;
    short _igi;
    short _corr;
    short _sfs;
    short _sfe;
    short _slen;
    short _styp;
    short _stas;
    short _stae;
    short _tatyp;
    short _afilf;
    short _afils;
    short _nofilf;
    short _nofils;
    short _lcf;
    short _hcf;
    short _lcs;
    short _hcs;
    short _year;
    short _day;
    short _hour;
    short _minute;
    short _sec;
    short _timbas;
    short _trwf;
    short _grnors;
    short _grnofr;
    short _grnlof;
    short _gaps;
    short _otrav;
    float _d1;
    float _f1;
    float _d2;
    float _f2;
    float _ungpow;
    float _unscale;
    int _ntr;
    short _mark;
    short _shortpad;
    short _unass[14];
    std::vector<float> _data;

  public:
    su_trace(int ns = 0);

    bool fgettr(std::ifstream& bin_file);
    void fputtr(std::ofstream& bin_file);

    float source_x();
    float source_y();

    float receiver_x();
    float receiver_y();

    float halfoffset();
    float halfoffset_x();
    float halfoffset_y();

    float fscalco() const;
    inline short scalco() const { return _scalco; }

    su_trace& operator=(const su_trace& other);

    std::vector<float>& data() { return _data; }
    const std::vector<float>& data() const { return _data; }

    inline int  & tracl()      { return _tracl;    }
    inline int  & tracr()      { return _tracr;    }
    inline int  & fldr()       { return _fldr;     }
    inline int  & tracf()      { return _tracf;    }
    inline int  & ep()         { return _ep;       }
    inline int  & cdp()        { return _cdp;      }
    inline int  & cdpt()       { return _cdpt;     }
    inline short& trid()       { return _trid;     }
    inline short& nvs()        { return _nvs;      }
    inline short& nhs()        { return _nhs;      }
    inline short& duse()       { return _duse;     }
    inline int  & offset()     { return _offset;   }
    inline int  & gelev()      { return _gelev;    }
    inline int  & selev()      { return _selev;    }
    inline int  & sdepth()     { return _sdepth;   }
    inline int  & gdel()       { return _gdel;     }
    inline int  & sdel()       { return _sdel;     }
    inline int  & swdep()      { return _swdep;    }
    inline int  & gwdep()      { return _gwdep;    }
    inline short& scalel()     { return _scalel;   }
    inline int  & sx()         { return _sx;       }
    inline int  & sy()         { return _sy;       }
    inline int  & gx()         { return _gx;       }
    inline int  & gy()         { return _gy;       }
    inline short& counit()     { return _counit;   }
    inline short& wevel()      { return _wevel;    }
    inline short& swevel()     { return _swevel;   }
    inline short& sut()        { return _sut;      }
    inline short& gut()        { return _gut;      }
    inline short& sstat()      { return _sstat;    }
    inline short& gstat()      { return _gstat;    }
    inline short& tstat()      { return _tstat;    }
    inline short& laga()       { return _laga;     }
    inline short& lagb()       { return _lagb;     }
    inline short& delrt()      { return _delrt;    }
    inline short& muts()       { return _muts;     }
    inline short& mute()       { return _mute;     }
    inline unsigned short& ns()         { return _ns;       }
    inline unsigned short& dt()         { return _dt;       }
    inline short& gain()       { return _gain;     }
    inline short& igc()        { return _igc;      }
    inline short& igi()        { return _igi;      }
    inline short& corr()       { return _corr;     }
    inline short& sfs()        { return _sfs;      }
    inline short& sfe()        { return _sfe;      }
    inline short& slen()       { return _slen;     }
    inline short& styp()       { return _styp;     }
    inline short& stas()       { return _stas;     }
    inline short& stae()       { return _stae;     }
    inline short& tatyp()      { return _tatyp;    }
    inline short& afilf()      { return _afilf;    }
    inline short& afils()      { return _afils;    }
    inline short& nofilf()     { return _nofilf;   }
    inline short& nofils()     { return _nofils;   }
    inline short& lcf()        { return _lcf;      }
    inline short& hcf()        { return _hcf;      }
    inline short& lcs()        { return _lcs;      }
    inline short& hcs()        { return _hcs;      }
    inline short& year()       { return _year;     }
    inline short& day()        { return _day;      }
    inline short& hour()       { return _hour;     }
    inline short& minute()     { return _minute;   }
    inline short& sec()        { return _sec;      }
    inline short& timbas()     { return _timbas;   }
    inline short& trwf()       { return _trwf;     }
    inline short& grnors()     { return _grnors;   }
    inline short& grnofr()     { return _grnofr;   }
    inline short& grnlof()     { return _grnlof;   }
    inline short& gaps()       { return _gaps;     }
    inline short& otrav()      { return _otrav;    }
    inline float& d1()         { return _d1;       }
    inline float& f1()         { return _f1;       }
    inline float& d2()         { return _d2;       }
    inline float& f2()         { return _f2;       }
    inline float& ungpow()     { return _ungpow;   }
    inline float& unscale()    { return _unscale;  }
    inline int  & ntr()        { return _ntr;      }
    inline short& mark()       { return _mark;     }
    inline short& shortpad()   { return _shortpad; }

    inline const int  & tracl()    const { return _tracl;    }
    inline const int  & tracr()    const { return _tracr;    }
    inline const int  & fldr()     const { return _fldr;     }
    inline const int  & tracf()    const { return _tracf;    }
    inline const int  & ep()       const { return _ep;       }
    inline const int  & cdp()      const { return _cdp;      }
    inline const int  & cdpt()     const { return _cdpt;     }
    inline const short& trid()     const { return _trid;     }
    inline const short& nvs()      const { return _nvs;      }
    inline const short& nhs()      const { return _nhs;      }
    inline const short& duse()     const { return _duse;     }
    inline const int  & offset()   const { return _offset;   }
    inline const int  & gelev()    const { return _gelev;    }
    inline const int  & selev()    const { return _selev;    }
    inline const int  & sdepth()   const { return _sdepth;   }
    inline const int  & gdel()     const { return _gdel;     }
    inline const int  & sdel()     const { return _sdel;     }
    inline const int  & swdep()    const { return _swdep;    }
    inline const int  & gwdep()    const { return _gwdep;    }
    inline const short& scalel()   const { return _scalel;   }
    inline const int  & sx()       const { return _sx;       }
    inline const int  & sy()       const { return _sy;       }
    inline const int  & gx()       const { return _gx;       }
    inline const int  & gy()       const { return _gy;       }
    inline const short& counit()   const { return _counit;   }
    inline const short& wevel()    const { return _wevel;    }
    inline const short& swevel()   const { return _swevel;   }
    inline const short& sut()      const { return _sut;      }
    inline const short& gut()      const { return _gut;      }
    inline const short& sstat()    const { return _sstat;    }
    inline const short& gstat()    const { return _gstat;    }
    inline const short& tstat()    const { return _tstat;    }
    inline const short& laga()     const { return _laga;     }
    inline const short& lagb()     const { return _lagb;     }
    inline const short& delrt()    const { return _delrt;    }
    inline const short& muts()     const { return _muts;     }
    inline const short& mute()     const { return _mute;     }
    inline const unsigned short& ns()       const { return _ns;       }
    inline const unsigned short& dt()       const { return _dt;       }
    inline const short& gain()     const { return _gain;     }
    inline const short& igc()      const { return _igc;      }
    inline const short& igi()      const { return _igi;      }
    inline const short& corr()     const { return _corr;     }
    inline const short& sfs()      const { return _sfs;      }
    inline const short& sfe()      const { return _sfe;      }
    inline const short& slen()     const { return _slen;     }
    inline const short& styp()     const { return _styp;     }
    inline const short& stas()     const { return _stas;     }
    inline const short& stae()     const { return _stae;     }
    inline const short& tatyp()    const { return _tatyp;    }
    inline const short& afilf()    const { return _afilf;    }
    inline const short& afils()    const { return _afils;    }
    inline const short& nofilf()   const { return _nofilf;   }
    inline const short& nofils()   const { return _nofils;   }
    inline const short& lcf()      const { return _lcf;      }
    inline const short& hcf()      const { return _hcf;      }
    inline const short& lcs()      const { return _lcs;      }
    inline const short& hcs()      const { return _hcs;      }
    inline const short& year()     const { return _year;     }
    inline const short& day()      const { return _day;      }
    inline const short& hour()     const { return _hour;     }
    inline const short& minute()   const { return _minute;   }
    inline const short& sec()      const { return _sec;      }
    inline const short& timbas()   const { return _timbas;   }
    inline const short& trwf()     const { return _trwf;     }
    inline const short& grnors()   const { return _grnors;   }
    inline const short& grnofr()   const { return _grnofr;   }
    inline const short& grnlof()   const { return _grnlof;   }
    inline const short& gaps()     const { return _gaps;     }
    inline const short& otrav()    const { return _otrav;    }
    inline const float& d1()       const { return _d1;       }
    inline const float& f1()       const { return _f1;       }
    inline const float& d2()       const { return _d2;       }
    inline const float& f2()       const { return _f2;       }
    inline const float& ungpow()   const { return _ungpow;   }
    inline const float& unscale()  const { return _unscale;  }
    inline const int  & ntr()      const { return _ntr;      }
    inline const short& mark()     const { return _mark;     }
    inline const short& shortpad() const { return _shortpad; }
};

////////////////////////////////////////////////////////////////////////////////

#endif /*! SU_HPP */

////////////////////////////////////////////////////////////////////////////////
