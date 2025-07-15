#include "SAIF.h"

using std::to_string;

SAIFprinter::SAIFprinter
(tUnit on, tUnit off, string& f, string& mName, size_t TransC, short modc)
: filename(f), dumpOff(off), dumpOn(on), modC(modc),
        buffer_size(TransC), counter(0)
        //, saiffile(filename, std::ios::out) { 
{
    buffer.reserve(TransC+1);


    string Header = "(SAIFILE\n(SAIFVERSION \"2.0\")\n(DIRECTION \"backward\")\n(DESIGN )\nDIVIDER / )\n(TIMESCALE 1 ps)\n(DURATION " + to_string(off-on) + ")\n" + mName + "(NET\n";
    buffer.push_back(Header);
    // saiffile << Header;
}

void 
SAIFprinter::dump(sWire* w) {
    tHistory*   His  = w->getHis();
    if (His->isDeleted()) return ;
    string      name = w->getName();
    char        prevV, currV;
    tUnit       prevT, currT,
                T0 = 0, T1 = 0, TX = 0;
    size_t      i=0,
                size = His->size();

    for(;i < size && (*His)[i].t < dumpOn;++i) {}

    prevT = dumpOn         , prevV = His->getValue(i-1);
    currT = His->getTime(i), currV = His->getValue(i)  ;

    while(i < size && currT < dumpOff) {
        switch (prevV) {
            case 0: T0 += (currT-prevT); break;
            case 1: T1 += (currT-prevT); break;
            case 2:
            case 3: TX += (currT-prevT); break;
        }

        prevT = currT, prevV = currV;
        ++i;
        currT = His->getTime(i), currV = His->getValue(i);
    }

    switch (prevV) {
        case 0: T0 += (dumpOff-prevT); break;
        case 1: T1 += (dumpOff-prevT); break;
        case 2:
        case 3: TX += (dumpOff-prevT); break;
    }

    string result = '(' + name + "\n(T0 " + to_string(T0) + ") (T1 " \
            + to_string(T1) + ") (TX " + to_string(TX) + ")\n(TC  0) (IG 0 )\n)\n";
    buffer.push_back(result);
    // saiffile << result;
}

void 
SAIFprinter::close() {
    fstream saiffile(filename, std::ios::out);

    for (size_t i=0;i<buffer.size();++i) {
        saiffile << buffer[i];
    }

    for(short i=0;i<modC+2;++i)
        saiffile << ")\n";
    saiffile.close();
}
