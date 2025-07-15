#include "Wire.h"
#include <algorithm>

void 
sWire::addSignal(char* c, tUnit time) {
    His->push_back(time, vEncoder(*c));
}


void 
mWire::addSignal(char* c, tUnit time) {
    char extend = vEncoder((*c=='1')?'0':*c);
    size_t s = 0;
    while (*(c+s) != ' ') ++s;

    unsigned short ei = 0;
    for (;ei<Size-s;++ei)
        HisList[ei]->push_back(time, extend);
    for (;ei<Size;++c, ++ei)
        HisList[ei]->push_back(time, vEncoder(*c));
}