#include "Gate.h"

DelayMgr* const Gate::_dmemMgr = new DelayMgr;

Gate::Gate(tTable* table, size_t funcSer, size_t inputSize) :truthtable(table), truthIdx(funcSer) {
    if (inputSize)
        delaytable  = _dmemMgr->allocateArr(MAX_DTUPLE*inputSize);
    transtable.resize(inputSize+1);
}

Gate::~Gate() {
    _dmemMgr->freeArray(
        delaytable, MAX_DTUPLE*truthtable->input_size()
    );
}

void 
Gate::insert_delay(dUnit delay, short iPos, short dPos) {
    *(delaytable + (MAX_DTUPLE*iPos) + dPos) = delay;
}

inline const dUnit 
Gate::getDelay(char iPos, char dPos) const {
    return *(delaytable + (MAX_DTUPLE*iPos) + dPos);
}

#ifdef GATE_DEBUG
ostream& 
operator<<(ostream& os, Gate& gate) {
    cout << "Input Size: " << gate.truthtable->input_size() << endl;
    cout << "Truth table: "
        << (gate.truthtable) << endl
        << *(gate.truthtable) << endl
        << "Delay table: ";
    for(unsigned short i=0;i<gate.truthtable->input_size();++i)
        for(unsigned short j=0;j<MAX_DTUPLE;++j)
            cout << *(gate.delaytable + i*MAX_DTUPLE + j) << ' ';
    cout << endl;
    return os;
}
#endif