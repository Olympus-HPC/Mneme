#ifndef GATE_H
#define GATE_H

#include "Signal.h"
#include "DelayMgr.h"
#include "TruthTable.h"

typedef dUnit dTable;
class Gate
{
private:
    tTable*    truthtable;
    dTable*    delaytable;
    vector<tHistory*> transtable;

    static DelayMgr* const _dmemMgr;
    size_t truthIdx;
public:
    Gate(tTable*, size_t, size_t);
    ~Gate();

    void insert_delay(dUnit, short, short);
    inline const dUnit getDelay (char, char) const;

    inline void  insert_wire (tHistory* t, char p) { transtable[p] = t; }
    inline const tHistory* getSignal(char p) const { return transtable[p]; }
    inline vector<tHistory*>& getPortTable() { return transtable; }

    inline const size_t    getInputSize () const { return truthtable->input_size(); }
    inline const tTable*   getTruthTable() const { return truthtable; }
    inline const size_t&   getTruthIndex() const { return truthIdx; }
    inline const dTable*   getDelayTable() const { return delaytable; }

#ifdef GATE_DEBUG
    friend ostream& operator<<(ostream& os, Gate& gate);
#endif

#ifdef DELAYMGR_DEBUG
    void check_mem() { cout << _dmemMgr; }
#endif
};

#endif