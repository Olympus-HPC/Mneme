#ifndef GATEMGR_H
#define GATEMGR_H

#include "Gate.h"

#include <vector>
using std::vector;

typedef vector<Gate*>     GateList;

class GateMgr
{
private:
    GateList _gatelist;

public:
    GateMgr() {}
    ~GateMgr() { reset(); }


    void addGate (Gate* gate) { _gatelist.push_back(gate); }
    void reset   () { for(auto &g: _gatelist) delete g; _gatelist.clear(); }
    GateList& getList() { return _gatelist; }

};

#endif