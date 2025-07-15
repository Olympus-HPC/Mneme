#ifndef WIREMGR_H
#define WIREMGR_H

#include "Wire.h"

#include <unordered_map>
using std::unordered_map;


class WireMgr
{
private:
    unordered_map<string,Wire*> nameMapper;   // name -> signal *For searching the signal by name while parsing Netlist
    vector<Wire*>                 WireList;   // Contains all wire (bus and not bounded wire.)
    vector<sWire*>               sWireList;
    vector<tHistory*>            TransList;   

public:
    WireMgr(size_t ts, size_t ws)
        : nameMapper() {
            TransList.resize(ts); WireList.reserve(ws);
            for(size_t i=0;i<ts;++i) TransList[i] = new tHistory();
        }

    void addsWire    (string&, size_t, size_t, bool);
    void addmWire    (string&, size_t, size_t, size_t, size_t, bool);
    void connectWire (size_t , size_t);


    inline tHistory*     operator[] (size_t i)     const { return TransList[i-4]; }
    inline Wire*         getWire    (string& name) const { return nameMapper.find(name)->second; }
    inline vector<Wire*>& getWireList()   { return WireList; }
    inline vector<sWire*>& getsWireList() { return sWireList; }
    inline sWire* getsWire(size_t i) { return sWireList[i-4]; }

#ifdef DEBUG
    friend ostream& operator<<(ostream& os, WireMgr& mgr) {
        for(auto w: mgr.WireList)
            cout << *w;
        return os;
    }
#endif
};

#endif