#ifndef EVALMGR_H
#define EVALMGR_H

#include "TruthTable.h"

#include <vector>
using std::vector;

class EvalMgr {
    private:
        vector<tTable*> evalList;

    public:
        EvalMgr(size_t s) { evalList.resize(s); }
        ~EvalMgr() { for(auto t:evalList) delete t; }

        void addTable(size_t& s, char* truth, size_t serial) {
            tTable* newTable = new tTable(s, truth);
            evalList[serial] = newTable;
        }

        tTable* getTable(size_t i) const { return evalList[i]; }

#ifdef TRUTH_DEBUG
        friend ostream& operator<<(ostream& os, EvalMgr mgr) {
            for (auto table: mgr.evalList)
                os << *table << endl;
        }
#endif
};

#endif