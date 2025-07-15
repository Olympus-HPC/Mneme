#ifndef INTERPARSER_H
#define INTERPARSER_H

#include "WireMgr.h"
#include "TruthTable.h"
#include "GateMgr.h"
#include "Simulator.h"
#include "SAIF.h"

#include <vector>
using std::vector;

typedef vector<tTable*> EvalMgr;

class interParser {
    private:
        WireMgr* _wMgr;
        GateMgr* _gMgr;
        EvalMgr* _eMgr;

        SAIFprinter* _printer;
        Simulator* _simulator;

        // parseEval Cache
        vector<char> truth_iPortSize;
        char* h_eTable;
        unsigned int StartCPU;

        // parseBlock Cache
        size_t  popC,
                delC,
                addC,
                wireSer,
                simulateSize;

        // parseGate Cache
        gpu_GATE  h_gate[C_PARALLEL_LIMIT];
        tHistory* h_oWireArr[C_PARALLEL_LIMIT];
        size_t    expArray[C_PARALLEL_LIMIT][3];
        size_t    oWireSer[C_PARALLEL_LIMIT];


        // Read in counter
        size_t EvalC;
        size_t TransC;
        size_t iWireC;
        size_t oWireC;
        size_t AssignedC;
        size_t BlockC;
        size_t gateC;

        size_t lineCounter;
        size_t Gcounter;
        vector<sWire*> toBeDel;
        sWire* swire;
    public:
        interParser(Simulator* sim) 
            :EvalC(0), TransC(0), iWireC(0), oWireC(0), AssignedC(0), BlockC(0), _simulator(sim)
        { 
            gateC = C_PARALLEL_LIMIT;
        }
        ~interParser() {}

        WireMgr* getWireMgr() const { return _wMgr; }
        GateMgr* getGateMgr() const { return _gMgr; }
        EvalMgr* getEvalMgr() const { return _eMgr; }

        char* parseHeader(char*);
        char* parseInst  (char*, SAIFprinter*);

        void parseInit  (char*&, size_t&);
        void parseEval  (char*&, size_t&);
        void parseIWire (char*&, size_t&);
        void parseOWire (char*&, size_t&);
        void parseAssign(char*&, size_t&);
        void parseBlock (char*&, size_t&);
        void parseGate  (char*&, size_t&, bool=true);

        char* jumpBlock (char*);

        inline const size_t getTransC() const { return TransC; }
};

#endif