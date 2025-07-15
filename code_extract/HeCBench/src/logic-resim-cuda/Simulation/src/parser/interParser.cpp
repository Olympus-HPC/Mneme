#include "interParser.h"
#include "getChar.h"

#include<algorithm>
using std::sort;

char*
interParser::parseHeader(char* c) {
    char* begin = c;
    parseInit(c, lineCounter);

    lineCounter = 0;
    while(lineCounter < EvalC)
        parseEval(c, lineCounter);
    _simulator->addEvalMgr(h_eTable, EvalC);

    lineCounter = 0;
    while(lineCounter < AssignedC)
        parseAssign(c, lineCounter);

    lineCounter = 0;
    while (iWireC > 0) {
        parseIWire(c, lineCounter);
        --iWireC;
    }

    while (oWireC > 0) {
        parseOWire(c, lineCounter);
        --oWireC;
    }
    return c;
}

char* 
interParser::parseInst(char* c, SAIFprinter* saif) {
    lineCounter = 0;
    _printer = saif;
    while(lineCounter < BlockC) {
        parseBlock(c, lineCounter);
    }

    for (auto& w: _wMgr->getsWireList()) {
        _printer->dump(w);
    }
    _printer->close();
    return c;
}

void
interParser::parseInit(char*& c, size_t& serial) {
    TransC    = getSizeT(c, 0);
    iWireC    = getSizeT(c, 0);
    oWireC    = getSizeT(c, 0);
    AssignedC = getSizeT(c, 0);
    EvalC     = getSizeT(c, 0);
    BlockC    = getSizeT(c, 0);
    StartCPU  = getSizeT(c, 0);

    _wMgr = new WireMgr(TransC, iWireC+oWireC);
    _gMgr = new GateMgr();
    _eMgr = new EvalMgr(EvalC,  0);

    size_t tableSize = EvalC*TRUTH_SIZE;
    h_eTable = new char[tableSize];
    truth_iPortSize.reserve(EvalC);

    ++serial;
}

void
interParser::parseEval(char*& c, size_t& serial) {
    size_t is = getSizeT(c, 0);
    (*_eMgr)[serial] = new tTable(is, c);

    truth_iPortSize.push_back((char)is);

    size_t  head = serial*TRUTH_SIZE,
            counter = 0;

    // Parse truth into cache
    while (*c != '\n') {
        h_eTable[head + (counter>>2)] |= (((*c)-'0')<<(counter&3)*2);
        ++c, ++counter;
    }
    ++c;

    h_eTable[head+TRUTH_SIZE-1] = (char)is;

    ++serial;
}

void
interParser::parseIWire(char*& c, size_t& serial) {
    size_t size = getSizeT(c, 0);
    string name = getString(c, 1);
    size_t msb, lsb;
    if (size > 1) {
        msb = getSizeT(c);
        lsb = getSizeT(c);
        _wMgr->addmWire(name, size, serial, msb, lsb, true);
    }
    else          _wMgr->addsWire(name, size, serial, true);

    serial += size;
}

void
interParser::parseOWire(char*& c, size_t& serial) {
    size_t size = getSizeT(c, 0);
    string name = getString(c, 1);
    size_t msb, lsb;
    if (size > 1) {
        msb = getSizeT(c);
        lsb = getSizeT(c);
        _wMgr->addmWire(name, size, serial, msb, lsb, false);
    }
    else          _wMgr->addsWire(name, size, serial, false);

    serial += size;
}

void
interParser::parseAssign(char*& c, size_t& serial) {
    size_t l = getSizeT(c, 0);
    size_t r = getSizeT(c, 0);

    _wMgr->connectWire(l, r);
    ++serial;
}

void
interParser::parseBlock(char*& c, size_t& serial) {
    #ifdef DEBUG_LIB
    cout << "Block " << serial << endl;
    #endif
    if (serial == StartCPU)
        _simulator->cleanGPU();
    if (serial < StartCPU) {
        // Pop from Device
        popC = getSizeT(c, 0);
        while (popC) {
            wireSer = getSizeT(c, 0);
            _simulator->popTrans(wireSer);
            --popC;
        }

        // Delete from Host
        delC = getSizeT(c, 0);
        while (delC) {
            swire = _wMgr->getsWire(getSizeT(c, 0));
            _printer->dump(swire);
            toBeDel.push_back(swire);
            --delC;
        }
        for (size_t i=0;i < toBeDel.size();++i) {
            toBeDel[i]->remove();
        }
        toBeDel.clear();

        // Add to Device
        addC = getSizeT(c, 0);
        while (addC) {
            wireSer = getSizeT(c, 0);
            _simulator->addTrans((*_wMgr)[wireSer], wireSer);
            --addC;
        }

        // Parse Gate
        Gcounter = 0;
        simulateSize = 0;
        while(Gcounter < C_PARALLEL_LIMIT) {
            parseGate(c, Gcounter);
        }
        _simulator->addGateMgr(h_gate, simulateSize);
        _simulator->simulateBlock(h_oWireArr, oWireSer, expArray);
    }
    else {
        _gMgr->reset();
        c = jumpline(c); // Pop from Device
        size_t delCounter = getSizeT(c, 0);
        for (size_t i=0;i<delCounter;++i) {
            swire = _wMgr->getsWire(getSizeT(c, 0));
            _printer->dump(swire);
            toBeDel.push_back(swire);
        }
        for (size_t i=0;i < toBeDel.size();++i) {
            toBeDel[i]->remove();
        }
        toBeDel.clear();

        c = jumpline(c); // Add to Device
        Gcounter = 0;
        while(Gcounter < gateC) {
            parseGate(c, Gcounter, false);
        }
        _simulator->simulateBlock(_gMgr->getList());
    }
    ++serial;
}

void 
interParser::parseGate(char*& c, size_t& serial, bool useGPU) {
    size_t funcSer = getSizeT(c, 0);
    if (useGPU) {
        size_t iPortSize    = truth_iPortSize[funcSer];
        size_t inputSizeArr[MAX_INPUT_PORT] = {0};

        h_gate[serial].funcSer = funcSer;
        if (funcSer >= EvalC) { oWireSer[serial] = 0; ++serial; return ; }

        for(unsigned short i=0;i < iPortSize; ++i) {
            wireSer = getSizeT(c, 0);
            h_gate[serial].iPort[i] = _simulator->getTrans(wireSer);
            inputSizeArr[i] = (*_wMgr)[wireSer]->size();
        }

        sort(inputSizeArr, inputSizeArr + 6);
        expArray[serial][0] = inputSizeArr[5] + inputSizeArr[3] + inputSizeArr[2] + inputSizeArr[4] + inputSizeArr[1] + inputSizeArr[0] + 2;
        expArray[serial][1] = inputSizeArr[0];
        expArray[serial][2] = inputSizeArr[0] + 1;

        oWireSer[serial] = getSizeT(c, 0);
        h_oWireArr[serial] = (*_wMgr)[oWireSer[serial]];

        for(unsigned short i=0;i < iPortSize * MAX_DTUPLE; ++i) 
            h_gate[serial].dTable[i] = getSizeT(c, 0);
        ++simulateSize;
    } else {
        if (funcSer >= EvalC)  { ++serial; return; }

        tTable* table     = (*_eMgr)[funcSer];
        size_t  inputSize = table->input_size();
        Gate* gate = new Gate(table, funcSer, inputSize);

        for(unsigned short i=0;i < inputSize+1; ++i)
            gate->insert_wire((*_wMgr)[getSizeT(c, 0)], i);

        for(unsigned short i=0;i < inputSize; ++i) {
            for(unsigned short d=0; d < MAX_DTUPLE; ++d) {
                gate->insert_delay(getSizeT(c), i, d);
            }
        }

        _gMgr->addGate(gate);
    }
    ++serial;
}

char*
interParser::jumpBlock(char* c) {
    size_t gateC = getSizeT(c, 0);
    return jumpline(c, gateC);
}
