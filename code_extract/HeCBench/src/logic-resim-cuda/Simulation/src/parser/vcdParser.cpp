#include "vcdParser.h"

// #include <thread>
// using std::thread;
using std::make_pair;

#define TDLIMIT 10000

char const vcdParser::timeDecl[] = "$timescale";
char const vcdParser::moduleDecl[] = "$scope";
char const vcdParser::endmodDecl[] = "$upscope";
char const vcdParser::varDecl[] = "$var";
char const vcdParser::initDecl[] = "$dumpvars";
char const vcdParser::endDecl[] = "$end";
string const vcdParser::SAIFmodDecl = "(INSTANCE ";

void 
vcdParser::parseAll(char* c) {
    parseHeader(c);
    while (*c != '\0') {
        if (parseTimeblock(c)) return;
    }
}

void 
vcdParser::parseTimescale(char*& c) {
    c = jump(c); // Jump timescale declaration

    ++c; // Jump 1
    short t = 1;
    while (!(*c - '0')) { t = t*10; ++c;}

    switch ((*c)) {
        case 'f' : timescale = 10e-3; break;
        case 'p' : timescale = 1;     break;
        case 'n' : timescale = 10e3;  break;
        case 'u' : timescale = 10e6;  break;
        case 'm' : timescale = 10e9;  break;
        case 's' : timescale = 10e12; break;
        default: timescale = 1; break;
    }
}

void 
vcdParser::parseVar(char*& c, size_t s) {
    size_t  size;
    string  name,
            symbol;
    while (s) {
        c = jump(c, 2); // jump "$var wire"
        size   = getSizeT(c);
        symbol = getString(c);
        name   = getString(c);
        if (size > 1) c = jump(c); // jump slice

        symbolMapper.insert(make_pair(symbol, _wMgr->getWire(name)));
        c = jump(c); // jump "$end"

        --s;
    }

}

void 
vcdParser::parseHeader(char*& c) {
    char* mName;
    while(!myCmp(c, timeDecl, 3)) c = jump(c);
    parseTimescale(c);

    while (!myCmp(c, moduleDecl, 3)) c = jump(c);

    while (!myCmp(c, varDecl, 3)) {
        mName = jump(c,2);
        ++modCounter;
        SAIFmoduleName.append(SAIFmodDecl + getString(mName) + '\n');
        modHeader.append(getString(c, 4) + '\n');
    }
    while (!myCmp(c, endmodDecl, 3)) parseVar(c, 1);
    while (!myCmp(c, initDecl, 3)) endHeader.append(getString(c)+'\n');
    endHeader.append(getString(c)+'\n');
    while (!myCmp(c, endDecl, 3)) parseDump(c, 0, 1);
    c = jump(c); // jump $end
    ++c;
}

void 
vcdParser::parseDump(char*& c, tUnit t, size_t s) {
    char* val;
    string symbol;
    while (s) {
        val = c;
        if (*c != 'b') {
            ++c; //jump val
        }
        else {
            c = jump(c); // jump val
            ++val;
        }

        symbol = getString(c);
        symbolMapper.find(symbol)->second->addSignal(val, t);
        --s;
    }
}

void 
vcdParser::parseDumpM(string symbol, char* val, tUnit t) {
    symbolMapper.find(symbol)->second->addSignal(val, t);
}

bool 
vcdParser::parseTimeblock(char*& c) {
    tUnit time = getULL(c, 0);
    if (time > dumpOff)
        return true;

    while (*c != '#' && *c != '\0') {
        parseDump(c, time);
    }

    if (*c != '\0') ++c;
    return false;

}