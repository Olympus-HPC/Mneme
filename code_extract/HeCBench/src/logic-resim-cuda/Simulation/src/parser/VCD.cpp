#include "VCD.h"
#include "util.h"

#include <climits>
#include <fstream>
using std::fstream;
using std::to_string;

string const VCDprinter::varDecl = "$var wire ";
string const VCDprinter::endDecl = " $end\n";


// Helper functions
vNode& 
VCDprinter::get(const tUnit& t) {
    TimeList::iterator node = timelist.find(t);
    if (node != timelist.end()) {
        return *(node->second);
    } else {
        vNode* n = new vNode();
        timelist.insert(std::pair<tUnit, vNode*>(t,n));
        return *n;
    }
}

string 
VCDprinter::assignSymbol(size_t s) {
    string symbol;
    if (s == 0) return "0";
    while (s != 0) {
        symbol = symbolPrefab[s%symbolLength] + symbol;
        s = s / symbolLength;
    }
    return symbol;
}

tUnit 
VCDprinter::getCurrTrans
(vector<size_t>& head, mWire& wire, string& value, size_t size) {
    tUnit currT = -1;
    vector<tUnit> headTime(size, 0);

    // Get next trans time
    for (size_t i = 0;i<size;++i) {
        headTime[i] = wire[i]->getTime(head[i]+1);
        if (headTime[i] < currT) currT = headTime[i];
    }

    // Update head and value if next trans time equals to next trans time
    for (size_t i=0;i<size;++i) {
        if (headTime[i] == currT) {
            ++head[i]; value[i] = vDecoder(wire[i]->getValue(head[i]));
        }
    }
    return currT;
}

string 
VCDprinter::strip(string& v, size_t s) {
    // if (v[0] == '1') return v;
// 
    // char extend = v[0];
    // size_t i = 1;
    // while(i < s && v[i] == extend) ++i;
// 
    // if (v[i] == '1' && extend == '0')
        // i += 1;
// 
    // return v.substr(i-1);
    return v;
}

// Public func
void 
VCDprinter::dump(sWire& wire, string& symbol) {
    size_t head = 0;
    tHistory* his = wire.getHis();
    tUnit currT = his->getTime(0);

    if (dumpOn != 0) {
        // Add init value
        get(0).addTrans("x"+symbol);

        // Update head to closet trans after(or equal) dumpOn
        while (currT < dumpOn && currT != (tUnit)-1) {
            ++head; currT = his->getTime(head);
        }

        // If closet trans not equal to dumpOn, add closet trans before dumpOn
        if (currT != dumpOn) {
            if (his->getValue(head-1) != valueX)
                get(dumpOn).addTrans(vDecoder(his->getValue(head-1))+symbol);
        }
    }

    while (currT != (tUnit)-1 && currT < dumpOff) {
        get(currT).addTrans(vDecoder(his->getValue(head))+ symbol);
        ++head; currT = his->getTime(head);
    }

    // if (his->getValue(head-1) != valueX)
    //     get(dumpOff).addTrans("x"+symbol);
}

void 
VCDprinter::dump(mWire& wire, string& symbol) {
    size_t size = wire.getSize();

    vector<size_t>    head(size, 0);
    tUnit  currT = -1;
    string value;
    value.resize(size);

    if (dumpOn != 0) {
        // Add init value
        get(0).addTrans("bx "+symbol);


        // Update head to closet trans before dumpOn
        for (size_t i=0;i<size;++i) {
            while (wire[i]->getTime(head[i]+1) < dumpOn) ++head[i];
            if (currT > wire[i]->getTime(head[i]+1)) currT = wire[i]->getTime(head[i]);
        }

        // If closet trans not equal to dumpOn, add closet trans before dumpOn
        if (currT != dumpOn) {
            for (size_t i=0;i<size;++i) {
                value[i] = vDecoder(wire[i]->getValue(head[i]));
            }
            if (strip(value, size) != "x")
                get(dumpOn).addTrans('b'+strip(value,size)+' '+symbol);
        }
    }
    else {
        // Add trans at time 0
        for (size_t i=0;i<size;++i) {
            value[i] = vDecoder(wire[i]->getValue(0));
        }
        get(dumpOn).addTrans('b'+strip(value,size)+' '+symbol);
    }

    currT = getCurrTrans(head, wire, value, size);
    while (currT != (tUnit)-1 && currT < dumpOff) {
        get(currT).addTrans('b'+strip(value,size)+' '+symbol);
        currT = getCurrTrans(head, wire, value, size);
    }

    // get(dumpOff).addTrans("bx "+symbol);
}

void 
VCDprinter::dump(Wire* wire, size_t serial) {
    size_t size = wire->getSize();
    string symbol = assignSymbol(serial);
    wireDecl += varDecl + to_string(wire->getSize()) + ' ' + \
                symbol + ' ' + wire->getDecl() + endDecl;

    if (size > 1)
        dump(*(mWire*)wire, symbol);
    else
        dump(*(sWire*)wire, symbol);
}

void 
VCDprinter::print(string& moduleDecl, string& endHeader) {
    TimeList::const_iterator node = timelist.begin();
    fstream vcdfile(filename, std::ios::out);

    vcdfile << "$timescale 1ps $end\n";
    vcdfile << moduleDecl;
    vcdfile << wireDecl;
    vcdfile << endHeader;
    vcdfile << node->second->transition;
    vcdfile << "$end\n";
    ++node;
    for (;node != timelist.end(); ++node)
    {
        vcdfile << '#' << node->first << '\n';
        vcdfile << node->second->transition;
    }
    vcdfile.close();
}
