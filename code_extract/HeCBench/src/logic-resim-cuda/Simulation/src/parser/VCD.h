#ifndef VCD_H
#define VCD_H

#include "Wire.h"

#include <map>
using std::map;

const char symbolPrefab[] = {
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','c','d','e','f','g','h','i','j',
    'k','l','m','n','o','p','q','r','s','t',
    'u','v','w','x','y','z',
    '[',']','\"','\'','?','!','@','#','$','%',
    '^','&','*','{','}','(',')','<','>',':',
    ';','|','\\','+','=','/','~','`','_','-',
    '.',',',
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z'
};

const short symbolLength = 94;

typedef class VCDNode {
    friend class VCDprinter;
    string transition;


    VCDNode () { transition.reserve(1000); };
    ~VCDNode() {}
    void addTrans(string trans) { transition += trans + '\n'; }
} vNode;

typedef map<tUnit, VCDNode*> TimeList;

class VCDprinter {
        const static string varDecl;
        const static string endDecl;

    private:
        TimeList timelist;
        tUnit dumpOn;
        tUnit dumpOff;
        string wireDecl;
        string filename;

        vNode& get(const tUnit&);
        string assignSymbol(size_t);
        tUnit getCurrTrans(vector<size_t>&, mWire&, string&, size_t);
        string strip(string&, size_t);

    public:
        VCDprinter(tUnit on, tUnit off, string& f) :dumpOn(on), dumpOff(off), filename(f) {}
        ~VCDprinter() {}


        void dump(sWire&, string&);
        void dump(mWire&, string&);
        void dump(Wire*, size_t);
        void print(string&, string&);
};

#endif