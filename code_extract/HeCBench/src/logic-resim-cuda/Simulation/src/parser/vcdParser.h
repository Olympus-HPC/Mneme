#ifndef VCDPARSER_H
#define VCDPARSER_H

#include "WireMgr.h"
#include "getChar.h"

class vcdParser {
    private:
        WireMgr* _wMgr;
        unordered_map<string, Wire*> symbolMapper;
        tUnit dumpOff;

        string modHeader;
        string endHeader;
        string SAIFmoduleName;
        short  modCounter;
        double timescale;


        const static char    timeDecl[];
        const static char    moduleDecl[];
        const static char    endmodDecl[];
        const static char    varDecl[];
        const static char    initDecl[];
        const static char    endDecl[];

        const static string  SAIFmodDecl;
        // Need max delay of netlist to decided where to stop parsing waveform

    public:
        vcdParser(WireMgr* mgr, tUnit off): _wMgr(mgr), modCounter(0), dumpOff(off) {}
        ~vcdParser(){}

        inline string& getModHeader () { return modHeader;     }
        inline string& getEndHeader () { return endHeader;     }
        inline string& getSAFIHeader() { return SAIFmoduleName;}
        inline short&  getModCounter() { return modCounter;    }

        void parseAll      (char*);
        void parseTimescale(char*&);
        void parseHeader   (char*&);
        void parseVar      (char*&, size_t=1);
        void parseDump     (char*&, tUnit, size_t=1);
        void parseDumpM    (string, char*, tUnit);
        bool parseTimeblock(char*&);
};

#endif