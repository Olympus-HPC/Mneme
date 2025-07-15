#ifndef SAIF_H
#define SAIF_H

#include "Wire.h"

#include <string>
#include <fstream>
using std::string;
using std::fstream;


class SAIFprinter{
    private:
        vector<string>  buffer;
        size_t          buffer_size,
                        counter;
        short           modC;
        string          filename;
        // fstream         saiffile;

        tUnit dumpOn, dumpOff;
    public:
        SAIFprinter(tUnit, tUnit, string&, string&, size_t, short);
        ~SAIFprinter() {
            
        }

        void dump(sWire*);
        void close();
};

#endif