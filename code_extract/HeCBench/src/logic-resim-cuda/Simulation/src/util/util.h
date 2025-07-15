#ifndef UTIL_H
#define UTIL_H

#ifdef DEBUG
#define GATE_DEBUG
#endif

#ifdef GATE_DEBUG
#define TRUTH_DEBUG
#define DELAY_DEBUG
#endif

#ifdef TRUTH_DEBUG
#define DEBUG_LIB
#endif

#ifdef GPU_DEBUG
#define DEBUG_LIB
#endif

#ifdef DELAY_DEBUG
#define DEBUG_LIB
#define PRINTBLOCK 6
#endif
#ifdef TIME
#define DEBUG_LIB
#include <time.h>
static time_t startTimer, endTimer;
#define timer(h, c)               \
startTimer = time(0);             \
c;                                \
endTimer   = time(0);             \
cout << h << " runtime: "         \
    << endTimer - startTimer << " s" << endl
#else
#define timer(h, c) c
#endif


#ifdef DEBUG_LIB
#include <iostream>
#include <iomanip>
#include <cassert>
using std::cout;
using std::endl;
using std::setw;
using std::left;
using std::right;
using std::ostream;
#endif




const char value0 = 0;
const char value1 = 1;
const char valueX = 2;
const char valueZ = 3;

extern char vEncoder(char);
extern char vDecoder(char);

#endif