#ifndef GETCHAR_H
#define GETCHAR_H

#include <string>
#define ULL unsigned long long

using std::string;
using std::vector;

static char*  jump(char*&, size_t=1);
static char*  jumpline(char*&, size_t=1);
static string getString(char*&, size_t=1);
static size_t getSizeT(char*&, size_t=0);
static ULL    getULL(char*&, ULL=0);
static bool   myCmp(char*, const char*, size_t);


static char*
jump(char*& c, size_t s) {
    char* jumper = c;
    while(s != 0) {
        if ((*jumper == ' ') || (*jumper == '\n'))
            --s;
        ++jumper;
    }
    return jumper;
}

static char*
jumpline(char*& c, size_t s) {
    char* jumper = c;
    while(s) { if(*jumper == '\n') --s; ++jumper; }
    return jumper;
}

static string
getString(char*& c, size_t s) {
    string raw;
    while(s) {
        if(*c == ' ' || *c == '\n') --s;
        raw.push_back(*c); ++c;
    }
    raw.pop_back();
    return raw;
}

static size_t
getSizeT(char*& c, size_t value) {
    while( *c != ' ' && *c != '\n') { value = value*10 + (size_t)(*c-'0'); ++c; }
    ++c;
    return value;
}

static ULL
getULL(char*& c, ULL value) {
    while( *c != ' ' && *c != '\n') { value = value*10 + (size_t)(*c-'0'); ++c; }
    ++c;
    return value;
}

static bool
myCmp(char* c, const char* p, size_t s) {
    size_t i = 0;
    while(s) {
        if (*(c+i) != *(p+i)) {
            return false;
        }
        --s, ++i;
    }
    return true;
}

#endif