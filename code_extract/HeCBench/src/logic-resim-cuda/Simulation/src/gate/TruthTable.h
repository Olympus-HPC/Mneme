#ifndef TRUTHTABLE_H
#define TRUTHTABLE_H
#include "util.h"

#include <stdlib.h>
#include <cmath>

#define TRUTHTABLE_SIZE(i)    short(pow(4, i))
#define VALARRSIZE(t)         (t&3)?(t>>2)+1:(t>>2)

typedef class Value {
    private:
        char data;

    public:
        Value(): data(0) {}
        ~Value() {}

        // insert value
        inline void insert(char, char);
        // get value
        inline char get(char pos) { return (data >> (pos<<1)) & 3; }

#ifdef TRUTH_DEBUG
        friend ostream& operator<<(ostream&, Value&);
#endif
} VAL;

typedef class TruthTable {
    private:
        VAL* array; // Store truth table address
        char isize; // Store input size of truth table

    public:
        TruthTable(size_t&, char*);
        ~TruthTable() { delete [] array; }

        inline void insert(short&, char&);
        inline const char get   (short& pos) const { return (array+(pos>>2))->get(pos&3); }
        inline const size_t input_size() const { return isize; }

        inline const VAL* getVAL() const { return array; }

#ifdef TRUTH_DEBUG
        friend ostream& operator<<(ostream&, TruthTable&);
#endif
} tTable;

#endif