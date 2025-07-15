#include "TruthTable.h"

inline void
Value::insert(char pos, char value) {
    data |= (value<<(pos<<1));
}


#ifdef TRUTH_DEBUG
ostream&
operator<<(ostream& os, Value& val) {
    for(char i=0;i < 4;++i)
        os << (short)val.get(i);
    return os;
}
#endif


TruthTable::TruthTable(size_t& is, char* truth): isize(is) {
    short tableSize = TRUTHTABLE_SIZE(is);
    array = new VAL[VALARRSIZE(tableSize)];
    // Insert truth into array
    for(short pos=0;pos<tableSize;++pos)
        (array + (pos>>2))->insert(pos&3, *(truth+pos) - '0');
}

inline void 
TruthTable::insert(short& pos, char& val) {
    (array + (pos>>2))->insert(pos&3, val);
}

#ifdef TRUTH_DEBUG
ostream&
operator<<(ostream& os, TruthTable& table) {
    size_t tableSize = TRUTHTABLE_SIZE(table.isize);
    tableSize = VALARRSIZE(tableSize);
    os << &table << " Truth: " << (size_t)table.isize << ' ';
    for(unsigned char i=0;i < tableSize;++i)
        os << (*(table.array+i));
    return os;
}
#endif