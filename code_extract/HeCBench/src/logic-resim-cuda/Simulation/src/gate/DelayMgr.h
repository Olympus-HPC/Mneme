#ifndef DELAYMGR_H
#define DELAYMGR_H

#include <stdlib.h>
#include "util.h"

#define MAX_INPUT_PORT  6
#define MAX_DTUPLE 12
#define MAX_DTABLE MAX_DTUPLE * MAX_INPUT_PORT
#define RSIZE      MAX_DTABLE

typedef unsigned int dUnit;

typedef class dUnitMemBlock {
    friend class DelayMgr;
    dUnit*             _begin;
    dUnit*             _ptr;
    dUnit*             _end;
    dUnitMemBlock*     _nextBlock;

    dUnitMemBlock(dUnitMemBlock* n, size_t b) : _nextBlock(n) {
        _begin = _ptr = (dUnit*)malloc(b * sizeof(dUnit)); _end = _begin + b;
#ifdef DELAYMGR_DEBUG
        cout << "New block: " << this << " with begin:" << _begin << " end:" << _end << endl;
#endif
    }
    ~dUnitMemBlock() { free(_begin); }

    // Get the size of rxemaining memory in block
    inline size_t getRemainSize() const { return size_t(_end - _ptr); }
    // Clean the memory block
    inline void   reset() { _ptr = _begin; }
    // Get the next block
    inline dUnitMemBlock* getNextBlock() const { return _nextBlock; }


    bool getArray(size_t, dUnit*&);
#ifdef DELAYMGR_DEBUG
    friend ostream& operator<<(ostream&, dUnitMemBlock&);
#endif
} dBlock;

typedef class dUnitRecycleList {
    friend class DelayMgr;
    dUnit*      _first;

    dUnitRecycleList(): _first(0) {}
    ~dUnitRecycleList() {while (_first != nullptr) pop_front(); }

    void push_front(dUnit*);
    dUnit* pop_front();
    size_t numElem() const;
    // check whether the list is empty
    inline bool empty() const { return (_first)?false:true; }

} dRList;

class DelayMgr {
    private:
        size_t  _blockSize;
        dBlock* _activeBlock;
        dRList  _recycleList[RSIZE];

        dUnit* getMemArr(size_t);
        size_t getNumBlocks() const;
    public:
        DelayMgr(size_t b=MAX_DTABLE): _blockSize(b), _recycleList {} {
            _activeBlock = new dBlock(0, b);
        }
        ~DelayMgr() {
            dBlock* tmp = _activeBlock;
            while(tmp) {
                tmp = _activeBlock->getNextBlock();
                delete _activeBlock;
                _activeBlock = tmp;
            }
        }

        // Get memory address of dTable array
        inline dUnit* allocateArr(size_t arrSize) { return getMemArr(arrSize); }
        // Recycle deleted mem block
        inline void   freeArray(dUnit* p, size_t arrSize) { _recycleList[arrSize-1].push_front(p); }

#ifdef DELAYMGR_DEBUG
        friend ostream& operator<<(ostream&, DelayMgr&);
#endif
};

#endif