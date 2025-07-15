#include "DelayMgr.h"


/* dUnitMemBlock */
/* ------------- */

// Get t * sizeof(dUnit) bytes of memory from block.
bool 
dBlock::getArray(size_t t, dUnit*& ret) {
    ret = _ptr;
    if(_ptr+t > _end) {
#ifdef DELAYMGR_DEBUG
    cout << "Allocate: " << ret << ", now ptr to: " << _end <<  endl;
#endif
        _ptr = _end; return false;
    }
    _ptr += t;
#ifdef DELAYMGR_DEBUG
    cout << "Allocate: " << ret << ", now ptr to: " << _ptr <<  endl;
#endif
    return true;
}

#ifdef DELAYMEM_DEBUG
// Print all content in block
ostream&
operator<<(ostream&os, dBlock& d) {
        dUnit* it = d._begin;
        size_t lineCounter = 1;
        while(it != d._ptr && (*it) < 1000)
        {
            cout << right << setw(4) << (*it);
            if (!(lineCounter%PRINTBLOCK)) cout << endl;
            ++it, ++lineCounter;
        }
}
#endif


/* dUnitRecycleList */
/* ---------------- */

// push the element 'p' to the beginning of the recycle list
void 
dRList::push_front(dUnit* p) {
    dUnit** tmp = (dUnit**)p;
    *tmp = _first;
    _first = p;
}

// pop out the first element in the recycle list
dUnit* 
dRList::pop_front() {
    if(!_first) return 0;
    dUnit* tmp = _first;
    _first = *(dUnit**)_first;
    return tmp;
}

// count the number of elements in the recycle list
size_t 
dRList::numElem() const {
    int num = 0;
    dUnit* tmp = _first;
    while(tmp) {
        num += 1;
        tmp = *(dUnit**)tmp;
    }
    return num;
}


/* DelayMgr */
/* -------- */

// Manage memory allocator of the dTable array
dUnit* 
DelayMgr::getMemArr(size_t t) {
    dUnit* ret = 0;
    size_t recycleSize = t;
    while (
        (recycleSize <= RSIZE) && 
        (_recycleList[recycleSize-1].empty())
    ) { ++recycleSize; }

    if (recycleSize <= RSIZE) {
        ret = _recycleList[recycleSize-1].pop_front();
        if (recycleSize -= t) {
#ifdef DELAYMGR_DEBUG
            cout << "Recycling Ret: " << ret << " to bin: " << recycleSize << endl;
#endif
            _recycleList[recycleSize-1].push_front(ret + t);
        }
    }
    else {
        size_t remain = _activeBlock->getRemainSize();
        if(!_activeBlock->getArray(t, ret)) {
            if(remain) {
#ifdef DELAYMGR_DEBUG
            cout << _activeBlock << " Recycling ret: " << ret << " to bin: " << remain << endl;
#endif
                _recycleList[remain-1].push_front(ret);
            }

            _activeBlock = new dBlock(_activeBlock, _blockSize);
            _activeBlock->getArray(t, ret);
        }
    }
    return ret;
}

// Get the currently allocated number of Block's
size_t 
DelayMgr::getNumBlocks() const {
    int num = 0;
    dBlock* tmp = _activeBlock;
    while(tmp) {
        tmp = tmp->getNextBlock();
        num += 1;
    }
    return num;
}

// Print out info of delay manager
#ifdef DELAYMEM_DEBUG
ostream& 
operator<<(ostream& os, DelayMgr& mgr) {
    cout << "=========================================" << endl
            << "=         Dtable Memory Manager         =" << endl
            << "=========================================" << endl
            << "* Dunit size            : " << sizeof(dUnit) << " Bytes" << endl
            << "* Block size            : " << sizeof(dUnit) * mgr._blockSize << " Bytes" << endl
            << "* Number of blocks      : " << mgr.getNumBlocks() << endl
            << "* Free mem in last block: " << mgr._activeBlock->getRemainSize()
            << endl
            << "* Recycle list          : " << endl;
    for (size_t i=0, line=1;i<RSIZE;i++) {
        size_t listSize = mgr._recycleList[i].numElem();
        if (listSize)
        {
            cout << "[" << i+1 << "] = " << setw(10) << left << listSize;
            ++line;
        }
        if(!line%PRINTBLOCK) cout << endl;
    }
    cout << endl
            << "* Block content         :  " << endl;
    dBlock* it = mgr._activeBlock;
    size_t blockNum = 0;
    while (it != nullptr)
    {
        cout << "Block " << blockNum << endl;
        cout << *it;
        cout << endl;
        it = it->getNextBlock(), ++blockNum;
    }
}
#endif
