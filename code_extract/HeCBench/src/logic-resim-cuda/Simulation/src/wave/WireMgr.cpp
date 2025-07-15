#include "WireMgr.h"
using std::make_pair;

void 
WireMgr::addsWire(string& name, size_t size, size_t serial, bool isInput) {
	sWire* wire = new sWire(name, 1, TransList[serial]);
	if (isInput) {
		nameMapper.insert(make_pair(name, (Wire*)wire));
	}
	WireList.push_back((Wire*)wire);
	sWireList.push_back(wire);
}

void 
WireMgr::addmWire(string& name, size_t size, size_t serial, size_t msb, size_t lsb, bool isInput) {
	mWire* wire = new mWire(name, size, msb, lsb);
	sWire* swire;

	for (size_t i=0;i<size;++i) {
		wire->push(i, TransList[i+serial]);
		swire = new sWire(name +"\\[" + to_string(msb>lsb?msb-i:msb+i) + "\\]", 1, TransList[i+serial]);
		sWireList.push_back(swire);
	}

	if (isInput) {
		nameMapper.insert(make_pair(name, (Wire*)wire));
	}
	WireList.push_back((Wire*)wire);
}

void 
WireMgr::connectWire(size_t l, size_t r) {
	if (r > 3) {
		delete TransList[l-4];
		TransList[l-4] = TransList[r-4];
	}
	else {
		TransList[l-4]->push_back(0, r);
	}
}
