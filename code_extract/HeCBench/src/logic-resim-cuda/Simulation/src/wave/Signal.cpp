#include <climits>
#include "Signal.h"

const Event tHistory::dumb = {(tUnit)-1, valueZ};

void 
tHistory::push_back(tUnit t, char v) {
	while (Size && (back().t >= t)) pop_back();
	if (Size && back().v == v) return;
	History.push_back({t, v});
	Size += 1;
}

void 
tHistory::pop_back() {
	if (Size == 0) return;
	History.pop_back();
	Size -= 1;
}

const Event& 
tHistory::operator[](const size_t i) const {
    if (i >= Size)
		return dumb;
	return History[i];
}

const tUnit& 
tHistory::getTime   (const size_t i) const {
	if (i < Size) return History[i].t;
	else          return dumb.t;
}

const char& 
tHistory::getValue  (const size_t i) const {
	if (i < Size) return History[i].v;
	else          return dumb.v;
}