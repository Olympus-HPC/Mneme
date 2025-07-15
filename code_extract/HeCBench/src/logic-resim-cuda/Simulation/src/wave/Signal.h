#ifndef SIGNAL_H
#define SIGNAL_H

#include "util.h"

#include <cstddef>
#include <vector>
using std::vector;

#define ULL unsigned long long
typedef unsigned long long tUnit;

struct Event {
    tUnit t;
    char v;

#ifdef DEBUG_LIB
	bool operator== (const Event& e) {
		return t == e.t && v == e.v;
	}
#endif
};
typedef vector<Event> EventList;

typedef class TransitionHistory {
	private:
		EventList History;
		size_t Size;
		const static Event dumb;
		bool deleted;

	public:
		TransitionHistory(): History(), Size(0), deleted(false) {
			History.reserve(64);
			this->push_back(0, valueX);
		};

		void push_back (tUnit, char);
		void pop_back  ();

		const Event& operator[] (const size_t) const;
		const tUnit& getTime    (const size_t) const;
		const char&  getValue   (const size_t) const;

		inline const size_t& size () const { return Size; }
		inline       void clear()       { EventList().swap(History); Size = 0; deleted = true; }
		inline       void resize(size_t &s){ History.clear(); Size = s; History.resize(s); }
		inline const bool isDeleted()   { return deleted; }
		inline 		 void push_end()    { if ( History.back().t == (tUnit)-1 ) return ; History.push_back(dumb); ++Size; }

		inline const Event& front() { return History[0]; }
		inline       Event* getHis() { return &History.front(); }
		inline const Event& back () { return History.back(); }
		inline EventList::iterator begin() { return History.begin(); }
		inline EventList::iterator end  () { return History.end();   }

} tHistory;

#endif
