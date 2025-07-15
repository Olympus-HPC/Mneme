#ifndef Wire_H
#define Wire_H

#include <string>
using std::string;
using std::to_string;

#include "Signal.h"
#include "util.h"

class Wire {
	protected:
		string Name;
		size_t Size;

	public:
		Wire(string name, size_t size)
			:Name(name), Size(size) {};

		inline const string& getName  () const { return Name; }
		inline const size_t& getSize  () const { return Size; }
#ifdef DEBUG
		friend ostream& operator<<(ostream& os, Wire& wire) {
			os << "Wire: " << wire.Name << " Width: " << wire.Size << endl;
			return os;
		}
#endif
	virtual inline string getDecl  () = 0;
	virtual void   addSignal(char*, tUnit) = 0;
};

typedef 
class singleWire: public Wire {
	private:
		tHistory* His;

	public:
		singleWire(string name, size_t s, tHistory* t)
			:Wire(name, s), His(t) { }


		inline 	tHistory* getHis() const { return His; }
		inline  void      remove() 		 { His->clear(); }
		inline  string    getDecl()      { return Name; }

		void addSignal (char*, tUnit);
} sWire;


typedef 
class multipleWire: public Wire {
	private:
		vector<tHistory*> HisList;
		size_t msb, lsb;

	public:
		multipleWire(string name, size_t s, size_t m, size_t l)
			: Wire(name, s), msb(m), lsb(l) { HisList.resize(s); }
		inline void      push(size_t i, tHistory* his) { HisList[i] = his; }
		inline tHistory* operator[](int index) const { return HisList[index]; }
		inline size_t 	 getMsb() { return msb; }
		inline size_t 	 getLsb() { return lsb; }

		void   addSignal(char*, tUnit);
		inline string getDecl() { return Name + " [" + to_string(msb) + ':' + to_string(lsb) + ']'; }
} mWire;

#endif