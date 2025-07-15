#include "util.h"

char 
vEncoder(char c) {
	switch (c)
	{
		case '0': return value0; break;
		case '1': return value1; break;
		case 'x': return valueX; break;
		case 'z': return valueZ; break;
		default : return valueZ; break;
	}
}

char 
vDecoder(char c) {
	switch (c)
	{
		case value0: return '0'; break;
		case value1: return '1'; break;
		case valueX: return 'x'; break;
		case valueZ: return 'z'; break;
		default    : return 'z'; break;
	}
}