// constants for host and kernel codes
#define x_dim   11
#define y_dim   40
#define batch   4
#define states  3

// forward type
typedef double fArray[y_dim+1][batch][states-1];

// transition type
typedef double tArray[batch][states-1][states];

// likelihood type
typedef double lArray[2][batch][states-1];

// start type
typedef double sArray[states-1];

