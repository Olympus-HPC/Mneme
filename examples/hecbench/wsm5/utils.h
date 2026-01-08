// thread block sizes
#include "device_traits.hpp"
#define XXX 8
#define YYY 8

// data allocation on host and device
// data initialization on host
// host to device copy
# define TODEV(A,s) A = (float*) malloc ((s) * sizeof(float)); \
                    float *A##_d;\
                    { \
                    for (int i = 0; i < s; i++) A[i] = 0.001; \
                    DEVICE_CHECK(Device::deviceMalloc(reinterpret_cast<void **>(&A##_d),((s))*sizeof(float)));\
                    DEVICE_CHECK(Device::deviceCopy(reinterpret_cast<void *>(A##_d), A, (s)*sizeof(float), Device::memcpyHostToDeviceKind()));\
                    }

// device to host copy
# define FROMDEV(A,s) DEVICE_CHECK(Device::deviceCopy(A, A##_d, (s)*sizeof(float), Device::memcpyDeviceToHostKind()));

// deallocation host and device memory
# define FREE(A) free(A);\
                 DEVICE_CHECK(Device::deviceFree(A##_d));

# define TODEV3(A) TODEV(A,d3)
# define TODEV2(A) TODEV(A,d2)
# define FROMDEV3(A) FROMDEV(A,d3)
# define FROMDEV2(A) FROMDEV(A,d2)

