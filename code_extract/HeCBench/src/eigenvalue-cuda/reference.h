
uint calNumEigenValuesLessThan(const float *diagonal,
                               const float *offDiagonal,
                               const uint  length,
                               const float x);

uint eigenValueCPUReference(float * diagonal,
                            float * offDiagonal,
                            uint    length,
                            float * eigenIntervals,
                            float * newEigenIntervals,
                            float tolerance);

int isComplete(float * eigenIntervals, const int length, const float tolerance);

void computeGerschgorinInterval(float * lLimit,
                                float * uLimit,
                                const float * diagonal,
                                const float * offDiagonal,
                                const uint  length);
