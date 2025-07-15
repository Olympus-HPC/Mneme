/*
 * function to calculate the number of eigenvalues less than (x) for a tridiagonal symmetric matrix
 */

#include <math.h>

uint calNumEigenValuesLessThan(const float *diagonal,
                               const float *offDiagonal,
                               const uint  length,
                               const float x)
{
  uint count = 0;

  float prev_diff = (diagonal[0] - x);
  count += (prev_diff < 0)? 1 : 0;
  for(uint i = 1; i < length; i += 1)
  {
    float diff = (diagonal[i] - x) - ((offDiagonal[i-1] *
          offDiagonal[i-1])/prev_diff);

    count += (diff < 0)? 1 : 0;
    prev_diff = diff;
  }
  return count;
}
/*
 * Calculates the eigenvalues of a tridiagonal symmetrix matrix
 */
uint eigenValueCPUReference(float * diagonal,
                            float * offDiagonal,
                            uint    length,
                            float * eigenIntervals,
                            float * newEigenIntervals,
                            float tolerance)
{
  uint offset = 0;
  for(uint i =0; i < length; ++i)
  {
    uint lid = 2*i;
    uint uid = lid + 1;

    uint eigenValuesLessLowerBound = calNumEigenValuesLessThan(diagonal,
        offDiagonal, length, eigenIntervals[lid]);
    uint eigenValuesLessUpperBound = calNumEigenValuesLessThan(diagonal,
        offDiagonal, length, eigenIntervals[uid]);

    uint numSubIntervals = eigenValuesLessUpperBound - eigenValuesLessLowerBound;

    if(numSubIntervals > 1)
    {
      float avgSubIntervalWidth = (eigenIntervals[uid] -
          eigenIntervals[lid])/numSubIntervals;

      for(uint j=0; j < numSubIntervals; ++j)
      {
        uint newLid = 2* (offset+j);
        uint newUid = newLid + 1;

        newEigenIntervals[newLid] = eigenIntervals[lid]       + j * avgSubIntervalWidth;
        newEigenIntervals[newUid] = newEigenIntervals[newLid] +     avgSubIntervalWidth;
      }
    }
    else if(numSubIntervals == 1)
    {
      float lowerBound = eigenIntervals[lid];
      float upperBound = eigenIntervals[uid];

      float mid        = (lowerBound + upperBound)/2;

      uint newLid = 2* offset;
      uint newUid = newLid + 1;

      if(upperBound - lowerBound < tolerance)
      {
        newEigenIntervals[newLid] = lowerBound;
        newEigenIntervals[newUid] = upperBound;
      }
      else if(calNumEigenValuesLessThan(diagonal,offDiagonal, length,
            mid) == eigenValuesLessUpperBound)
      {
        newEigenIntervals[newLid] = lowerBound;
        newEigenIntervals[newUid] = mid;
      }
      else
      {
        newEigenIntervals[newLid] = mid;
        newEigenIntervals[newUid] = upperBound;
      }
    }
    offset += numSubIntervals;
  }
  return offset;
}

/*
 * Checks if the difference between lowerlimit and upperlimit of all intervals is below
 * tolerance levels
 */
int isComplete(float * eigenIntervals, const int length, const float tolerance)
{
  for(int i=0; i< length; i++)
  {
    uint lid = 2*i;
    uint uid = lid + 1;
    if(eigenIntervals[uid] - eigenIntervals[lid] >= tolerance)
    {
      return 1;
    }
  }
  return 0;
}

/*
 * function to calculate the gerschgorin interval(lowerbound and upperbound of the eigenvalues)
 *                                              of a tridiagonal symmetric matrix
 */
void computeGerschgorinInterval(float * lLimit,
                                float * uLimit,
                                const float * diagonal,
                                const float * offDiagonal,
                                const uint length)
{

  float lowerLimit = diagonal[0] - fabs(offDiagonal[0]);
  float upperLimit = diagonal[0] + fabs(offDiagonal[0]);

  for(uint i = 1; i < length-1; ++i)
  {
    float r =  fabs(offDiagonal[i-1]) + fabs(offDiagonal[i]);
    lowerLimit = (lowerLimit > (diagonal[i] - r))? (diagonal[i] - r): lowerLimit;
    upperLimit = (upperLimit < (diagonal[i] + r))? (diagonal[i] + r): upperLimit;
  }

  lowerLimit = (lowerLimit > (diagonal[length-1] - fabs(offDiagonal[length-2])))?
    (diagonal[length-1] - fabs(offDiagonal[length-2])): lowerLimit;
  upperLimit = (upperLimit < (diagonal[length-1] + fabs(offDiagonal[length-2])))?
    (diagonal[length-1] + fabs(offDiagonal[length-2])): upperLimit;

  *lLimit = lowerLimit;
  *uLimit = upperLimit;

}
