#include <math.h>

#ifdef _OPENMP
typedef struct __attribute__((__aligned__(32)))
{
  double x, y, z;
}
double3;
#endif

void ref_Extrema(const double* history, const int history_length, double *result, int& result_length)
{
  result[0] = history[0];

  int eidx = 0;
  for (int i = 1; i < history_length - 1; i++)
    if ((history[i] > result[eidx] && history[i] > history[i + 1]) ||
        (history[i] < result[eidx] && history[i] < history[i + 1]))
      result[++eidx] = history[i];

  result[++eidx] = history[history_length - 1];
  result_length = eidx + 1;
}

void ref_Execute(const double* history, const int history_length,
             double *extrema, int* points, double3 *results, int *results_length )
{
  int extrema_length = 0;
  ref_Extrema(history, history_length, extrema, extrema_length);

  int pidx = -1, eidx = -1, ridx = -1;

  for (int i = 0; i < extrema_length; i++)
  {
    points[++pidx] = ++eidx;
    double xRange, yRange;
    while (pidx >= 2 && (xRange = fabs(extrema[points[pidx - 1]] - extrema[points[pidx]]))
           >= (yRange = fabs(extrema[points[pidx - 2]] - extrema[points[pidx - 1]])))
    {
      double yMean = 0.5 * (extrema[points[pidx - 2]] + extrema[points[pidx - 1]]);

      if (pidx == 2)
      {
        results[++ridx] = { 0.5, yRange, yMean };
        points[0] = points[1];
        points[1] = points[2];
        pidx = 1;
      }
      else
      {
        results[++ridx] = { 1.0, yRange, yMean };
        points[pidx - 2] = points[pidx];
        pidx -= 2;
      }
    }
  }

  for (int i = 0; i <= pidx - 1; i++)
  {
    double range = fabs(extrema[points[i]] - extrema[points[i + 1]]);
    double mean = 0.5 * (extrema[points[i]] + extrema[points[i + 1]]);
    results[++ridx] = { 0.5, range, mean };
  }

  *results_length = ridx + 1;
}

void reference (const double *history,
                const int *history_lengths,
                    double *extrema,
                       int * points,
                    double3 *results,
                    int *result_length,
                const int num_history )
{
  for (int i = 0; i < num_history; i++) {
    const int offset = history_lengths[i];
    const int history_length = history_lengths[i+1] - offset;
    ref_Execute(history + offset, history_length,
                extrema + offset, points + offset,
                results + offset, result_length + i);
  }
}

