#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <sys/times.h>

class Stopwatch {
  public:
    Stopwatch();
    ~Stopwatch();
    void start();
    double stop();

  private:
    clock_t m_start;
};

#endif
