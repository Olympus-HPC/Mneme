#include <unistd.h>
#include <sys/times.h>
#include <stdexcept>
#include "timer.h"

Stopwatch::Stopwatch() : m_start(static_cast<clock_t>(-1))
{
}

Stopwatch::~Stopwatch()
{
}

void Stopwatch::start()
{
  struct tms t;
  m_start = times(&t);

  if (m_start == static_cast<clock_t>(-1)) {
    throw std::runtime_error("Error calling times()");
  }
}

double Stopwatch::stop()
{
  struct tms t;
  clock_t stop = times(&t);

  if (m_start == static_cast<clock_t>(-1)) {
    throw std::runtime_error("Start time not set");
  }

  if (stop == static_cast<clock_t>(-1)) {
    throw std::runtime_error("Error calling times()");
  }

  return (static_cast<double>(stop - m_start)) / (static_cast<double>(sysconf(_SC_CLK_TCK)));
}
