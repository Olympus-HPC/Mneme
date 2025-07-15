
#include "timer.hpp"

#include <time.h>
#include <sys/time.h>

#include <iomanip>

namespace Timer {

struct Timer::TimerData {
  struct timespec startTime, endTime;
  struct timespec elapsedTime;
  struct timespec deltaTime;
  bool started;
  int err;
};

Timer::Timer() : data(new Timer::TimerData) { reset(); }

void Timer::reset() {
  data->startTime.tv_nsec = 0;
  data->startTime.tv_sec = 0;
  data->endTime.tv_nsec = 0;
  data->endTime.tv_sec = 0;
  data->elapsedTime.tv_nsec = 0;
  data->elapsedTime.tv_sec = 0;
  data->deltaTime.tv_nsec = 0;
  data->deltaTime.tv_sec = 0;
  data->started = false;
}

void Timer::startTimer() {
  if(!data->started) {
    data->started = true;
    data->deltaTime.tv_nsec = 0;
    data->deltaTime.tv_sec = 0;
    data->err = clock_gettime(CLOCK_PROCESS_CPUTIME_ID,
                              &data->startTime);
  }
}

void Timer::stopTimer() {
  if(data->started) {
    data->err = clock_gettime(CLOCK_PROCESS_CPUTIME_ID,
                              &data->endTime);
    data->started = false;
    updateTimer();
  }
}

void Timer::updateTimer() {
  /* Update the "instantaneous" time elapsed */
  data->deltaTime.tv_sec =
      data->endTime.tv_sec - data->startTime.tv_sec;
  data->deltaTime.tv_nsec =
      data->endTime.tv_nsec - data->startTime.tv_nsec;
  if(data->deltaTime.tv_nsec < 0) {
    data->deltaTime.tv_nsec += 1 * s_to_ns;
    data->deltaTime.tv_sec -= 1;
  }
  /* Update the total time elapsed */
  data->elapsedTime.tv_sec += data->deltaTime.tv_sec;
  data->elapsedTime.tv_nsec += data->deltaTime.tv_nsec;
  if(data->elapsedTime.tv_nsec > s_to_ns) {
    data->elapsedTime.tv_nsec -= 1 * s_to_ns;
    data->elapsedTime.tv_sec += 1;
  }
}

int Timer::elapsed_ns() {
  /* Reports the total number of us mod 1e9 */
  return data->elapsedTime.tv_nsec % s_to_ns;
}

int Timer::elapsed_us() {
  /* Reports the total number of us mod 1e9 */
  int us = data->elapsedTime.tv_nsec / us_to_ns;
  int s = data->elapsedTime.tv_sec % (((int)1e9) / s_to_us);
  us += s_to_us * s;
  return us;
}

int Timer::elapsed_ms() {
  /* Reports the total number of ms mod 1e9 */
  int ms = data->elapsedTime.tv_nsec / ms_to_ns;
  int s = data->elapsedTime.tv_sec % (((int)1e9) / s_to_ms);
  ms += s_to_ms * s;
  return ms;
}

int Timer::elapsed_s() {
  /* Reports the total number of s mod 1e9 */
  return data->elapsedTime.tv_sec;
}

int Timer::instant_ns() {
  /* Reports the total number of us mod 1e9 */
  return data->deltaTime.tv_nsec % s_to_ns;
}

int Timer::instant_us() {
  /* Reports the total number of us mod 1e9 */
  int us = data->deltaTime.tv_nsec / us_to_ns;
  int s = data->deltaTime.tv_sec % (((int)1e9) / s_to_us);
  us += s_to_us * s;
  return us;
}

int Timer::instant_ms() {
  /* Reports the total number of ms mod 1e9 */
  int ms = data->deltaTime.tv_nsec / ms_to_ns;
  int s = data->deltaTime.tv_sec % (((int)1e9) / s_to_ms);
  ms += s_to_ms * s;
  return ms;
}

int Timer::instant_s() {
  /* Reports the total number of s mod 1e9 */
  return data->deltaTime.tv_sec;
}

std::ostream &operator<<(std::ostream &os, const Timer &t) {
  os << "Instantaneous Time: " << t.data->deltaTime.tv_sec
     << "." << std::setfill('0') << std::setw(9)
     << t.data->deltaTime.tv_nsec << "\n";
  os << "Total Time: " << t.data->elapsedTime.tv_sec << "."
     << std::setfill('0') << std::setw(9)
     << t.data->elapsedTime.tv_nsec;
  return os;
}
};
