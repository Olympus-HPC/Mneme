
#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <memory>
#include <ostream>
#include <type_traits>

namespace Timer {
class Timer {
 public:
  Timer();
  void reset();
  virtual void startTimer();
  virtual void stopTimer();
  /* Reports the total number of ns mod 1e9 */
  virtual int elapsed_ns();
  /* Reports the total number of us mod 1e9 */
  virtual int elapsed_us();
  /* Reports the total number of ms mod 1e9 */
  virtual int elapsed_ms();
  /* Reports the total number of s mod 1e9 */
  virtual int elapsed_s();

  /* Reports the total number of ns mod 1e9 */
  virtual int instant_ns();
  /* Reports the total number of us mod 1e9 */
  virtual int instant_us();
  /* Reports the total number of ms mod 1e9 */
  virtual int instant_ms();
  /* Reports the total number of s mod 1e9 */
  virtual int instant_s();

  friend std::ostream &operator<<(std::ostream &os,
                                  const Timer &t);

  static constexpr const int us_to_ns = 1e3;
  static constexpr const int ms_to_ns = 1e6;
  static constexpr const int s_to_ns = 1e9;
  static constexpr const int ms_to_us = 1e3;
  static constexpr const int s_to_us = 1e6;
  static constexpr const int s_to_ms = 1e3;

 private:
  virtual void updateTimer();

  struct TimerData;
  std::shared_ptr<TimerData> data;
};
};

#endif
