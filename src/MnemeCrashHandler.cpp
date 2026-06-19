#include "mneme/MnemeCrashHandler.hpp"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>

namespace mneme {

namespace {

void crashSignalHandler(int Signal) {
  // A second fatal signal (possibly on another thread) must not re-enter the
  // handler; bail straight to _exit with the conventional 128 + signal code.
  static std::atomic_flag InHandler = ATOMIC_FLAG_INIT;
  if (InHandler.test_and_set())
    _exit(128 + Signal);

  // Only async-signal-safe functions are allowed here. spdlog/iostream are NOT
  // safe, especially during program teardown.
  static constexpr char Msg[] = "=== PROGRAM CRASHED (see core dump) ===\n";
  (void)!write(STDERR_FILENO, Msg, sizeof(Msg) - 1);

  // SA_RESETHAND already restored SIG_DFL, so re-raising runs the default
  // action (core dump / termination) for the signal we trapped.
  std::raise(Signal);
}

} // namespace

void installCrashHandler() {
  // The recorder singleton constructs once per process, but guard anyway so the
  // handler is never installed twice.
  static std::atomic_flag Installed = ATOMIC_FLAG_INIT;
  if (Installed.test_and_set())
    return;

  struct sigaction SigAction {};
  SigAction.sa_handler = crashSignalHandler;
  sigemptyset(&SigAction.sa_mask);
  SigAction.sa_flags = SA_RESETHAND;
  for (int Sig : {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS})
    sigaction(Sig, &SigAction, nullptr);
}

} // namespace mneme
