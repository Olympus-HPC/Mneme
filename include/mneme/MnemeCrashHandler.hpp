#pragma once

namespace mneme {

/// Install a handler for fatal signals (SIGSEGV, SIGABRT, SIGFPE, SIGILL,
/// SIGBUS) that writes an async-signal-safe banner to stderr and then re-raises
/// the signal so its default action (core dump / termination) still runs.
///
/// Intended to be called once after main has started — e.g. from a recorder
/// singleton constructor — so that a crash while mneme is recording is clearly
/// attributed instead of producing a bare core dump. Repeated calls are no-ops.
void installCrashHandler();

} // namespace mneme
