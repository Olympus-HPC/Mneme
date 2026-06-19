#pragma once

namespace mneme {

// Install a fatal-signal handler so a crash during recording is reported
// instead of producing a bare core dump. Repeated calls are no-ops.
void installCrashHandler();

} // namespace mneme
