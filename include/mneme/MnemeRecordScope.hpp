#pragma once

namespace mneme {

namespace detail {
// Internal launch-time query; state is owned by the shared mnemert runtime.
bool scopeAllowsRecording(const void *Kernel) noexcept;
} // namespace detail

// Controls eligibility in scoped filtering mode on the calling host thread.
// Scopes must be destroyed in reverse construction order on that same thread.
class record_scope {
public:
  explicit record_scope(bool Enabled) noexcept;

  template <typename Return, typename... Args>
  record_scope(Return (*Kernel)(Args...), bool Enabled)
      : record_scope(reinterpret_cast<const void *>(Kernel), Enabled) {}

  ~record_scope() noexcept;

  record_scope(const record_scope &) = delete;
  record_scope &operator=(const record_scope &) = delete;
  record_scope(record_scope &&) = delete;
  record_scope &operator=(record_scope &&) = delete;

private:
  record_scope(const void *Kernel, bool Enabled);
  friend bool detail::scopeAllowsRecording(const void *Kernel) noexcept;

  static thread_local record_scope *Top;
  const void *Kernel;
  bool Enabled;
  record_scope *Previous;
};

} // namespace mneme
