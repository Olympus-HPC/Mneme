import sys

from rich.console import Console
from rich.text import Text

console = Console()


def _supports_color() -> bool:
    # rich knows if we're in a real terminal and what color system is available
    return console.is_terminal and console.color_system is not None


def _supports_emoji() -> bool:
    enc = (sys.stdout.encoding or "").lower()
    try:
        "🚀".encode(enc or "utf-8")
        return True
    except Exception:
        return False


# Choose icons with graceful fallback
USE_EMOJI = _supports_emoji()
ICONS = {
    "rocket": "🚀" if USE_EMOJI else "[rocket]",
    "fail": "❌" if USE_EMOJI else "[x]",
    "slow": "🐢" if USE_EMOJI else "[slow]",
    "correct": "✅" if USE_EMOJI else "[ok]",
    "wrong": "❓" if USE_EMOJI else "[nok]",
}


def print_experiment_status(
    exp_hash,
    success,
    verified=False,
    speedup=None,  # only needed if success=True
    best_speedup=None,  # only needed if success=True
    baseline_speedup=1.0,  # "faster than baseline" threshold
):
    """
    Prints a fancy status line for an experiment run.

    Rules:
      1) Failure -> red: "X Experiment with hash {hash} failed."
      2) Success & speedup > baseline -> green: "[rocket] ... speedup {speedup:.2f} Best: {best:.2f}"
      3) Success & speedup > current best -> bold green (supersedes #2)
      4) Success but slower (<= baseline) -> orange: "[icon] ... slowdown: {speedup:.2f} Best: {best:.2f}"
    """

    color_ok = _supports_color()

    if not success:
        msg = f"{ICONS['fail']} Experiment with hash {exp_hash} failed."
        if color_ok:
            console.print(Text(msg, style="red"))
        else:
            print(msg)
        return

    # For success cases we need speed numbers
    if speedup is None or best_speedup is None:
        raise ValueError("speedup and best_speedup must be provided when success=True")

    # Format numbers with 2 decimals
    sp = f"{speedup:.2f}"
    bp = f"{best_speedup:.2f}"
    ver = ICONS["wrong"]

    if verified and success:
        ver = ICONS["correct"]

    # Faster than current best -> bold green
    if speedup > best_speedup:
        msg = f"{ICONS['rocket']} Experiment with hash {exp_hash} shows speedup {sp} Best: {bp} verified:{ver}"
        if color_ok:
            console.print(Text(msg, style="bold green"))
        else:
            print(msg)
        return

    # Faster than baseline (but not new best) -> green
    if speedup > baseline_speedup:
        msg = f"{ICONS['rocket']} Experiment with hash {exp_hash} shows speedup {sp} Best: {bp} verified:{ver}"
        if color_ok:
            console.print(Text(msg, style="green"))
        else:
            print(msg)
        return

    # Slower -> orange (use 'orange3' if available, else 'yellow')
    msg = f"{ICONS['slow']} Experiment with hash {exp_hash} shows slowdown: {sp} Best: {bp} verified:{ver}"
    if color_ok:
        # 'orange3' is a Rich color; if your terminal downgrades, Rich maps it reasonably.
        console.print(Text(msg, style="orange3"))
    else:
        print(msg)
