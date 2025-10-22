import sys
from typing import List, Optional, Union

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

console = Console(soft_wrap=False)


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


def _fmt_delta(v: Optional[float]):
    if v is None:
        return ""
    style = "bold green" if v > 0 else ("red" if v < 0 else "dim")
    sign = "-" if v > 0 else "+"
    return Text(f"{sign}{abs(v):.2f}", style=style)


class PrettyTablePrinter:
    def __init__(self):
        self._step = 0
        self._prev_time = 0
        self._table = Table(
            show_header=True,  # enable header row
            header_style="bold cyan",  # style for headers
            box=None,  # border style (None for no lines)
            padding=(0, 1),
        )

        self._table.add_column("#", justify="left", no_wrap=True)
        self._table.add_column("Pass", justify="center", no_wrap=False)
        self._table.add_column("Exec Time", justify="right", no_wrap=True)
        self._table.add_column("Delta", justify="right", no_wrap=True)
        self._table.add_column("Speedup", justify="right", no_wrap=True)
        self._table.add_column("Verified", justify="right", no_wrap=True)
        self._console = None

    def open(self):
        if self._console is None:
            self._console = Live(self._table, refresh_per_second=10)
        self._console.__enter__()
        return

    def __enter__(self):
        self.open()
        return self  # so you can use "as" in with-block

    def __exit__(self, exc_type, exc_value, traceback):
        self._console.__exit__(exc_type, exc_value, traceback)

    def print_pass_result(
        self,
        pass_name: Union[str, List[str]],
        exec_time: float,
        verified,
        prev_time: Optional[float] = None,
    ) -> None:
        """
        Print a single result line for an experiment step.

        - pass_name: e.g. 'function<eager-inv>(bdce)'
        - exec_time: current execution time (float)
        - prev_time: previous step's time (float) or None for the first step
        """

        if isinstance(pass_name, list):
            pass_name = ",".join(pass_name)
        d_prev = _fmt_delta(0)
        speedup = 1.0
        if self._prev_time != 0:
            d_prev = _fmt_delta(self._prev_time - exec_time)
            speedup = float(self._prev_time) / float(exec_time)

        self._prev_time = exec_time
        self._step += 1
        self._table.add_row(
            str(self._step),
            pass_name,
            f"{exec_time:.2f}",
            d_prev,
            f"{speedup:.5f}",
            f"{verified}",
        )
        self._console.update(self._table)


def print_experiment_status(
    exp_id,
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
        msg = f"{ICONS['fail']} [{exp_id}] Experiment with hash {exp_hash} failed."
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
        msg = f"{ICONS['rocket']} [{exp_id}] Experiment with hash {exp_hash} shows speedup {sp} Best: {bp} verified:{ver}"
        if color_ok:
            console.print(Text(msg, style="bold green"))
        else:
            print(msg)
        return

    # Faster than baseline (but not new best) -> green
    if speedup > baseline_speedup:
        msg = f"{ICONS['rocket']} [{exp_id}] Experiment with hash {exp_hash} shows speedup {sp} Best: {bp} verified:{ver}"
        if color_ok:
            console.print(Text(msg, style="green"))
        else:
            print(msg)
        return

    # Slower -> orange (use 'orange3' if available, else 'yellow')
    msg = f"{ICONS['slow']} [{exp_id}] Experiment with hash {exp_hash} shows slowdown: {sp} Best: {bp} verified:{ver}"
    if color_ok:
        # 'orange3' is a Rich color; if your terminal downgrades, Rich maps it reasonably.
        console.print(Text(msg, style="orange3"))
    else:
        print(msg)
