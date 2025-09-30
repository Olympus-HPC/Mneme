import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from mneme.db import MnemeDB
from mneme.llvm import debug, module
from mneme.logging import logger
from mneme.proteus import jit
from mneme.recorded_execution import RecordedExecution
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.measure import Measurement
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree


def _copy_or_move(sources, dest, move=False):
    for s in sources:
        if not s.exists():
            continue

        exec = RecordedExecution.from_json(str(s))
        new_ir_files = []
        for ll in exec.llvm_files:
            dest_ll = dest / Path(ll).name
            if move:
                shutil.move(ll, dest_ll)
            else:
                shutil.copy(ll, dest_ll)
            new_ir_files.append(str(dest_ll))

        exec.llvm_files = new_ir_files

        for _, kernel in exec.items():
            kfn = Path(kernel.prologue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.prologue.fn = str(dest_kfn)

            kfn = Path(kernel.epilogue.fn)
            dest_kfn = dest / kfn.name
            if move:
                shutil.move(kfn, dest_kfn)
            else:
                shutil.copy(kfn, dest_kfn)
            kernel.epilogue.fn = str(dest_kfn)

        json_fn = dest / s.name
        exec.to_json(str(json_fn))
        if move:
            s.unlink()


class Clean:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument("dbs", help="Recorded Mneme DB files", nargs="+")
        parser.set_defaults(func=Clean.run)

    @staticmethod
    def run(args):
        dbs = args.dbs
        for db in dbs:
            if not Path(db).exists():
                raise FileNotFoundError(f"Mneme Database '{db}' does not exist")

        ir_records = set()
        mneme_db_files = set()
        for db in dbs:
            exec = RecordedExecution.from_json(db)
            ir_records |= {Path(e) for e in exec.llvm_files}
            for _, kernel in exec.items():
                mneme_db_files.add(Path(kernel.prologue.fn))
                mneme_db_files.add(Path(kernel.epilogue.fn))
            Path(db).unlink()

        for fn in ir_records:
            if fn.exists():
                fn.unlink()

        for fn in mneme_db_files:
            if fn.exists():
                fn.unlink()


class Copy:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "dbs",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be copied to",
        )
        parser.set_defaults(func=Copy.run)

    @staticmethod
    def run(args):
        paths = args.dbs
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]
        _copy_or_move(sources, dest)


class Move:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "dbs",
            nargs="+",
            help="Source paths of Mneme DB records followed by a destination path to which data will be moved to",
        )
        parser.set_defaults(func=Move.run)

    @staticmethod
    def run(args):
        paths = args.dbs
        if len(paths) < 2:
            raise ValueError(
                f"Please provide both source and destination directories {paths}"
            )

        *_sources, _dest = paths
        dest = Path(_dest).absolute()
        if not dest.exists():
            raise RuntimeError(f"Destination target '{str(dest)}' does not exist")
        if not dest.is_dir():
            raise NotADirectoryError(
                f"Destination target '{str(dest)}' is not a directory"
            )

        sources = [Path(s) for s in _sources]

        _copy_or_move(sources, dest, move=True)


class Summary:
    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "-db",
            "--record-database",
            dest="db",
            required=True,
            help="Path to Mneme JSON/db file",
        )

        parser.add_argument(
            "results",
            help="CSV database containing the performance data of the tuning runs",
        )

        parser.add_argument(
            "--json",
            dest="json",
            help="Emit to stdout as a json file",
            action=argparse.BooleanOptionalAction,
            required=False,
        )

        parser.set_defaults(json=False, func=Summary.analyze)

    @staticmethod
    def compute_speedups(df: pd.DataFrame):
        if (
            "specialize" not in df.columns
            or "max_threads" not in df.columns
            or "min_blocks_per_sm" not in df.columns
            or "exec_time_median" not in df.columns
            or "passes" not in df.columns
        ):
            raise ValueError(f"DataFrame missing required columns. {df.columns}")

        # Coerce types we rely on
        spec = df["specialize"]
        max_threads = df["max_threads"]
        min_blocks_per_sm = df["min_blocks_per_sm"]
        exec_time = df["exec_time_median"]
        passes = df["passes"]

        # ---- Baseline: specialize == False AND max_threads == 0
        baseline_mask = (
            (spec == False) & (max_threads == 0) & (passes == "default<O3>,globaldce")
        )
        baseline_candidates = exec_time.where(baseline_mask)
        # If multiple rows match baseline, use the minimum time as the baseline
        baseline_time = baseline_candidates.min(skipna=True)

        if not np.isfinite(baseline_time):
            return False, None

        # ---- LB: max_threads != 0 AND min_blocks_per_sm == 0 AND specialize == False
        lb_mask = (
            (max_threads != 0)
            & (min_blocks_per_sm == 0)
            & (spec == False)
            & (passes == "default<O3>,globaldce")
        )
        lb_time = exec_time.where(lb_mask).min(skipna=True)

        # ---- Spec: max_threads != 0 AND min_blocks_per_sm == 0 AND specialize == True
        spec_mask = (
            (max_threads == 0)
            & (min_blocks_per_sm == 0)
            & (spec == True)
            & (passes == "default<O3>,globaldce")
        )
        spec_time = exec_time.where(spec_mask).min(skipna=True)

        spec_tunepipeline_mask = (
            (max_threads == 0)
            & (min_blocks_per_sm == 0)
            & (spec == True)
            & (passes != "default<O3>,globaldce")
        )
        spec_tunepipeline_time = exec_time.where(spec_tunepipeline_mask).min(
            skipna=True
        )

        spec_lb_mask = (
            (max_threads != 0)
            & (min_blocks_per_sm == 0)
            & (spec == True)
            & (passes == "default<O3>,globaldce")
        )
        spec_lb_time = exec_time.where(spec_lb_mask).min(skipna=True)

        # ---- Tune: max_threads != 0 AND min_blocks_per_sm != 0 AND specialize == True
        tune_mask = (
            (max_threads != 0)
            & (min_blocks_per_sm != 0)
            & (spec == True)
            & (passes == "default<O3>,globaldce")
        )
        tune_time = exec_time.where(tune_mask).min(skipna=True)

        # ---- Tune: max_threads != 0 AND min_blocks_per_sm != 0 AND specialize == True
        tune_mask_nospec = (
            (max_threads != 0)
            & (min_blocks_per_sm != 0)
            & (spec == False)
            & (passes == "default<O3>,globaldce")
        )
        tune_time_nospec = exec_time.where(tune_mask_nospec).min(skipna=True)

        pipeline_nospec_no_lb = (
            (max_threads == 0) & (spec == False) & (passes != "default<O3>,globaldce")
        )
        pipeline_nospec_no_lb_time = exec_time.where(pipeline_nospec_no_lb).min(
            skipna=True
        )

        pipeline_nospec_with_lb = (
            (max_threads != 0)
            & (min_blocks_per_sm == 0)
            & (spec == False)
            & (passes != "default<O3>,globaldce")
        )
        pipeline_nospec_with_lb_time = exec_time.where(pipeline_nospec_with_lb).min(
            skipna=True
        )

        pipeline_nospec_with_tunelb = (
            (max_threads != 0)
            & (min_blocks_per_sm != 0)
            & (spec == False)
            & (passes != "default<O3>,globaldce")
        )
        pipeline_nospec_with_tunelb_time = exec_time.where(
            pipeline_nospec_with_tunelb
        ).min(skipna=True)

        pipeline_withspec_with_lb = (
            (max_threads != 0)
            & (min_blocks_per_sm == 0)
            & (spec == True)
            & (passes != "default<O3>,globaldce")
        )
        pipeline_withspec_with_lb_time = exec_time.where(pipeline_withspec_with_lb).min(
            skipna=True
        )

        pipeline_withspec_with_tunelb = (
            (max_threads != 0)
            & (min_blocks_per_sm != 0)
            & (spec == True)
            & (passes != "default<O3>,globaldce")
        )
        pipeline_withspec_with_tunelb_time = exec_time.where(
            pipeline_withspec_with_tunelb
        ).min(skipna=True)

        def speedup(candidate_time):
            # speedup = baseline / candidate (higher is better)
            if not np.isfinite(candidate_time) or candidate_time == 0:
                return np.nan
            return baseline_time / candidate_time

        speedups = {
            "Baseline": baseline_time,
            "TunePipeline": speedup(pipeline_nospec_no_lb_time),
            "LB": speedup(lb_time),
            "LB+TunePipeline": speedup(pipeline_nospec_with_lb_time),
            "TuneLB": speedup(tune_time_nospec),
            "TuneLB+TunePipeline": speedup(pipeline_nospec_with_tunelb_time),
            "Spec": speedup(spec_time),
            "Spec+TunePipeline": speedup(spec_tunepipeline_time),
            "Spec+LB": speedup(spec_lb_time),
            "Spec+LB+TunePipeline": speedup(pipeline_withspec_with_lb_time),
            "Spec+TuneLB": speedup(tune_time),
            "Spec+TuneLB+TunePipeline": speedup(pipeline_withspec_with_tunelb_time),
        }
        return True, speedups

    @staticmethod
    def measure_width(console, renderable, max_width=None):
        # Measure the renderable’s preferred width in the current console
        options = console.options
        if max_width is not None:
            options = options.update(width=max_width)
        m = Measurement.get(console, options, renderable)
        return m.maximum  # or (m.min, m.max) if you want a range

    @staticmethod
    def build_signature_panel(console, signature: str, file_path: str, line_no: int):
        sig = Text(signature, style="bold yellow", no_wrap=False, overflow="fold")
        return Panel(
            sig,
            title="[bold cyan]Analysis of GPU Kernel[/bold cyan]",
            subtitle=f"[dim]{file_path}:{line_no}[/dim]",
            subtitle_align="right",
            border_style="blue",
        )

    @staticmethod
    def build_summary_table(console, data: dict):
        table = Table(
            title=f"[bold magenta]Mneme Summary[/bold magenta] "
            f"([bold yellow]Baseline Execution Time (ns): {data['Baseline']}[/bold yellow])",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
            pad_edge=False,
        )
        table.add_column("Config", justify="left", no_wrap=True, overflow="fold")
        table.add_column("Speedup (×)", justify="right", overflow="fold")
        table.add_section()
        for k, v in data.items():
            if k != "Baseline":
                table.add_row(k, f"{v:.3f}")
        return table

    @staticmethod
    def render_report(console, signature, file_path, line_no, data):
        sig_panel = Summary.build_signature_panel(
            console, signature, file_path, line_no
        )
        tbl = Summary.build_summary_table(console, data)

        term_w = console.size.width
        # Measure both, then choose a shared width (don’t exceed terminal)
        w_sig = Summary.measure_width(console, sig_panel, max_width=term_w)
        w_tbl = Summary.measure_width(console, tbl, max_width=term_w)
        width = min(term_w, max(w_sig, w_tbl))  # big enough for both, but <= terminal

        # Re-render with fixed width so they match
        sig_panel.width = width
        tbl.width = width
        console.print(sig_panel)
        console.print(tbl)

    @staticmethod
    def analyze(args):
        kernel_descr = RecordedExecution.from_json(args.db)
        OrigMod = jit.link_llvm_modules(
            kernel_descr.llvm_files, kernel_descr.kernel_name, False, False
        )
        mod = None
        src = None
        root = None
        loc = -1
        for ll in kernel_descr.llvm_files:
            with open(ll, "rb") as fd:
                ir = fd.read()
                mod = module.parse_bitcode(ir)
                try:
                    Func = mod.get_function(kernel_descr.kernel_name)
                except NameError:
                    logger.debug(
                        f"Could not find function in {kernel_descr.kernel_name} in file {ll}"
                    )
                    continue

                root, src, loc = Func.get_function_location()
                break

        if not MnemeDB.verify_db(args.results):
            print("Missing DB fields, exiting")

        fsrc = "[Unknown]-no-dbg-info."
        if src is not None:
            fsrc = (Path(root) / Path(src)).resolve()

        df = pd.read_csv(str(args.results))
        console = Console()
        verified, data = Summary.compute_speedups(df)
        if args.json:
            baseline = data.pop("Baseline")
            json_data = {}
            json_data["speedup"] = data
            json_data["Root directory"] = root
            json_data["filename"] = src
            json_data["Baseline Time (ns)"] = baseline
            json_data["Kernel Name"] = kernel_descr.demangled_name
            json_data["line"] = loc
            print(json.dumps(json_data, indent=2))
            return
        Summary.render_report(
            console, kernel_descr.demangled_name, str(fsrc), loc, data
        )


class Serve:
    theme = Theme(
        {
            "ok": "bold green",
            "warn": "bold yellow",
            "bad": "bold red",
            "title": "bold cyan",
            "dim": "grey62",
            "hint": "italic dim",
        }
    )

    @staticmethod
    def _make_path_link(path: Path, line: Optional[int], style: str = "bold") -> Text:
        """
        Rich supports true hyperlinks. Different editors understand different URL schemes.
        We'll include several in priority order; terminals/editors that support them will
        make them clickable. Others will still see a readable path:line.
        """
        text = Text(f"{path}:{line if line else ''}".rstrip(":"), style=style)
        # VS Code deep link
        try:
            vsc_url = f"vscode://file/{path}:{line or 1}"
            text.stylize(f"link {vsc_url}", 0, len(text))
        except Exception:
            pass
        # Fallback file:// (many terminals make file:// clickable)
        try:
            file_url = f"file://{path}"
            text.append(" ", style="")
            t2 = Text("(open file)", style=f"link {file_url} dim")
            text += t2
        except Exception:
            pass
        return text

    @staticmethod
    def _render_json_tree(data: Any, label: str = "result") -> Tree:
        tree = Tree(Text(label, style="title"))

        def add_branch(node: Any, parent: Tree, key: Optional[str] = None):
            if isinstance(node, dict):
                branch = parent.add(f"[bold]{key}[/]" if key else "[bold]{label}[/]")
                for k, v in node.items():
                    add_branch(v, branch, k)
            elif isinstance(node, list):
                branch = parent.add(
                    f"[bold]{key}[/] [dim](list, {len(node)} items)[/]"
                    if key
                    else "[bold]list[/]"
                )
                for i, v in enumerate(node):
                    add_branch(v, branch, f"[{i}]")
            else:
                parent.add(f"[dim]{key}[/]: {node!r}" if key else f"{node!r}")

        add_branch(data, tree)
        return tree

    @staticmethod
    def _attribute_line(params: Iterable[int]) -> str:
        # e.g. __attribute__((annotate("jit", 16,17,...)))
        parts = ",".join(str(p) for p in params)
        return f'__attribute__((annotate("jit", {parts})))'

    @staticmethod
    def _cmake_snippet():
        return "find_package(proteus CONFIG REQUIRED)\nadd_proteus(<target>)\n"

    @staticmethod
    def _makefile_snippet(install_path):
        return (
            f"CXXFLAGS += -I{install_path}/include \\\n"
            f"    -fpass-plugin={install_path}/lib64/libProteusPass.so\n\n"
            f"LDFLAGS += -L {install_path}/lib64 \\\n"
            f"    -Wl,-rpath,{install_path}/lib64 \\\n"
            f"    -lproteus $(llvm-config --libs) -lclang-cpp\n"
        )

    @staticmethod
    def render_proteus_build_integration(
        console: Console, install_path="<install_path>"
    ):
        cmake_code = Serve._cmake_snippet()
        make_code = Serve._makefile_snippet(install_path)

        cmake_view = Syntax(cmake_code, "cmake", word_wrap=False, line_numbers=False)
        make_view = Syntax(make_code, "make", word_wrap=False, line_numbers=False)

        # Minimal, copy-first layout
        console.print(
            Text("Install Proteus and point your build system at it.\n"),
            style="bold cyan",
        )
        console.print(
            Text(
                "• CMake: ensure proteus is on CMAKE_PREFIX_PATH (or pass -Dproteus_DIR=…)\n"
                "• Make : extend CXXFLAGS/LDFLAGS as shown.",
            )
        )
        console.print()
        console.print(Text("CMake — paste into CMakeLists.txt", style="bold cyan"))
        console.print(cmake_view)
        console.print()
        console.print(Text("Make — paste into your Makefile", style="bold cyan"))
        console.print(make_view)
        return

    @staticmethod
    def render_output(
        mneme_config,
        exec_time,
        console,
        func,
        file_path,
        line,
        params,
        show_raw: bool = False,
    ) -> None:
        """
        Render the 'mneme serve' optimal result payload with a clean, guided UI.
        Expected shape (extend as needed):
        {
            "function": "my_kernel",
            "file": "/path/to/source.cu",
            "line": 123,
            "annotation_params": [16,17,18, ...],
            "speedup": 1.42,                # optional
            "baseline_time_ms": 10.2,       # optional
            "optimized_time_ms": 7.2,       # optional
            "notes": "tuned on A100",       # optional
            "extra": {...}                  # optional nested data
        }
        """

        console.print(Text("Analyzed Function", style="bold cyan"))
        console.print(Syntax(func, "c", word_wrap=True, line_numbers=False))
        console.print(Text("Defined in file: ", style="bold cyan"))
        console.print(Serve._make_path_link(file_path, line, style=""))

        console.print()
        Serve.render_proteus_build_integration(console)

        # What to change in the source
        console.print()
        attr_line = Serve._attribute_line(params or [])
        console.print(
            Text(
                "To specialize the kernel, add this attribute to the function definition:",
                style="bold cyan",
            ),
        )

        console.print(Syntax(attr_line, "c", word_wrap=True, line_numbers=False))

        console.print()
        console.print(
            Text(
                "After building the application expose the mneme identified optimal parameters to proteus by",
                style="bold cyan",
            )
        )
        console.print(Syntax(f"export PROTEUS_TUNED_KERNELS={mneme_config}", "bash"))

        console.print(
            Text(f"\nExpected execution time is {exec_time} ns", style="bold red")
        )

    @staticmethod
    def set_cli_args(parser):
        parser.add_argument(
            "-db",
            "--record-database",
            dest="db",
            required=True,
            help="Path to Mneme JSON/db file",
        )

        parser.add_argument(
            "--results",
            dest="results",
            required=True,
            help="CSV database containing the performance data of the tuning runs",
        )

        parser.add_argument(
            "json",
            help="json file to store proteus configuration, if the file exists we append the configuration",
        )

        parser.set_defaults(func=Serve.serve)

    @staticmethod
    def serve(args):
        console = Console(theme=Serve.theme, soft_wrap=False)
        kernel_descr = RecordedExecution.from_json(args.db)
        jsFn = args.json
        data = {}
        if Path(jsFn).exists():
            with open(jsFn, "r") as fd:
                data = json.load(fd)

        df = pd.read_csv(str(args.results))

        # Filter verified=True and failed=False
        filtered = df[(df["verified"]) & (~df["failed"])]
        best_row = filtered.loc[filtered["exec_time_median"].idxmin()]
        best_row = best_row.to_dict()
        res = {}
        res["Pipeline"] = best_row["passes"]
        res["CodeGen"] = best_row["codegen_method"]
        res["SpecializeArgs"] = best_row["specialize"]
        res["SpecializeDims"] = best_row["specialize_dims"]
        res["SpecializeDimsAssume"] = True
        res["LaunchBounds"] = best_row["max_threads"] != 0
        res["OptLevel"] = str(3)
        res["CodeGenOptLevel"] = best_row["codegen_opt"]
        data[kernel_descr.kernel_name] = res

        fsrc = "[Unknown]-no-dbg-info."
        if src is not None and root is not None:
            fsrc = (Path(root) / Path(src)).resolve()

        with open(jsFn, "w") as fd:
            json.dump(data, fd, indent=2)

        Serve.render_output(
            str(Path(jsFn).resolve()),
            best_row["exec_time_median"],
            console,
            kernel_descr.demangled_name,
            fsrc,
            loc,
            [i + 1 for i, v in enumerate(kernel_descr.specializations) if v],
            False,
        )
