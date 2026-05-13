import argparse
import json
from dataclasses import fields
from typing import Any, Dict

from mneme.tuning.session import TuneOptions, TuningSession, merge_config_and_args


def _none_bool_action() -> Any:
    return argparse.BooleanOptionalAction


def add_tune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-rdb", "--record-database", dest="record_database", default=None)
    parser.add_argument("-rid", "--record-id", dest="record_id", default=None)
    parser.add_argument("--space-preset", choices=["quick", "standard", "launch", "compiler", "full"], default=None)
    parser.add_argument("--sampler", choices=["random", "tpe", "grid", "exhaustive"], default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--metric", choices=["mean", "median", "min"], default=None)
    parser.add_argument("--objective", choices=["time", "speedup"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--rerun-baseline", action="store_true", default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--keep-going", dest="fail_fast", action="store_false", default=None)
    parser.add_argument("--print-space", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--baseline-only", action="store_true", default=None)
    parser.add_argument("--quiet", action="store_true", default=None)

    parser.add_argument("--study-name", default=None)
    parser.add_argument("--optuna-storage", default=None)
    parser.add_argument("--pruner", choices=["none", "median"], default=None)

    parser.add_argument("--launch-dim", choices=["none", "auto", "x", "xy", "xyz"], default=None)
    parser.add_argument("--launch-safety", choices=["conservative", "balanced", "aggressive"], default=None)
    parser.add_argument(
        "--adaptive-invalid-ban",
        dest="adaptive_invalid_ban",
        action=_none_bool_action(),
        default=None,
    )

    parser.add_argument("--passes", nargs="+", default=None)
    parser.add_argument("--fixed-passes", default=None)
    parser.add_argument("--pipeline-file", default=None)
    parser.add_argument("--codegen-opt-range", default=None)
    parser.add_argument("--fixed-codegen-opt", type=int, default=None)
    parser.add_argument("--codegen-method", default=None)

    parser.add_argument("--specialize-space", choices=["fixed", "on", "off", "on-off"], default=None)
    parser.add_argument("--specialize-dims-space", choices=["fixed", "on", "off", "on-off"], default=None)
    parser.add_argument("--launch-bounds-space", choices=["fixed", "on", "off", "on-off"], default=None)
    parser.add_argument("--min-blocks-per-sm-range", default=None)
    parser.add_argument("--fixed-min-blocks-per-sm", type=int, default=None)
    parser.add_argument("--max-threads-range", default=None)
    parser.add_argument(
        "--max-threads-policy",
        choices=["recorded", "block-threads", "powers-of-two", "hardware"],
        default=None,
    )

    parser.add_argument("--space-module", default=None)
    parser.add_argument("--space-arg", action="append", default=None)

    parser.add_argument("--config", default=None)
    parser.add_argument("--dump-config", default=None)
    parser.add_argument("--proteus-output", default=None)
    parser.add_argument("--no-proteus-output", dest="proteus_enabled", action="store_false", default=None)
    parser.set_defaults(func=run_tune)


def _options_from_mapping(data: Dict[str, Any]) -> TuneOptions:
    """ Incoming options includes both CLI args and config file options
        Merge back into TuneOptions config obj.
    """
    valid = {f.name for f in fields(TuneOptions)}
    kwargs = {key: value for key, value in data.items() if key in valid}

    missing = [key for key in ("record_database", "record_id") if not kwargs.get(key)]
    if missing:
        raise ValueError(f"Missing required tune option(s): {', '.join(missing)}")
    
    if kwargs.get("space_arg") is None:
        kwargs["space_arg"] = []
    return TuneOptions(**kwargs)


def run_tune(args: argparse.Namespace, verbosity) -> int:
    try:
        # Create TuneOptions from cli and config file
        merged = merge_config_and_args(args)
        options = _options_from_mapping(merged)

        if options.dump_config:
            resolved = options.to_config_dict()
            with open(options.dump_config, "w") as fd:
                json.dump(resolved, fd, indent=2)
                fd.write("\n")
            return 0
        return TuningSession(options).run()
    except ValueError as exc:
        parser = getattr(args, "parser", None)
        if parser is not None:
            parser.error(str(exc))
        print(f"Invalid tuning configuration: {exc}")
        return 3
