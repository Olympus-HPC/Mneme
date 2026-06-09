import importlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mneme.mneme_types import ExperimentConfiguration, dim3
from mneme.recorded_execution import RecordedExecution
from mneme.tuning.search_space import (
    BaseParam,
    BoolParam,
    CategoricalParam,
    FixedParam,
    IntRangeParam,
    RealRangeParam,
    SearchSpace,
)


DEFAULT_PIPELINES = [
    "default<O3>",
    "default<O2>",
    "default<O1>",
    "default<Os>",
    "default<Oz>",
]

# for 'quick' default only search these pipelines
QUICK_PIPELINES = ["default<O3>", "default<O2>"]

MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCK_DIMS = (1024, 1024, 64)
MAX_GRID_DIMS = ((1 << 31) - 1, 65535, 65535)


def parse_range(spec: str) -> Tuple[int, int, int]:
    parts = [p for p in str(spec).split(":") if p != ""]
    if len(parts) not in (2, 3):
        raise ValueError(f"Expected range LOW:HIGH[:STEP], got {spec!r}")
    
    low = int(parts[0])
    high = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1

    if step <= 0:
        raise ValueError(f"Range step must be positive, got {spec!r}")
    if low > high:
        raise ValueError(f"Range low must be <= high, got {spec!r}")
    
    return low, high, step


def parse_key_value_args(values: Optional[Sequence[str]]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE for --space-arg, got {value!r}")
        
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError(f"Empty --space-arg key in {value!r}")
        
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    
    return parsed


def dim3_to_shape(value: dim3) -> str:
    return f"{int(value.x)}x{int(value.y)}x{int(value.z)}"


def shape_to_dim3(value: str) -> dim3:
    try:
        x, y, z = (int(v) for v in value.split("x"))
    except ValueError as exc:
        raise ValueError(f"Invalid block shape {value!r}; expected XxYxZ") from exc
    return dim3(x, y, z)


def _axis_values(recorded: int, total_work: int, axis_limit: int) -> List[int]:
    values = {max(1, int(recorded))}

    power = 1
    while power <= axis_limit and power <= max(1, total_work):
        values.add(power)
        power *= 2

    for scale in (0.5, 0.75, 1.25, 1.5, 2.0):
        scaled = max(1, int(round(recorded * scale)))
        if scaled <= axis_limit:
            values.add(scaled)

    # Include practical divisors without making very large spaces explode.
    limit = int(math.sqrt(max(1, total_work)))
    for divisor in range(1, min(limit, 2048) + 1):
        if total_work % divisor != 0:
            continue
        other = total_work // divisor
        if divisor <= axis_limit:
            values.add(divisor)
        if other <= axis_limit:
            values.add(other)

    return sorted(v for v in values if v > 0 and v <= axis_limit)


def _active_axes(kernel: RecordedExecution.KernelInstance, launch_dim: str) -> Tuple[bool, bool, bool]:
    if launch_dim == "none":
        return (False, False, False)
    if launch_dim == "x":
        return (True, False, False)
    if launch_dim == "xy":
        return (True, True, False)
    if launch_dim == "xyz":
        return (True, True, True)
    if launch_dim != "auto":
        raise ValueError(f"Unknown launch_dim {launch_dim!r}")
    
    return tuple(
        bool(getattr(kernel.block_dim, axis) > 1 or getattr(kernel.grid_dim, axis) > 1)
        for axis in ("x", "y", "z")
    )


def _launch_budget(recorded_threads: int, safety: str) -> int:
    if safety == "conservative":
        return min(MAX_THREADS_PER_BLOCK, max(1, recorded_threads))
    if safety == "balanced":
        return min(MAX_THREADS_PER_BLOCK, max(1, recorded_threads * 2))
    if safety == "aggressive":
        return MAX_THREADS_PER_BLOCK
    
    raise ValueError(f"Unknown launch_safety {safety!r}")


def _candidate_block_shapes(
    kernel: RecordedExecution.KernelInstance,
    launch_dim: str,
    launch_safety: str,
) -> List[str]:
    """ Generate a candidate list of block shapes to explore.
        This can be very helpful since for some kernels most random launch configs will be
        incorrect. We want to intelligently select launch dims.
    """

    recorded = (
        int(kernel.block_dim.x),
        int(kernel.block_dim.y),
        int(kernel.block_dim.z),
    )
    grid = (
        int(kernel.grid_dim.x),
        int(kernel.grid_dim.y),
        int(kernel.grid_dim.z),
    )
    total_work = tuple(max(1, recorded[i] * grid[i]) for i in range(3))
    active = _active_axes(kernel, launch_dim)
    budget = _launch_budget(recorded[0] * recorded[1] * recorded[2], launch_safety)

    per_axis: List[List[int]] = []
    for i in range(3):
        if active[i]:
            per_axis.append(_axis_values(recorded[i], total_work[i], MAX_BLOCK_DIMS[i]))
        else:
            per_axis.append([recorded[i]])

    shapes = set()
    baseline = dim3_to_shape(kernel.block_dim)
    shapes.add(baseline)
    for x, y, z in itertools.product(*per_axis):
        threads = x * y * z
        if threads <= 0 or threads > MAX_THREADS_PER_BLOCK:
            continue
        if threads > budget and (x, y, z) != recorded:
            continue
        derived_grid = (
            int(math.ceil(total_work[0] / x)),
            int(math.ceil(total_work[1] / y)),
            int(math.ceil(total_work[2] / z)),
        )
        if any(derived_grid[i] <= 0 or derived_grid[i] > MAX_GRID_DIMS[i] for i in range(3)):
            continue
        shapes.add(f"{x}x{y}x{z}")

    return sorted(shapes, key=lambda s: tuple(int(v) for v in s.split("x")))


def _pipeline_choices(
    preset: str,
    passes: Optional[Sequence[str]],
    fixed_passes: Optional[str],
    pipeline_file: Optional[str],
) -> Tuple[Optional[List[str]], str]:
    if fixed_passes:
        return None, fixed_passes

    choices: List[str] = []
    if passes:
        choices.extend(str(p) for p in passes)
    if pipeline_file:
        with open(pipeline_file, "r") as fd:
            choices.extend(line.strip() for line in fd if line.strip() and not line.lstrip().startswith("#"))

    if choices:
        return list(dict.fromkeys(choices)), choices[0]

    if preset == "quick":
        return QUICK_PIPELINES, QUICK_PIPELINES[0]
    if preset == "launch":
        return None, "default<O3>"
    
    return DEFAULT_PIPELINES, DEFAULT_PIPELINES[0]


def _space_param(name: str, mode: str, fixed_default: bool = False) -> BaseParam:
    if mode == "fixed":
        return FixedParam(name, fixed_default)
    if mode == "on":
        return FixedParam(name, True)
    if mode == "off":
        return FixedParam(name, False)
    if mode == "on-off":
        return BoolParam(name)
    
    raise ValueError(f"Unknown {name} mode {mode!r}")


def param_to_description(param: BaseParam) -> Dict[str, Any]:
    if isinstance(param, FixedParam):
        return {"type": "fixed", "value": param.value}
    if isinstance(param, BoolParam):
        return {"type": "bool", "choices": list(param.choices)}
    if isinstance(param, CategoricalParam):
        return {"type": "categorical", "choices": list(param.choices)}
    if isinstance(param, IntRangeParam):
        return {
            "type": "int_range",
            "low": param.low,
            "high": param.high,
            "step": param.step,
        }
    if isinstance(param, RealRangeParam):
        return {"type": "real_range", "low": param.low, "high": param.high}
    return {"type": type(param).__name__}


def describe_search_space(space: Any, *, preset: Optional[str] = None, custom: bool = False) -> Dict[str, Any]:
    dimensions = {
        name: param_to_description(param)
        for name, param in space.dimensions().items()
    }
    data = {
        "type": "custom" if custom else "builtin",
        "dimensions": dimensions,
    }
    if preset:
        data["preset"] = preset
    if hasattr(space, "metadata"):
        data["metadata"] = getattr(space, "metadata")
    return data


class BuiltinTuneSearchSpace(SearchSpace):
    """ Derived from Mneme search space"""

    def __init__(
        self,
        recorded_kernel: RecordedExecution.KernelInstance,
        *,
        preset: str = "standard",
        launch_dim: str = "auto",
        launch_safety: Optional[str] = None,
        passes: Optional[Sequence[str]] = None,
        fixed_passes: Optional[str] = None,
        pipeline_file: Optional[str] = None,
        codegen_opt_range: Optional[str] = None,
        fixed_codegen_opt: Optional[int] = None,
        specialize_space: Optional[str] = None,
        specialize_dims_space: Optional[str] = None,
        launch_bounds_space: Optional[str] = None,
        min_blocks_per_sm_range: Optional[str] = None,
        fixed_min_blocks_per_sm: Optional[int] = None,
        max_threads_range: Optional[str] = None,
        max_threads_policy: str = "block-threads",
        codegen_method: Optional[str] = None,
    ):
        if preset not in {"quick", "standard", "launch", "compiler", "full"}:
            raise ValueError(f"Unknown tuning preset {preset!r}")

        self.recorded_kernel = recorded_kernel
        self.preset = preset
        self.codegen_method = codegen_method
        self.launch_dim = "none" if preset == "compiler" else launch_dim
        if launch_safety is None:
            launch_safety = "conservative" if preset == "quick" else "aggressive" if preset == "full" else "balanced"
        self.launch_safety = launch_safety

        default_codegen_range = "2:3" if preset == "quick" else "1:3"
        self._passes_choices, self._baseline_passes = _pipeline_choices(
            preset, passes, fixed_passes, pipeline_file
        )

        if fixed_codegen_opt is None and codegen_opt_range is None and preset == "launch":
            fixed_codegen_opt = 3
        if fixed_codegen_opt is None:
            low, high, step = parse_range(codegen_opt_range or default_codegen_range)
            self._codegen_param = IntRangeParam("codegen_opt", low, high, step)
            self._baseline_codegen_opt = high
        else:
            self._codegen_param = FixedParam("codegen_opt", int(fixed_codegen_opt))
            self._baseline_codegen_opt = int(fixed_codegen_opt)

        specialize_space = specialize_space or ("on-off" if preset == "full" else "fixed")
        specialize_dims_space = specialize_dims_space or ("on-off" if preset == "full" else "fixed")
        launch_bounds_space = launch_bounds_space or ("on-off" if preset == "full" else "fixed")

        self._dimensions: Dict[str, BaseParam] = {}
        if preset in {"quick", "standard", "launch", "full"} and self.launch_dim != "none":
            self._dimensions["block_shape"] = CategoricalParam(
                "block_shape",
                _candidate_block_shapes(recorded_kernel, self.launch_dim, self.launch_safety),
            )
        else:
            self._dimensions["block_shape"] = FixedParam("block_shape", dim3_to_shape(recorded_kernel.block_dim))

        if self._passes_choices is None:
            self._dimensions["passes"] = FixedParam("passes", self._baseline_passes)
        else:
            self._dimensions["passes"] = CategoricalParam("passes", self._passes_choices)

        self._dimensions["codegen_opt"] = self._codegen_param
        self._dimensions["specialize"] = _space_param("specialize", specialize_space, False)
        self._dimensions["specialize_dims"] = _space_param("specialize_dims", specialize_dims_space, False)
        self._dimensions["set_launch_bounds"] = _space_param("set_launch_bounds", launch_bounds_space, False)

        if fixed_min_blocks_per_sm is not None:
            self._dimensions["min_blocks_per_sm"] = FixedParam("min_blocks_per_sm", int(fixed_min_blocks_per_sm))
        elif min_blocks_per_sm_range is not None or preset == "full":
            low, high, step = parse_range(min_blocks_per_sm_range or "1:8")
            self._dimensions["min_blocks_per_sm"] = IntRangeParam("min_blocks_per_sm", low, high, step)
        else:
            self._dimensions["min_blocks_per_sm"] = FixedParam("min_blocks_per_sm", 0)

        self.max_threads_policy = max_threads_policy
        if max_threads_range:
            low, high, step = parse_range(max_threads_range)
            self._dimensions["max_threads"] = IntRangeParam("max_threads", low, high, step)
        elif max_threads_policy == "powers-of-two":
            self._dimensions["max_threads"] = CategoricalParam(
                "max_threads", [64, 128, 256, 512, 1024]
            )
        elif max_threads_policy in {"recorded", "block-threads", "hardware"}:
            self._dimensions["max_threads"] = FixedParam("max_threads", 0)
        else:
            raise ValueError(f"Unknown max_threads_policy {max_threads_policy!r}")

        self.metadata = {
            "launch_dim": self.launch_dim,
            "launch_safety": self.launch_safety,
            "baseline_passes": self._baseline_passes,
            "baseline_codegen_opt": self._baseline_codegen_opt,
            "codegen_method": self.codegen_method,
        }

    def dimensions(self) -> Dict[str, BaseParam]:
        return dict(self._dimensions)

    def _derive_grid(self, block: dim3) -> dim3:
        active = _active_axes(self.recorded_kernel, self.launch_dim)
        recorded_block = self.recorded_kernel.block_dim
        recorded_grid = self.recorded_kernel.grid_dim

        values = []
        for axis, is_active in zip(("x", "y", "z"), active):
            if is_active:
                total = int(getattr(recorded_block, axis)) * int(getattr(recorded_grid, axis))
                values.append(int(math.ceil(max(1, total) / int(getattr(block, axis)))))
            else:
                values.append(int(getattr(recorded_grid, axis)))

        return dim3(*values)

    def _max_threads(self, params: Mapping[str, Any], block: dim3) -> int:
        block_threads = int(block.x) * int(block.y) * int(block.z)
        if not params.get("set_launch_bounds", False):
            return 0
        
        explicit = int(params.get("max_threads", 0) or 0)
        if explicit > 0:
            return explicit
        if self.max_threads_policy == "recorded":
            return int(self.recorded_kernel.block_dim.x) * int(self.recorded_kernel.block_dim.y) * int(self.recorded_kernel.block_dim.z)
        if self.max_threads_policy == "hardware":
            return MAX_THREADS_PER_BLOCK
        
        return block_threads

    def derived(self, params: Dict[str, Any]) -> ExperimentConfiguration:
        block = shape_to_dim3(params["block_shape"])
        set_launch_bounds = bool(params.get("set_launch_bounds", False))
        max_threads = self._max_threads(params, block)
        min_blocks_per_sm = int(params.get("min_blocks_per_sm", 0) or 0) if set_launch_bounds else 0

        config = ExperimentConfiguration(
            grid=self._derive_grid(block),
            block=block,
            shared_mem=int(self.recorded_kernel.shared_mem),
            specialize=bool(params.get("specialize", False)),
            set_launch_bounds=set_launch_bounds,
            max_threads=max_threads if set_launch_bounds else 0,
            min_blocks_per_sm=min_blocks_per_sm,
            specialize_dims=bool(params.get("specialize_dims", False)),
            passes=str(params.get("passes", self._baseline_passes)),
            codegen_opt=int(params.get("codegen_opt", self._baseline_codegen_opt)),
            prune=True,
            internalize=True,
        )
        return config

    def constraints(self, params: Any) -> bool:
        if isinstance(params, ExperimentConfiguration):
            config = params
        else:
            config = self.derived(dict(params))

        block = config.block
        grid = config.grid
        threads = int(block.x) * int(block.y) * int(block.z)
        if threads <= 0 or threads > MAX_THREADS_PER_BLOCK:
            return False
        
        for i, axis in enumerate(("x", "y", "z")):
            if int(getattr(block, axis)) <= 0 or int(getattr(block, axis)) > MAX_BLOCK_DIMS[i]:
                return False
            if int(getattr(grid, axis)) <= 0 or int(getattr(grid, axis)) > MAX_GRID_DIMS[i]:
                return False
            
        return config.is_valid()

    def baseline(self) -> ExperimentConfiguration:
        config = ExperimentConfiguration(
            grid=dim3(
                int(self.recorded_kernel.grid_dim.x),
                int(self.recorded_kernel.grid_dim.y),
                int(self.recorded_kernel.grid_dim.z),
            ),
            block=dim3(
                int(self.recorded_kernel.block_dim.x),
                int(self.recorded_kernel.block_dim.y),
                int(self.recorded_kernel.block_dim.z),
            ),
            shared_mem=int(self.recorded_kernel.shared_mem),
            specialize=False,
            set_launch_bounds=False,
            max_threads=0,
            min_blocks_per_sm=0,
            specialize_dims=False,
            passes=self._baseline_passes,
            codegen_opt=self._baseline_codegen_opt,
            prune=True,
            internalize=True,
        )
        return config


def load_custom_space(spec: str, recorded_kernel: RecordedExecution.KernelInstance, kwargs: Optional[Mapping[str, Any]] = None) -> Any:
    """ Allow users to dynamically load search space subclasses.
        This allos for more dynamic behavior and custom implementations from the user.
    """

    if ":" not in spec:
        raise ValueError("--space-module must have the form MODULE_OR_PATH:CLASS")
    
    module_name, class_name = spec.rsplit(":", 1)
    kwargs = dict(kwargs or {})

    if module_name.endswith(".py") or "/" in module_name:
        path = Path(module_name).resolve()
        name = f"mneme_user_space_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(name, str(path))
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not import custom search space from {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    cls = getattr(module, class_name)
    return cls(recorded_kernel, **kwargs)


def finite_param_values(param: BaseParam) -> List[Any]:
    if isinstance(param, FixedParam):
        return [param.value]
    if isinstance(param, BoolParam):
        return list(param.choices)
    if isinstance(param, CategoricalParam):
        return list(param.choices)
    if isinstance(param, IntRangeParam):
        return list(range(param.low, param.high + 1, param.step))
    raise ValueError(f"Parameter {param.name!r} is not finite/enumerable")


def iter_finite_params(space: Any) -> Iterable[Dict[str, Any]]:
    dims = space.dimensions()
    names = list(dims.keys())
    value_lists = [finite_param_values(dims[name]) for name in names]
    
    for combo in itertools.product(*value_lists):
        yield dict(zip(names, combo))
