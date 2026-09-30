"""GPU-independent validation and normalization of supplied launch layouts."""

import math


MAX_THREADS_PER_BLOCK = 1024
MAX_BLOCK_DIMS = (1024, 1024, 64)
MAX_GRID_DIMS = ((1 << 31) - 1, 65535, 65535)


def normalize_launch_candidates(candidates: list[dict], *, recorded_kernel=None) -> list[dict]:
    """Return a JSON-serializable, deduplicated list in first-occurrence order.

    Each entry must contain ``block`` and optionally ``grid``, each a list or
    tuple of three positive integers within Mneme's launch limits. Booleans,
    coercible strings/floats, and unknown keys are rejected with ``ValueError``.
    With a recorded kernel, omitted grids are resolved by ceiling-dividing the
    recorded work (block * grid) by the supplied block on each axis. Without a
    recording, omitted grids stay omitted; duplicates are removed again after
    resolution. No device queries or replay are needed.
    """
    if not isinstance(candidates, list):
        raise ValueError("launch_candidates must be a list of dictionaries")

    normalized = []
    seen = set()
    for index, candidate in enumerate(candidates):
        prefix = f"launch_candidates[{index}]"

        if not isinstance(candidate, dict) or "block" not in candidate:
            raise ValueError(f"{prefix} requires a block")
        if candidate.keys() - {"block", "grid"}:
            raise ValueError(f"{prefix} only accepts block and grid")

        layout = {}
        for name, limits in (("block", MAX_BLOCK_DIMS), ("grid", MAX_GRID_DIMS)):
            if name not in candidate:
                continue

            values = candidate[name]
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 3
                or any(type(v) is not int for v in values)
            ):
                raise ValueError(f"{prefix}.{name} must contain three integers")

            if any(v <= 0 or v > limit for v, limit in zip(values, limits)):
                raise ValueError(f"{prefix}.{name} exceeds launch dimension limits")

            layout[name] = list(values)

        if math.prod(layout["block"]) > MAX_THREADS_PER_BLOCK:
            raise ValueError(f"{prefix}.block exceeds the threads-per-block limit")

        if "grid" not in layout and recorded_kernel is not None:
            layout["grid"] = [
                (int(getattr(recorded_kernel.block_dim, axis))
                 * int(getattr(recorded_kernel.grid_dim, axis)) + block - 1) // block
                for axis, block in zip(("x", "y", "z"), layout["block"])
            ]

            if any(v <= 0 or v > limit for v, limit in zip(layout["grid"], MAX_GRID_DIMS)):
                raise ValueError(f"{prefix}.grid exceeds launch dimension limits after resolution")

        key = (tuple(layout["block"]), tuple(layout.get("grid", [])))
        if key not in seen:
            seen.add(key)
            normalized.append(layout)
            
    return normalized


def launch_layout_key(block, grid) -> str:
    """Stable categorical identity for the complete launch layout."""
    return "x".join(str(v) for v in block) + "/" + "x".join(str(v) for v in grid)
