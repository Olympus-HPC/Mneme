from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

from .parameters import BaseParam


class SearchSpace(ABC):
    """
    Declarative representation of a search space.

    A SearchSpace describes:
      • The **primary tunable dimensions** (dimensions())
      • Any **derived parameters** computed from these (derived())
      • Any **constraints** on valid configurations (constraints())
    """

    @abstractmethod
    def dimensions(self) -> Dict[str, BaseParam]:
        """
        Return the top-level parameters of this search space.
        Keys are parameter names, values are BaseParam instances.
        """
        pass

    def derived(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute derived parameters from the given primary parameters.

        Example:
          - compute launch_bounds if block_dim/max_threads implies it
          - compute grid_dim automatically from problem size
        """
        return {}

    def constraints(self, params: Dict[str, Any]) -> bool:
        """
        Validate that this parameter assignment is legal.

        Returns:
            True if the assignment is valid, False otherwise.
        """
        return True

    def sample_random(self) -> Dict[str, Any]:
        """
        Generate one valid random configuration from this search space.
        """
        from .random_sampler import sample_random_param

        MAX_RETRIES = 1000
        dims = self.dimensions()

        for _ in range(MAX_RETRIES):
            result = {}

            # Step 1: sample primary dimensions
            for name, param in dims.items():
                result[name] = sample_random_param(param)

            # Step 2: derived parameters
            derived = self.derived(result)
            if derived:
                result.update(derived)

            # Step 3: constraints
            if self.constraints(result):
                return result

        raise RuntimeError(
            f"Failed to produce a valid random sample after {MAX_RETRIES} attempts."
        )

    def random_samples(self):
        """
        Infinite generator yielding valid random configurations.

        Usage:
            for config in space.random_samples():
                process(config)
                if done:
                    break
        """
        while True:
            yield self.sample_random()

