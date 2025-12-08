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

