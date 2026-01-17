# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
from typing import Union

_LEVELS = {
    "critical": logging.CRITICAL,
    "warn": logging.WARNING,  # accept 'warn' per your spec
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


# Library logger: silent by default
logger = logging.getLogger("mneme.replay")
logger.addHandler(logging.NullHandler())  # avoids "No handler" warnings
logger.propagate = False  # don't bubble up to root logger


def configure_replay_logging(level_name: Union[str, None]) -> None:
    """Attach a real handler only when -v is provided."""
    if not level_name:
        # Stay silent: leave only the NullHandler in place
        return

    level = _LEVELS[level_name.lower()]
    # Avoid double-adding if called twice
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        h = logging.StreamHandler()  # stderr by default
        # Short, friendly format; tweak as you like
        fmt = logging.Formatter("[mneme:%(levelname).1s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)

    logger.setLevel(level)
