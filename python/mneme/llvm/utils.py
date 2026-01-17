# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from ._lib_path_config import (
    MNEME_CONFIG_FILE,
    MNEME_CORE_LIB,
    MNEME_PROFILE_LIB,
    MNEME_RECORD_LIB,
)

# This module must be importable without loading the binding, to avoid
# bootstrapping issues in setup.py.


def get_mneme_core_library_name():
    """
    Return the name of the llvm4ml shared library file.
    """
    return MNEME_CORE_LIB


def get_profile_library():
    """
    Return the name of the mneme_profile eshared library file.
    """
    return MNEME_PROFILE_LIB


def get_mneme_record_library_name():
    """
    Return the name of the llvm4ml shared library file.
    """
    return MNEME_RECORD_LIB


def get_config_file():
    return MNEME_CONFIG_FILE
