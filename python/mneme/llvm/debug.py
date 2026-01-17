# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from mneme.llvm.common import _encode_string

from . import ffi, module


def get_source_and_line(mod: module.ModuleRef, fn_name: str):
    Func = mod.get_function(fn_name)
    return mod.source_file, Func.get_function_location()
