# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .llvm import initfini

initfini.initialize()

__version__ = "0.1.0"
