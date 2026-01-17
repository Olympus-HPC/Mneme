# Copyright (c) 2021-2024. LLNL-CODE-2000766 and Mneme Contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._version import get_versions
from .initfini import *
from .module import *
from .value import *
from .typeref import *
from .context import *


__version__ = get_versions()["version"]
del get_versions
