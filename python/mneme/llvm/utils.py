import os
import sys


# This module must be importable without loading the binding, to avoid
# bootstrapping issues in setup.py.


def get_library_name():
    """
    Return the name of the llvm4ml shared library file.
    """
    current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.name == "posix":
        if sys.platform == "darwin":
            return os.path.abspath(current_file_path + "/libmneme.dylib")
        else:
            return os.path.abspath(current_file_path + "/libmneme.so")
    else:
        return os.path.abspath(current_file_path + "/libmneme.dll")
