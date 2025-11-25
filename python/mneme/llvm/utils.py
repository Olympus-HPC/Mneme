import os
import sys

# This module must be importable without loading the binding, to avoid
# bootstrapping issues in setup.py.


def get_mneme_core_library_name():
    """
    Return the name of the llvm4ml shared library file.
    """
    current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.name == "posix":
        if sys.platform == "darwin":
            return os.path.abspath(current_file_path + "/native/lib64/libmneme.dylib")
        else:
            return os.path.abspath(current_file_path + "/native/lib64/libmneme.so")
    else:
        return os.path.abspath(current_file_path + "/native/lib64/libmneme.dll")


def get_profile_library():
    """
    Return the name of the mneme_profile eshared library file.
    """
    current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.name == "posix":
        if sys.platform == "darwin":
            return os.path.abspath(current_file_path + "/native/lib64/libmneme_profile.dylib")
        else:
            return os.path.abspath(current_file_path + "/native/lib64/libmneme_profile.so")
    else:
        return os.path.abspath(current_file_path + "/native/lib64/libmneme_profile.dll")

def get_mneme_record_library_name():
    """
    Return the name of the llvm4ml shared library file.
    """
    current_file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.name == "posix":
        if sys.platform == "darwin":
            return os.path.abspath(current_file_path + "/native/lib64/librecord.dylib")
        else:
            return os.path.abspath(current_file_path + "/native/lib64/librecord.so")
    else:
        return os.path.abspath(current_file_path + "/native/lib64/librecord.dll")

