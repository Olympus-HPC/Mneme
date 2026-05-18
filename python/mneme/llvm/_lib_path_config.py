from __future__ import annotations
from pathlib import Path
import sys
import json

_PKG_DIR = Path(__file__).resolve().parents[1]
_NATIVE = _PKG_DIR / "native"
_CONFIG_FILE = _NATIVE / "config.json"

# Try to read library directory from config.json otherwise read from pip installed location
_LIB64 = _NATIVE / "lib64"
if _CONFIG_FILE.exists():
    with open(_CONFIG_FILE) as _f:
        _cfg = json.load(_f)
        if "libdir" in _cfg:
            libdir = _cfg["libdir"]
            # Replace @PREFIX@ placeholder with actual package location
            if "@PREFIX@" in libdir:
                libdir = libdir.replace("@PREFIX@", str(_NATIVE.resolve()))
            _LIB64 = Path(libdir)

def _lib(name_linux: str, name_darwin: str) -> str:
    return str(_LIB64 / (name_darwin if sys.platform == "darwin" else name_linux))

MNEME_CORE_LIB    = _lib("libmneme.so",         "libmneme.dylib")
MNEME_PROFILE_LIB = _lib("libmneme_profile.so", "libmneme_profile.dylib")
MNEME_RECORD_LIB  = _lib("librecord.so",        "librecord.dylib")
MNEME_CONFIG_FILE = str(_CONFIG_FILE)
