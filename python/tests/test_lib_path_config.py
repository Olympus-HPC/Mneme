import importlib
import json
from pathlib import Path

import mneme.llvm._lib_path_config as lib_path_config


def test_sys_path_fallback_used_when_primary_config_missing(tmp_path, monkeypatch):
    """When native/config.json isn't next to the package (editable install),
    the module must fall back to searching sys.path for it."""

    fake_native = tmp_path / "site-packages" / "mneme" / "native"
    fake_native.mkdir(parents=True)
    fake_config = fake_native / "config.json"
    fake_config.write_text(json.dumps({"libdir": "@PREFIX@/lib64"}))

    monkeypatch.syspath_prepend(str(tmp_path / "site-packages"))

    real_exists = Path.exists
    primary_config = lib_path_config._NATIVE / "config.json"

    def fake_exists(self):
        if self == primary_config:
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    try:
        importlib.reload(lib_path_config)

        assert lib_path_config._CONFIG_FILE == fake_config
        assert lib_path_config._NATIVE == fake_native
        assert lib_path_config.MNEME_CONFIG_FILE == str(fake_config)
        assert str(fake_native.resolve()) in lib_path_config.MNEME_CORE_LIB
    finally:
        # Restore real sys.path/Path.exists before reloading back to the
        # module's real, un-faked state so later tests aren't affected.
        monkeypatch.undo()
        importlib.reload(lib_path_config)
