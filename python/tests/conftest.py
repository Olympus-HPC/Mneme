import pathlib
import subprocess

import pytest


# This is correct, as any tests that uses "subprocess" to test the cli will not be affected by "autoreuse"
@pytest.fixture(autouse=True)
def disable_all_dispose(monkeypatch):
    monkeypatch.setattr("mneme.llvm.buffer.MemBufferRef._dispose", lambda self: None)


import pathlib
import subprocess

import pytest


@pytest.fixture(params=["On", "Off"])
def build_test_program(tmp_path, request):
    """
    Build vecAdd with RDC On/Off depending on the parameter.
    """

    rdc_value = request.param

    src_dir = pathlib.Path(__file__).parent / "c_src" / "cmake"
    build_dir = tmp_path / f"build_rdc_{rdc_value.lower()}"
    build_dir.mkdir()

    # Query Mneme config
    mneme_cc = subprocess.check_output(["mneme", "config", "cc"], text=True).strip()
    mneme_cxx = subprocess.check_output(["mneme", "config", "cxx"], text=True).strip()
    mneme_cmake_dir = subprocess.check_output(
        ["mneme", "config", "cmakedir"], text=True
    ).strip()

    # CMake configure
    subprocess.run(
        [
            "cmake",
            f"-DCMAKE_C_COMPILER={mneme_cc}",
            f"-DCMAKE_CXX_COMPILER={mneme_cxx}",
            f"-DCMAKE_PREFIX_PATH={mneme_cmake_dir}",
            f"-DRDC={rdc_value}",
            str(src_dir),
        ],
        cwd=build_dir,
        check=True,
    )

    subprocess.run(["make", "-j"], cwd=build_dir, check=True)

    binary = build_dir / "vecAdd"
    assert binary.exists()

    return {
        "src_dir": src_dir,
        "build_dir": build_dir,
        "binary": binary,
        "cc": mneme_cc,
        "cxx": mneme_cxx,
        "cmakedir": mneme_cmake_dir,
        "rdc": rdc_value,
    }
