import glob
import json
import os
import subprocess
import sys
from distutils.sysconfig import get_python_lib
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop


# Helper function to run shell commands
def run_command(command, cwd=None):
    sys.stderr.write(f"Running: {' '.join(command)} in {cwd or os.getcwd()}")
    result = subprocess.run(command, cwd=cwd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(command)} failed")


def has_nvidia_gpu():
    try:
        subprocess.check_output("nvidia-smi", shell=True, text=True)
        return True
    except subprocess.CalledProcessError:
        return False


def has_amd_gpu():
    try:
        output = subprocess.check_output("rocminfo", shell=True, text=True)
        return "AMD" in output or "gfx" in output
    except subprocess.CalledProcessError:
        return False


def get_llvm_paths(llvm_dir):
    llvm_config = Path(llvm_dir) / "bin" / "llvm-config"
    if not llvm_config.exists():
        raise RuntimeError(f"llvm-config not found at {llvm_config}")

    def run(*args):
        return subprocess.check_output([str(llvm_config), *args], text=True).strip()

    return {
        "include": run("--includedir"),
        "libdir": run("--libdir"),
        "libs": run("--libs"),
        "ldflags": run("--ldflags"),
        "system_libs": run("--system-libs"),
    }


# Custom build class for CMake
class CMakeBuild(build_ext):
    PROTEUS_REPO = "https://github.com/Olympus-HPC/proteus.git"
    SPDLOG_REPO = "https://github.com/gabime/spdlog.git"

    def initialize_options(self):
        super().initialize_options()
        self.root_dir = os.path.abspath(os.path.dirname(__file__))
        build_cmd = self.get_finalized_command("build")
        self.build_lib = build_cmd.build_lib
        site_packages = Path(get_python_lib())
        self.install_dir = site_packages / "mneme/native/"
        self.has_nvidia = "On" if has_nvidia_gpu() else "Off"
        self.has_amd = "On" if has_amd_gpu() else "Off"
        self.llvm_dir = os.getenv("LLVM_INSTALL_DIR", None)
        if self.has_amd == "On":
            self.cxx = f"{self.llvm_dir}/bin/amdclang++"
            self.cc = f"{self.llvm_dir}/bin/amdclang"
            self.llvm_dir = f"{self.llvm_dir}/llvm/"
        else:
            self.cxx = f"{self.llvm_dir}/bin/clang++"
            self.cc = f"{self.llvm_dir}/bin/clang"
            self.llvm_dir = f"{self.llvm_dir}/"

        if not self.llvm_dir:
            raise RuntimeError(
                "Error: LLVM_INSTALL_DIR is not set. Please export it before running setup.py."
            )
        prefix = Path(self.install_dir).resolve()
        libdir = prefix / "lib64"
        includedir = prefix / "include"
        cmake_dir = libdir / "cmake"
        llvm_config = get_llvm_paths(self.llvm_dir)

        cfg = {
            "cc": self.cc,
            "cxx": self.cxx,
            "prefix": str(prefix),
            "libdir": str(libdir),
            "includedir": str(includedir),
            "cmakedir": str(cmake_dir),
            "cflags": f"-fpass-plugin={prefix}/lib64/libProteusPass.so -fplugin={prefix}/lib64/libProteusPass.so -fno-discard-value-names -ftrivial-auto-var-init=zero -Xclang -mllvm -Xclang -force-proteus-jit-annotate-all",
            "ldflags": f"-L{self.llvm_dir}/lib -L{self.llvm_dir}/llvm/lib {llvm_config['libs']} {llvm_config['system_libs']} -L{libdir}/ -Wl,-rpath,{libdir}/ -llldCommon -llldELF -lproteus",
        }
        with open(prefix / "config.json", "w") as fd:
            json.dump(cfg, fd, indent=2)

    def run(self):
        if not os.path.exists("third_party"):
            os.mkdir("third_party")
        if "PROTEUS_DIR" in os.environ:
            proteus_dir = os.environ["PROTEUS_DIR"]
        else:
            proteus_dir = self.clone_and_build_proteus()

        spdlog_dir = self.clone_and_build_spdlog()
        self.build_mneme(proteus_dir, spdlog_dir)

    def clone_and_build_proteus(self):
        if "PROTEUS_SRC" in os.environ:
            proteus_path = os.environ["PROTEUS_SRC"]
        else:
            proteus_path = os.path.abspath("third_party/proteus")
            if not os.path.exists(proteus_path):
                run_command(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        # "--branch",
                        # "features/blocks-per-eu",
                        self.PROTEUS_REPO,
                        proteus_path,
                    ],
                    cwd="third_party",
                )

        build_dir = os.path.join(proteus_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DBUILD_SHARED=On",
            f"-DCMAKE_INSTALL_RPATH={self.install_dir}/lib64/",
            "-DCMAKE_SKIP_INSTALL_RPATH=OFF",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=On",
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib64",
            "-DCMAKE_INSTALL_BINDIR=bin",
            "-DCMAKE_INSTALL_INCLUDEDIR=include",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DPROTEUS_ENABLE_CUDA={self.has_nvidia}",
            f"-DPROTEUS_ENABLE_HIP={self.has_amd}",
            "-DENABLE_TESTS=Off",
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            "..",
        ]

        run_command(
            ["cmake"] + cmake_options,
            cwd=build_dir,
        )
        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)
        return self.install_dir

    def clone_and_build_spdlog(self):
        spdlog_path = os.path.abspath("third_party/spdlog")
        if not os.path.exists(spdlog_path):
            run_command(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "v1.15.0",
                    "--single-branch",
                    self.SPDLOG_REPO,
                    spdlog_path,
                ]
            )

        build_dir = os.path.join(spdlog_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        run_command(
            [
                "cmake",
                f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
                "-DCMAKE_INSTALL_LIBDIR=lib64",
                "-DCMAKE_INSTALL_BINDIR=bin",
                "-DCMAKE_INSTALL_INCLUDEDIR=include",
                f"-DCMAKE_C_COMPILER={self.cc}",
                f"-DCMAKE_CXX_COMPILER={self.cxx}",
                "..",
            ],
            cwd=build_dir,
        )

        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)
        return self.install_dir

    def build_mneme(self, proteus_dir, spdlog_dir):
        mneme_path = self.root_dir
        build_dir = os.path.join(mneme_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib64",
            "-DCMAKE_INSTALL_BINDIR=bin",
            "-DCMAKE_INSTALL_INCLUDEDIR=include",
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DMNEME_ENABLE_HIP={self.has_amd}",
            "-DMNEME_ENABLE_TESTS=On",
            "-DMNEME_ENABLE_AUTOTUNE=On",
            f"-DCMAKE_INSTALL_RPATH={self.install_dir}/lib64/",
            "-DCMAKE_SKIP_INSTALL_RPATH=OFF",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON",
            "-DMNEME_ENABLE_LOGGER=On",
            f"-Dproteus_DIR={proteus_dir}",
            f"-Dspdlog_DIR={spdlog_dir}",
        ]

        run_command(["cmake", ".."] + cmake_options, cwd=build_dir)
        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "-j4", "install"], cwd=build_dir)


class CustomBuildPy(build_py):
    """Ensure CMake runs before packaging Python code"""

    def run(self):
        self.run_command("build_ext")
        super().run()


class CustomDevelop(develop):
    def run(self):
        self.run_command("build_ext")
        super().run()


# Setup Configuration
setup(
    name="mneme",
    version="0.1.0",
    author="Konstantinos Parasyris",
    author_email="parasyris1@llnl.gov",
    description="Python bindings for Mneme replay tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=[],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    entry_points={
        "console_scripts": [
            "mneme= mneme.cli:main",
        ],
    },
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "optuna>=4.4",
        "scipy",
        "rich",
        "pandas>=2.3.1",
        "pytest>=7.0",
        "pytest-mock>=3.0",
    ],
)
