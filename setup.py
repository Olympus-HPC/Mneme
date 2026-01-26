import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


# Helper function to run shell commands
def run_command(command, cwd=None):
    sys.stderr.write(f"Running: {' '.join(command)} in {cwd or os.getcwd()}")
    result = subprocess.run(command, cwd=cwd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command {' '.join(command)} failed")


def detect_local_sm_via_nvidia_smi() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True,
    ).strip()
    # If multiple GPUs, take the first line
    cc = out.splitlines()[0].strip()  # e.g. "9.0"
    major, minor = cc.split(".")
    return int(major + minor)  # "9"+"0" -> 90


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


def get_llvm_config(llvm_dir):
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
        "lib_names": run("--libnames"),
        "shared_mode": run("--shared-mode"),
    }


# Custom build class for CMake
class CMakeBuild(build_ext):
    PROTEUS_REPO = "https://github.com/Olympus-HPC/proteus.git"
    SPDLOG_REPO = "https://github.com/gabime/spdlog.git"

    def initialize_options(self):
        super().initialize_options()
        self.root_dir = Path(__file__).resolve().parent
        build_cmd = self.get_finalized_command("build")
        self.build_lib = Path(build_cmd.build_lib).resolve()
        build_pkg_dir = self.build_lib / "mneme"

        src_pkg_dir = self.root_dir / "python" / "mneme"
        inplace = bool(getattr(self, "inplace", False))
        self.editable = (
            "editable_wheel" in sys.argv or inplace
        )  # include editable_wheel to be pep 660 compliant

        pkg_dir = src_pkg_dir if self.editable else build_pkg_dir

        self.install_dir = pkg_dir / "native"
        self.config_json = self.install_dir / "config.json"

        self.install_dir.mkdir(parents=True, exist_ok=True)
        (self.install_dir / "lib64").mkdir(parents=True, exist_ok=True)
        (self.install_dir / "include").mkdir(parents=True, exist_ok=True)
        (self.install_dir / "lib64" / "cmake").mkdir(parents=True, exist_ok=True)
        (self.install_dir / "llvm").mkdir(parents=True, exist_ok=True)

        self.has_nvidia = "On" if has_nvidia_gpu() else "Off"
        self.cuda_arch = "native"
        if self.has_nvidia == "On":
            self.cuda_arch = detect_local_sm_via_nvidia_smi()
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
        self.llvm_config = get_llvm_config(self.llvm_dir)

        cfg = {
            "cc": self.cc,
            "cxx": self.cxx,
            "prefix": "@PREFIX@",
            "libdir": "@PREFIX@/lib64",
            "includedir": "@PREFIX@/include",
            "cmakedir": "@PREFIX@/lib64/cmake",
            "cflags": f"-fpass-plugin=@PREFIX@/lib64/libProteusPass.so -fplugin=@PREFIX@/lib64/libProteusPass.so -fno-discard-value-names -ftrivial-auto-var-init=zero -Xclang -mllvm -Xclang -force-proteus-jit-annotate-all",
            "ldflags": f"-L{self.llvm_dir}/lib -L{self.llvm_dir}/llvm/lib {self.llvm_config['libs']} {self.llvm_config['system_libs']} -L@PREFIX@/lib64/ -Wl,-rpath,@PREFIX@/lib64/ -llldCommon -llldELF -lproteus",
        }

        if not prefix.exists():
            prefix.mkdir(parents=True, exist_ok=True)
        with open(self.config_json, "w") as fd:
            json.dump(cfg, fd, indent=2)

    def run(self):
        self.build_scratch = Path(self.build_temp).resolve()
        self.build_scratch.mkdir(parents=True, exist_ok=True)
        self.build_scratch = str(self.build_scratch)

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
            proteus_path = os.path.abspath(f"{self.build_scratch}/proteus")
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
                    cwd=self.build_scratch,
                )
                run_command(
                    [
                        "git",
                        "fetch",
                        "--depth",
                        "1",
                        "origin",
                        "cuda-proteus-core-shared-llvm",
                    ],
                    cwd=str(Path(self.build_scratch) / "proteus"),
                )
                run_command(
                    ["git", "checkout", "-b", "features/cuda-shared", "FETCH_HEAD"],
                    cwd=str(Path(self.build_scratch) / "proteus"),
                )

        build_dir = os.path.join(proteus_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DBUILD_SHARED=On",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DCMAKE_SKIP_INSTALL_RPATH=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=On",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            "-DCMAKE_INSTALL_LIBDIR=lib64",
            "-DCMAKE_INSTALL_BINDIR=bin",
            "-DCMAKE_INSTALL_INCLUDEDIR=include",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DPROTEUS_ENABLE_CUDA={self.has_nvidia}",
            f"-DPROTEUS_ENABLE_HIP={self.has_amd}",
        ]

        if self.has_nvidia == "On":
            cmake_options.append(f"-DCMAKE_CUDA_ARCHITECTURES={self.cuda_arch}")
            cmake_options.append(f"-DCMAKE_CUDA_COMPILER={self.cxx}")
            cmake_options.append("-DCMAKE_CUDA_FLAGS=-std=c++17")

        cmake_options.append("-DENABLE_TESTS=Off")
        cmake_options.append(f"-DCMAKE_C_COMPILER={self.cc}")
        cmake_options.append(f"-DCMAKE_CXX_COMPILER={self.cxx}")
        cmake_options.append("..")

        run_command(
            ["cmake"] + cmake_options,
            cwd=build_dir,
        )
        run_command(["make", "-j10"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)
        return self.install_dir

    def clone_and_build_spdlog(self):
        spdlog_path = os.path.abspath(f"{self.build_scratch}/spdlog")
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
                f"-DCMAKE_INSTALL_PREFIX={str(Path(self.install_dir).resolve())}",
                "-DCMAKE_INSTALL_LIBDIR=lib64",
                "-DCMAKE_INSTALL_BINDIR=bin",
                "-DCMAKE_INSTALL_INCLUDEDIR=include",
                f"-DCMAKE_CXX_COMPILER={self.cxx}",
                "..",
            ],
            cwd=build_dir,
        )

        run_command(["make", "-j10"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)
        return self.install_dir

    def build_mneme(self, proteus_dir, spdlog_dir):
        mneme_path = str(self.root_dir)
        build_dir = os.path.join(self.build_scratch, "mneme/build")
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
            f"-DMNEME_ENABLE_CUDA={self.has_nvidia}",
            "-DMNEME_ENABLE_TESTS=Off",
            "-DMNEME_ENABLE_AUTOTUNE=On",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
            "-DCMAKE_SKIP_INSTALL_RPATH=OFF",
            "-DMNEME_ENABLE_LOGGER=On",
        ]

        if self.llvm_config["shared_mode"] == "shared":
            cmake_options.append("-DMNEME_LINK_SHARED_LLVM=On")

        if self.has_nvidia == "On":
            cmake_options.append(f"-DCMAKE_CUDA_ARCHITECTURES={self.cuda_arch}")
            cmake_options.append(f"-DCMAKE_CUDA_COMPILER={self.cxx}")
            cmake_options.append("-DCMAKE_CUDA_FLAGS=-std=c++17")

        cmake_options += [
            f"-DCMAKE_PREFIX_PATH={str(Path(self.install_dir).resolve())}",
        ]

        run_command(["cmake", mneme_path] + cmake_options, cwd=build_dir)
        run_command(["make", "-j10"], cwd=build_dir)
        run_command(["make", "-j10", "install"], cwd=build_dir)


class CustomDevelop(develop):
    def run(self):
        super().run()
        self.run_command("build_ext")


class CustomEggInfo(egg_info):
    def run(self):
        try:
            from pathlib import Path

            root = Path(__file__).resolve().parent
            so = root / "python" / "mneme" / "native" / "lib64" / "libmneme.so"
            if not so.exists():
                self.run_command("build_ext")
        except Exception:
            # if anything goes wrong, don't block egg_info
            pass
        super().run()


class CustomBuildPy(build_py):
    def run(self):
        build_ext_cmd = self.get_finalized_command("build_ext")
        build_ext_cmd.inplace = True
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
    package_data={
        "": [
            "native/**/*",
            "llvm/_lib_path_config.py",
        ]
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "mneme= mneme.cli:main",
        ],
    },
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CustomBuildPy,
        "develop": CustomDevelop,
        "egg_info": CustomEggInfo,
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
        "pytest>=7.0",
        "pytest-mock>=3.0",
    ],
)
