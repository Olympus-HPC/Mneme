import glob
import os
import shutil
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


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


# Custom build class for CMake
class CMakeBuild(build_ext):
    PROTEUS_REPO = "https://github.com/Olympus-HPC/proteus.git"
    SPDLOG_REPO = "https://github.com/gabime/spdlog.git"

    def initialize_options(self):
        super().initialize_options()
        self.root_dir = os.path.abspath(os.path.dirname(__file__))
        build_cmd = self.get_finalized_command("build")
        self.build_lib = build_cmd.build_lib
        self.install_dir = os.path.abspath(self.build_lib)
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

    def run(self):
        if not os.path.exists("third_party"):
            os.mkdir("third_party")

        self.clone_and_build_proteus()
        self.clone_and_build_spdlog()
        self.build_mneme()

    def clone_and_build_proteus(self):
        proteus_path = os.path.abspath("third_party/proteus")
        if not os.path.exists(proteus_path):
            run_command(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "features/mneme-integrations",
                    self.PROTEUS_REPO,
                    proteus_path,
                ],
                cwd="third_party",
            )

            # Ensure commands are run inside the Proteus directory
            # run_command(
            #    [
            #        "git",
            #        "fetch",
            #        "--depth",
            #        "1",
            #        "origin",
            #        "30f766dbbff8599479739450eee3fdb9bdd3c118",
            #    ],
            #    cwd=proteus_path,
            # )
            # run_command(
            #    ["git", "checkout", "30f766dbbff8599479739450eee3fdb9bdd3c118"],
            #    cwd=proteus_path,
            # )

        build_dir = os.path.join(proteus_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DBUILD_SHARED=Off",
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
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
                f"-DCMAKE_C_COMPILER={self.cc}",
                f"-DCMAKE_CXX_COMPILER={self.cxx}",
                "..",
            ],
            cwd=build_dir,
        )

        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)

    def build_mneme(self):
        mneme_path = self.root_dir
        build_dir = os.path.join(mneme_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DCMAKE_BUILD_TYPE=Relwithdebinfo",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DMNEME_ENABLE_HIP={self.has_amd}",
            "-DMNEME_ENABLE_TESTS=On",
            "-DMNEME_ENABLE_AUTOTUNE=On",
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=On",
            f"-Dproteus_DIR={self.install_dir}",
            f"-Dspdlog_DIR={self.install_dir}",
        ]

        run_command(["cmake", ".."] + cmake_options, cwd=build_dir)
        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "-j4", "install"], cwd=build_dir)

        build_module = [
            p
            for p in glob.glob(os.path.join(self.install_dir, "lib64", "libmneme*.so"))
        ]
        build_module += [
            p
            for p in glob.glob(
                os.path.join(self.install_dir, "lib64", "libmneme_profile*.so")
            )
        ]

        if not build_module:
            raise RuntimeError(
                "Error: mneme shared libraries not found in build output!"
            )

        python_package_dir = os.path.join(self.build_lib, "mneme")
        for built_module in build_module:
            sys.stderr.write(f"Writting package to {python_package_dir}")
            os.makedirs(python_package_dir, exist_ok=True)
            shutil.copy(built_module, python_package_dir)


class CustomBuildPy(build_py):
    """Ensure CMake runs before packaging Python code"""

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
    cmdclass={"build_ext": CMakeBuild, "build_py": CustomBuildPy},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["optuna>=4.4", "scipy"],
)
