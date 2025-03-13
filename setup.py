import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


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


# Detect if `--with-tests` flag is passed
WITH_TESTS = "--with-tests" in sys.argv
if WITH_TESTS:
    sys.argv.remove("--with-tests")  # Remove it so it doesn’t interfere with setuptools


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
                ["git", "clone", "--depth", "1", self.PROTEUS_REPO, proteus_path],
                cwd="third_party",
            )

            # Ensure commands are run inside the Proteus directory
            run_command(
                [
                    "git",
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    "30f766dbbff8599479739450eee3fdb9bdd3c118",
                ],
                cwd=proteus_path,
            )
            run_command(
                ["git", "checkout", "30f766dbbff8599479739450eee3fdb9bdd3c118"],
                cwd=proteus_path,
            )

        build_dir = os.path.join(proteus_path, "build")
        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [
            "-DBUILD_SHARED=Off",
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DPROTEUS_ENABLE_CUDA={self.has_nvidia}",
            f"-DPROTEUS_ENABLE_HIP={self.has_amd}",
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
            f"-DCMAKE_INSTALL_PREFIX={self.install_dir}",
            f"-DCMAKE_C_COMPILER={self.cc}",
            f"-DCMAKE_CXX_COMPILER={self.cxx}",
            f"-DLLVM_INSTALL_DIR={self.llvm_dir}",
            f"-DMNEME_ENABLE_HIP={self.has_amd}",
            "-DMNEME_ENABLE_TESTS=On",
            "-DMNEME_ENABLE_PYTHON=On",
            f"-Dproteus_DIR={self.install_dir}",
            f"-Dspdlog_DIR={self.install_dir}",
        ]

        # Enable Mneme's tests if --with-tests was passed
        if WITH_TESTS:
            cmake_options.append("-DMNEME_BUILD_TESTS=ON")

        run_command(["cmake", ".."] + cmake_options, cwd=build_dir)
        run_command(["make", "-j4"], cwd=build_dir)
        run_command(["make", "install"], cwd=build_dir)


# Define the Python Extension
ext_modules = [
    Extension(
        "mneme_python",
        sources=["src/mneme_bindings.cpp"],
        include_dirs=[
            "third_party/proteus/install/include",
            "third_party/spdlog/install/include",
        ],
        libraries=["proteus", "spdlog"],
        library_dirs=[
            "third_party/proteus/install/lib",
            "third_party/spdlog/install/lib",
        ],
        extra_compile_args=["-std=c++17"],
        language="c++",
    )
]

# Setup Configuration
setup(
    name="mneme-python",
    version="0.1.0",
    author="Konstantinos Parasyris",
    author_email="your_email@example.com",
    description="Python bindings for Mneme replay tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        "pybind11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
