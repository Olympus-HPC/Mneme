# Copyright 2024-2026 Lawrence Livermore National Security, LLC and
# Mneme developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 WITH LLVM-exception)

from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.python import PythonExtension


from spack.package import *


class Mneme(CMakePackage, CudaPackage, ROCmPackage, PythonExtension):
    """Mneme is a framework for recording and replaying GPU kernel executions (CUDA / HIP) as standalone, reproducible executables."""

    homepage = "https://github.com/Olympus-HPC/Mneme"
    git = "https://github.com/Olympus-HPC/Mneme.git"

    license("Apache-2.0 WITH LLVM-exception")

    version("develop", branch="develop")

    variant(
        "python",
        default=True,
        description="Build Python bindings, CLI, profiling, and autotuning.",
    )
    variant(
        "tests",
        default=False,
        description="Build and enable the test suite.",
    )

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.28.2:", type="build")

    depends_on("llvm@19:20 +clang targets=all", when="+cuda")
    # ROCm: use the AMDGPU LLVM build
    depends_on("llvm-amdgpu@6.2:", when="+rocm")
    requires("%[virtuals=c,cxx] llvm-amdgpu", when="+rocm")
    requires("%[virtuals=c,cxx] llvm", when="+cuda")

    depends_on("cuda@12:", when="+cuda")
    depends_on("hip@6.2:", when="+rocm")
    depends_on("proteus@=2026.05.0+impl_headers+shared")

    with when("+rocm"):
        for arch in ROCmPackage.amdgpu_targets:
            depends_on(f"proteus amdgpu_target={arch}", when=f"amdgpu_target={arch}")

    with when("+cuda"):
        for arch in CudaPackage.cuda_arch_values:
            depends_on(f"proteus cuda_arch={arch}", when=f"cuda_arch={arch}")

    depends_on("spdlog@1.15.0")

    with when("+python"):
        extends("python")
        depends_on("python@3.9:", type=("build", "run"))
        depends_on("py-optuna@4.4:", type=("build", "run"))
        depends_on("py-scipy", type=("build", "run"))

        # python tests
        with when("+tests"):
            depends_on("py-pytest@7.0:", type=("build", "run"))
            depends_on("py-pytest-mock@3.0:", type=("build", "run"))
            depends_on("py-pytest-cov", type=("build", "run"))

    conflicts(
        "+cuda +rocm",
        msg="Mneme cannot be built with both +cuda and +rocm",
    )
    conflicts(
        "~cuda ~rocm",
        msg="Mneme requires either +cuda or +rocm",
    )

    def cmake_args(self):
        args = []

        if "llvm-amdgpu" in self.spec:
            llvm_provider = self.spec["llvm-amdgpu"]
        elif "llvm" in self.spec:
            llvm_provider = self.spec["llvm"]
        else:
            raise InstallError("Mneme requires an LLVM provider")

        args.append(self.define("LLVM_INSTALL_DIR", llvm_provider.prefix))

        args.append(self.define_from_variant("MNEME_ENABLE_TESTS", "tests"))
        args.append(self.define_from_variant("MNEME_ENABLE_HIP", "rocm"))
        args.append(self.define_from_variant("MNEME_ENABLE_CUDA", "cuda"))
        args.append(self.define_from_variant("MNEME_ENABLE_PYTHON", "python"))
        args.append(self.define("MNEME_ENABLE_LOGGER", True))
        args.append(self.define("MNEME_PYTHON_WHEEL", False))

        if self.spec.satisfies("+python"):
            args.append(self.define("MNEME_PYTHON_SITE_PACKAGES", python_purelib))

        return args
