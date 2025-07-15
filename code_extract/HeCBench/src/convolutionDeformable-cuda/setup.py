import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file + source_cuda
    sources = [os.path.join(extensions_dir, s) for s in sources]

    extension = CUDAExtension

    ext_modules = [
        extension(
            name='_ext',
            sources=sources,
        )
    ]

    return ext_modules

setup(
    name="convolution",
    version="0.1",
    author="",
    url="https://github.com/lucasjinreal/DCNv2_latest",
    description="deformable convolution",
    packages=find_packages(exclude=("configs", "tests")),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)

