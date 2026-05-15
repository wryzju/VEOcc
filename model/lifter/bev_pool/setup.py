import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="bev_pool_ext",
    ext_modules=[
        CUDAExtension(
            name="bev_pool_ext",
            sources=[
                os.path.join(this_dir, "src", "bev_pool.cpp"),
                os.path.join(this_dir, "src", "bev_pool_cuda.cu"),
            ],
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
