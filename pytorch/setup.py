from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="matmult",
    include_dirs=["/home/oem/project/RST/model/include"],
    ext_modules=[
        CUDAExtension(
            "matmult",
            ["/home/oem/project/RST/pytorch/matmult_ops.cpp",
             "/home/oem/project/RST/model/kernel/matmult_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)