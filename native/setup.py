from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='srelu_cuda',
    ext_modules=[
        CUDAExtension('srelu_cuda', [
            'srelu_cuda.cpp',
            'srelu_cuda_kernel.cu',
        ],
          extra_compile_args={'cxx': [],
                              'nvcc': ['-gencode=arch=compute_60,code="sm_60,compute_60"']}
          ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
