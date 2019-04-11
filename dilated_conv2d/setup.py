from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
setup(name='dilated_conv2d',
      ext_modules=[CUDAExtension('dilated_conv2d_gpu', ['dilated_conv2d.cpp', 'dilated_conv2d_cuda.cu']),],
      cmdclass={'build_ext': BuildExtension})
