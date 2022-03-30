import platform
from setuptools import Extension, setup
import os
import numpy as np
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[np.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


if __name__ == '__main__':
    setup(
        name='extension',
        ext_modules=[
            make_cython_ext(
                name='rbox_overlaps',
                module='utils.rotation_overlaps',
                sources=['rbox_overlaps.pyx']),

            make_cython_ext(
                name='cpu_nms',
                module='utils.rotation_nms',
                sources=['cpu_nms.pyx']),
        ],
        cmdclass={'build_ext': BuildExtension},
    )