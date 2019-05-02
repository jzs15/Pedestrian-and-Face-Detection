import numpy as np
import os
from os.path import join as pjoin
#from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess

#change for windows, by MrX
nvcc_bin = 'nvcc.exe'
lib_dir = 'lib/x64'

import distutils.msvc9compiler
distutils.msvc9compiler.VERSION = 14.0

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules = [
    # unix _compile: obj, src, ext, cc_args, extra_postargs, pp_opts
    Extension(
        "bbox",
        sources=["bbox.pyx"],
        extra_compile_args={},
        include_dirs = [numpy_include]
    ),
]

setup(
    name='fast_rcnn',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': build_ext},
)