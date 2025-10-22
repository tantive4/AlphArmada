from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy  # We need this because you cimport numpy

# Define the C extension
extensions = [
    Extension("para_mcts", ["cython_compile/para_mcts.pyx"], include_dirs=[numpy.get_include()]),
    Extension("armada", ["cython_compile/armada.pyx"]),
    Extension("attack_info", ["cython_compile/attack_info.pyx"]),
    Extension("ship", ["cython_compile/ship.pyx"],include_dirs=[numpy.get_include()]),
    Extension("squad", ["cython_compile/squad.pyx"]),
    Extension("defense_token", ["cython_compile/defense_token.pyx"]),
]

# Run the setup
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"} # Use Python 3 syntax
    )
)

# python cython_compile/setup.py build_ext --inplace