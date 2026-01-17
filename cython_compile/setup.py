from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy  # We need this because you cimport numpy

# Define the C extension
extensions = [
    Extension("para_mcts", ["cython_compile/para_mcts.pyx"], include_dirs=[numpy.get_include()]),
    Extension("armada", ["cython_compile/armada.pyx"], include_dirs=[numpy.get_include()]),
    Extension("attack_info", ["cython_compile/attack_info.pyx"], include_dirs=[numpy.get_include()]),
    Extension("ship", ["cython_compile/ship.pyx"], include_dirs=[numpy.get_include()]),
    Extension("squad", ["cython_compile/squad.pyx"], include_dirs=[numpy.get_include()]),
    Extension("obstacle", ["cython_compile/obstacle.pyx"], include_dirs=[numpy.get_include()]),
    Extension("defense_token", ["cython_compile/defense_token.pyx"], include_dirs=[numpy.get_include()]),
    Extension("game_encoder", ["cython_compile/game_encoder.pyx"], include_dirs=[numpy.get_include()]),
    Extension("action_manager", ["cython_compile/action_manager.pyx"]),
]

# Run the setup
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3", 'profile': True}, # Use Python 3 syntax
    )
)

# cython_compile/setup.py