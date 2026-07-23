from pathlib import Path
import os

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

CORE_DIR = Path("armada_game") / "core"
MCTS_DIR = Path("learning") / "mcts"
ENCODING_DIR = Path("learning") / "encoding"
NP_INCLUDE = numpy.get_include()

# /MANIFEST:NO works around a broken rc.exe/link.exe manifest-embedding step on
# some MSVC Build Tools installs (LNK1158); irrelevant for a .pyd loaded by Python.
_NO_MANIFEST = ["/MANIFEST:NO"]

extensions = [
    Extension("para_mcts", [str(MCTS_DIR / "para_mcts.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("shared_mcts", [str(MCTS_DIR / "shared_mcts.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("armada", [str(CORE_DIR / "armada.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("attack_info", [str(CORE_DIR / "attack_info.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("ship", [str(CORE_DIR / "ship.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("squad", [str(CORE_DIR / "squad.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("obstacle", [str(CORE_DIR / "obstacle.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("defense_token", [str(CORE_DIR / "defense_token.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("game_encoder", [str(ENCODING_DIR / "game_encoder.pyx")], include_dirs=[NP_INCLUDE], extra_link_args=_NO_MANIFEST),
    Extension("action_manager", [str(CORE_DIR / "action_manager.pyx")], extra_link_args=_NO_MANIFEST),
]

setup(
    ext_modules=cythonize(
        extensions,
        include_path=[str(CORE_DIR), str(ENCODING_DIR), str(MCTS_DIR)],
        compiler_directives={"language_level": "3", "profile": True},
        force=bool(os.environ.get("CYTHON_FORCE")),
    )
)
