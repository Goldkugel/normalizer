import sys

# Prevent Python from generating .pyc files (compiled bytecode files)
sys.dont_write_bytecode = True

from .Normalizer import Normalizer

__all__ = ["Normalizer"]