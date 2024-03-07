try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = [
    'basis_functions',
    'spline_curves'
]

from .basis_functions import *
from .spline_curves import *
