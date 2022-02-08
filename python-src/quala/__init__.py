"""
Quasi-Newton algorithms and other accelerators
"""

__version__ = '0.0.1a0'

try:
    from quala._quala import *
    from quala._quala import __version__ as c_version
    assert __version__ == c_version
except ImportError:
    import warnings
    warnings.warn("Failed to import the quala C++ extension")