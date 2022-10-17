"""
Quasi-Newton algorithms and other accelerators
"""

__version__ = '0.0.1a1'

import os
import typing

def _is_truthy(s: typing.Optional[str]):
    if s is None: return False
    return not s.lower() in ('', 'false', 'no', 'off', '0')

if not typing.TYPE_CHECKING and _is_truthy(os.getenv('QUALA_PYTHON_DEBUG')):
    from quala._quala_d import *
    from quala._quala_d import __version__ as __c_version__
else:
    from quala._quala import *
    from quala._quala import __version__ as __c_version__
assert __version__ == __c_version__

del _is_truthy, typing, os