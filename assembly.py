"""Backwards-compatible facade. The implementation was split into problem.py (domain +
search model), solver.py (feasibility + A*) and visualization.py (viewers); this module
re-exports them so `import assembly as A` keeps working, including private `A._name`
access (resolved live via module __getattr__)."""

import problem as _problem
import solver as _solver
import visualization as _visualization

from problem import *
from solver import *
from visualization import *


def __getattr__(name):
    # live lookup for names `import *` does not bind (private helpers, mutable globals)
    for _m in (_problem, _solver, _visualization):
        try:
            return getattr(_m, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'assembly' has no attribute {name!r}")
