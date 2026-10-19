"""Facade over the domain, solver, and visualization modules.

Import this module as ``assembly_planner.assembly`` to retain the combined API, including
private attributes resolved live through module ``__getattr__``.
"""

import assembly_planner.problem as _problem
import assembly_planner.solver as _solver
import assembly_planner.visualization as _visualization

from assembly_planner.problem import *
from assembly_planner.solver import *
from assembly_planner.visualization import *


def __getattr__(name):
    # live lookup for names `import *` does not bind (private helpers, mutable globals)
    for _m in (_problem, _solver, _visualization):
        try:
            return getattr(_m, name)
        except AttributeError:
            continue
    raise AttributeError(f"module 'assembly_planner.assembly' has no attribute {name!r}")
