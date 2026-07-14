import os
import sys

# the planner modules (assembly, env, problem, solver, wrench) live in the repo ROOT and
# import each other by bare name; put the root on sys.path so the tests can import them.
#
# NOTE ON DETERMINISM: wA* is suboptimal and `successors` iterates hash-ordered frozensets,
# so the exact plan LENGTH depends on PYTHONHASHSEED (observed jitter ~±1-2). The tests
# therefore assert the hash-INDEPENDENT invariants -- a plan is found, it replays to the
# goal, every edge passes the cheap layer -- plus a loose upper bound on length to catch a
# blow-up. (Making successor order deterministic would tighten this but is an algorithm
# change, out of scope for the regression net.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
