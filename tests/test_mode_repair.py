"""The dropped zero-motion-mode failure class (2026-07-31): mr_bench's shortcutting can
erase a zero-motion task's whole mode stretch from a saved path (its zero-length dedup
branch never checks the middle state's mode), and the sim derives fire events from mode
transitions -- so the task's attach/release side effects silently never happen (the chair-8
holder-release: the arm kept its weld and dragged the chair). repair_mode_chain re-inserts
missing modes from the symbolic chain; these tests pin that contract."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import assembly_planner.mp_export as MP
import assembly_planner.retiming as retiming


def _m(ids):
  return MP._Mode(list(ids))


def _st(q, mode):
  return MP._PathState(np.asarray(q, float), mode)


def _mk(ref, mode):
  return MP._PathState(ref.q, MP._Mode([int(t) for t in np.atleast_1d(mode.task_ids)]))


# a chair-8-shaped chain: per-robot active-task-id tuples; [4, 4, 6] is the zero-motion
# release mode that shortcutting collapses (both its robots come straight from the assemble).
CHAIN = [_m([0, 1, 6]), _m([2, 1, 6]), _m([3, 3, 6]), _m([4, 4, 6]), _m([7, 5, 6]),
         _m([7, 7, 7])]


def _dirty_path():
  # the observed artifact: the release mode has NO state; the boundary survives as a
  # duplicated-config pair tagged with the surrounding modes.
  q = [[0.0] * 3, [1.0] * 3, [2.0] * 3, [2.0] * 3, [3.0] * 3, [4.0] * 3]
  modes = [CHAIN[0], CHAIN[1], CHAIN[2], CHAIN[4], CHAIN[4], CHAIN[5]]
  return [_st(qi, mi) for qi, mi in zip(q, modes)]


def _done_union(path):
  stretches = retiming.split_stretches(path, q_of=lambda st: st.q,
                                       mode_of=lambda st: st.mode,
                                       taskids_of=lambda m: np.atleast_1d(m.task_ids))
  out = set()
  for _w, tids in stretches:
    out.update(int(t) for t in tids)
  return out


def test_unrepaired_path_never_fires_dropped_task():
  # documents WHY the repair exists: split_stretches' done = ids(old) - ids(new) cannot see
  # a task that is active only inside the collapsed mode.
  assert 4 not in _done_union(_dirty_path())


def test_repair_reinserts_dropped_mode_at_boundary_config():
  out, n = MP.repair_mode_chain(_dirty_path(), CHAIN, _mk, verbose=False)
  assert n == 1
  ids = [tuple(s.mode.task_ids) for s in out]
  j = ids.index((4, 4, 6))
  assert np.array_equal(out[j].q, out[j - 1].q)


def test_repaired_path_fires_every_task():
  out, _ = MP.repair_mode_chain(_dirty_path(), CHAIN, _mk, verbose=False)
  assert _done_union(out) == set(range(8))


def test_repair_noop_on_clean_path():
  clean = [_st([float(i)] * 3, m) for i, m in enumerate(CHAIN)]
  out, n = MP.repair_mode_chain(clean, CHAIN, _mk, verbose=False)
  assert n == 0
  assert len(out) == len(clean) and all(a is b for a, b in zip(out, clean))


def test_repair_bails_on_foreign_mode():
  # a path whose mode is not in the chain is a mismatched plan/path pair: leave it for the
  # task-signature guards, do not "repair" against the wrong chain.
  path = _dirty_path()
  path[1] = _st(path[1].q, _m([99, 99, 99]))
  out, n = MP.repair_mode_chain(path, CHAIN, _mk, verbose=False)
  assert n == 0 and len(out) == len(path)


def test_repair_handles_multiple_consecutive_drops():
  # car-mini shape: several zero-motion modes collapsed at ONE boundary (all-slot jump).
  q = [[0.0] * 3, [1.0] * 3, [1.0] * 3, [4.0] * 3]
  modes = [CHAIN[0], CHAIN[1], CHAIN[4], CHAIN[5]]          # drops CHAIN[2] AND CHAIN[3]
  out, n = MP.repair_mode_chain([_st(qi, mi) for qi, mi in zip(q, modes)], CHAIN, _mk,
                                verbose=False)
  assert n == 2
  assert _done_union(out) == set(range(8))
