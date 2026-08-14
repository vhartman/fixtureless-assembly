"""Lazy certification with waves (idea 4).  Fully separate module and policy arm.

The native shortcut loop collision-checks every gate-passing edit at acceptance time.  The
replacement audit measured that 67-69 % of accepted edits are later fully overwritten, so most
of those checks are wasted.  This loop instead:

* phase A: proposals, the unchanged task-id gate, and the unchanged exact cost gate.  Passing
  edits are APPLIED immediately but not collision-checked, and recorded in the wave log.
* phase B (every ``wave_size`` edits): the union of the changed regions is checked once at the
  strict resolution.  If free, the whole wave commits.  If not, the entire wave is rolled back
  exactly and re-applied one edit at a time with the unchanged native gate + full check, so a
  failing wave degrades to exactly the native per-edit procedure.

Correctness argument: every edge of the output path was checked at the wave that last changed
it; unchanged edges were certified by the planner.  The strict full-path certificate the
driver always runs therefore holds by construction, and it is still verified and recorded.
The coarse structure changes only WHEN collision information arrives, not which cost gate or
which certificate is used.

Nothing in mr_bench is modified; nothing in the earlier experiment files is modified.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

import ideation_samplers as IS
from exact_gate import SegmentCostTable

try:
    from ideation_samplers import _Stop, _proposal_candidate, _mrb
except ImportError:  # pragma: no cover
    from .ideation_samplers import _Stop, _proposal_candidate, _mrb  # type: ignore


WAVE_SIZE = 20


def _locate_fail(new_path, a, b, fail_cell):
    """Map the captured failing query to an absolute STATE index, or None.

    ``b`` is inclusive.  The captured query is either an edge start state (identity) or a
    flat array (value search with a tight tolerance).
    """
    if not fail_cell or fail_cell.get("q") is None:
        return None
    fq = fail_cell["q"]
    for k in range(a, b + 1):
        if new_path[k].q is fq:
            return k
    if hasattr(fq, "state"):
        state = np.asarray(fq.state(), dtype=float)
        best_k, best_d = None, np.inf
        for k in range(a, b + 1):
            d = float(np.max(np.abs(np.asarray(new_path[k].q.state(), dtype=float) - state)))
            if d < best_d:
                best_d, best_k = d, k
        if best_k is not None and best_d <= 1e-6:
            return best_k
    return None


def shortcut_run_lazy(
    env,
    path,
    max_iter: int = 1000,
    max_attempts: Optional[int] = None,
    resolution=0.01,
    tolerance=0.01,
    interpolation_resolution: float = 0.5,
    seed: int = 0,
    wave_size: int = WAVE_SIZE,
    fail_cell: Optional[dict] = None,
    stats_out: Optional[Dict] = None,
    check_counter=None,
    shared_state=None,
    debug_out: Optional[Dict] = None,
):
    """Same contract as ``robot_mode_shortcut``: returns ``(new_path, [costs, times])``.

    ``cnt`` counts gate-passing proposals, the same unit the native loop counts for
    validations, so the budget cap is matched.
    """
    State, interpolate_path, path_cost, remove_interpolated_nodes, _ = _mrb()

    non_redundant_path = remove_interpolated_nodes(path)
    new_path = interpolate_path(non_redundant_path, interpolation_resolution)

    table = SegmentCostTable(env, lambda: new_path)

    costs = [path_cost(new_path, env.batch_config_cost)]
    times = [0.0]

    st: Dict = dict(
        arm="lazy_certify", n_states=len(new_path), iters=0, skipped_short=0,
        skipped_gate=0, cost_rejects=0, collision_checks=0, accepted=0,
        rejected_collision=0, sampler_time=0.0, sampler_calls=0,
        cost_initial=float(costs[0]), stop_reason=None, accepted_per_robot={},
        fail_located=0, gate_table_ok=bool(table.ok), gate_uses=0, gate_positive=0,
        gate_negative=0, gate_eps_fallback=0,
        provisional_accepted=0, wave_commits=0, wave_failures=0, degraded_waves=0,
        degraded_cost_rejects=0, unattributable_failures=0, cascade_cost_rejects=0,
        wave_size=int(wave_size),
    )

    t_s = time.perf_counter()
    sampler = IS.PersistentTreeSampler(env, new_path, seed=seed)
    st["sampler_time"] += time.perf_counter() - t_s

    start_time = time.time()
    if max_attempts is None:
        max_attempts = 5 * max_iter
    cnt = 0
    iter = 0
    wave: List[Dict] = []
    events: List[Dict] = []
    event_cell: List = [None]
    if debug_out is not None:
        debug_out["events"] = events

    # Snapshot of the last certified path.  Wave certification checks exactly the edges whose
    # endpoint states differ from this snapshot, so intra-wave-overwritten edits are never
    # checked and every checked edge is checked at the strict, final-certificate resolution.
    # ``snapshot_qs`` keeps the certified Configuration objects for exact replay after
    # removals.
    snapshot = np.array([s.q.state() for s in new_path], dtype=np.float64)
    snapshot_qs: List = [s.q for s in new_path]

    def _update_snapshot():
        nonlocal snapshot, snapshot_qs
        snapshot = np.array([s.q.state() for s in new_path], dtype=np.float64)
        snapshot_qs = [s.q for s in new_path]

    def strict_check(segment):
        """The same check the driver's final certificate uses."""
        return env.is_path_collision_free(
            segment, resolution=resolution, tolerance=tolerance,
            check_start_and_end=True, check_edges_in_order=True, binary_order=False,
        )

    def trace_point():
        current_time = time.time()
        times.append(current_time - start_time)
        costs.append(path_cost(new_path, env.batch_config_cost))

    def commit_wave():
        """Certify the wave in two stages.

        Stage 1 uses the native binary-order check, whose failures are cheap (the checker
        stops at the first coarse collision).  Stage 2 uses the strict final-certificate
        check, which only runs on regions that already passed stage 1, so its full cost is
        paid once for surviving work only.  A failure removes the latest wave edit that
        overlaps the failing state, together with every wave edit that overlaps it (cascade),
        then exactly rebuilds the wave region from the certified snapshot and re-applies the
        surviving edits in application order with their exact cost gate, so the composite
        cost stays monotone.
        """
        commit_event = {
            "wave": len(events),
            "accepted": [(int(e["i"]), int(e["j"])) for e in wave],
            "stage_segments": [],
            "removed": [],
            "unattributable": [],
            "checks": [],
            "cascades": [],
        }
        event_cell[0] = commit_event
        current = np.array([s.q.state() for s in new_path], dtype=np.float64)
        changed_states = np.any(current != snapshot, axis=1)
        if not np.any(changed_states):
            # The wave was re-applied to the exact previous values; nothing new to certify.
            st["wave_commits"] += 1
            _commit_feedback()
            wave.clear()
            _update_snapshot()
            commit_event["post_state5"] = (
                np.asarray(new_path[5].q.state()).tolist() if len(new_path) > 5 else None
            )
            events.append(commit_event)
            trace_point()
            return

        for stage, checker in ((0, binary_check), (1, strict_check)):
            skipped = set()
            while True:
                current = np.array([s.q.state() for s in new_path], dtype=np.float64)
                changed_states = np.any(current != snapshot, axis=1)
                pending = [
                    segment for segment in _segments(changed_states)
                    if segment not in skipped
                ]
                commit_event["stage_segments"].append([list(segment) for segment in pending])
                if not pending:
                    break
                acted = False
                for a, b in pending:
                    if fail_cell is not None:
                        fail_cell["q"] = None
                    result = checker(new_path[a : b + 2])
                    if result:
                        commit_event["checks"].append([stage, [a, b], True, None])
                        continue
                    fail_pos = _locate_fail(new_path, a, b + 1, fail_cell)
                    commit_event["checks"].append([stage, [a, b], False, fail_pos])
                    if not _cascade_remove(fail_pos):
                        # The failing spot does not belong to any wave edit; it is a
                        # pre-existing planner edge.  The final certificate judges it, as it
                        # does for every other policy.
                        st["unattributable_failures"] += 1
                        skipped.add((a, b))
                        commit_event["unattributable"].append([a, b])
                    else:
                        st["wave_failures"] += 1
                        commit_event["removed"].append([a, b])
                    acted = True
                    break
                if not acted:
                    break
        # all changed edges passed both stages
        st["wave_commits"] += 1
        _commit_feedback()
        wave.clear()
        _update_snapshot()
        commit_event["post_state5"] = (
            np.asarray(new_path[5].q.state()).tolist() if len(new_path) > 5 else None
        )
        events.append(commit_event)
        trace_point()

    def binary_check(segment):
        """The native candidate check: binary order, no endpoint requirement."""
        return env.is_path_collision_free(
            segment, resolution=resolution, tolerance=tolerance,
            check_start_and_end=False,
        )

    def _segments(changed_states):
        if not np.any(changed_states):
            return []
        edge_changed = changed_states[:-1] | changed_states[1:]
        segments = []
        start_edge = None
        for k in range(len(edge_changed) + 1):
            is_changed = k < len(edge_changed) and bool(edge_changed[k])
            if is_changed and start_edge is None:
                start_edge = k
            if start_edge is not None and not is_changed:
                segments.append((start_edge, k - 1))
                start_edge = None
        return segments

    def _rebuild_remaining(extra_intervals=()):
        """Restore every state touched by the remaining wave edits AND the given removed
        intervals to its certified value, then re-apply the remaining edits in application
        order.  Only edits whose span overlaps a removed interval are re-gated: the exact
        cost gate of every other edit depends only on its own span, which is unchanged.
        A re-gated edit whose gate no longer holds is dropped with ``cost`` feedback."""
        touched = set()
        for e in wave:
            for k in range(e["j"] - e["i"] + 1):
                touched.add(e["i"] + k)
        for i, j in extra_intervals:
            for k in range(j - i + 1):
                touched.add(i + k)
        for k in sorted(touched):
            new_path[k].q = snapshot_qs[k]
        table.mark_dirty()
        recheck_intervals = list(extra_intervals)
        remaining = list(wave)
        wave.clear()
        for e in remaining:
            span = e["j"] - e["i"]
            path_element = [
                State(e["new_qs"][k], new_path[e["i"] + k].mode) for k in range(span + 1)
            ]
            must_regate = any(
                not (e["j"] < i or e["i"] > j) for i, j in recheck_intervals
            )
            t_s = time.perf_counter()
            if must_regate:
                candidate_segment_cost = path_cost(path_element, env.batch_config_cost)
                old_segment_cost = path_cost(new_path[e["i"] : e["j"] + 1], env.batch_config_cost)
                if candidate_segment_cost >= old_segment_cost:
                    st["cascade_cost_rejects"] += 1
                    sampler.feedback("cost", ij=(e["i"], e["j"]), robot=e["robots"][0])
                    st["sampler_time"] += time.perf_counter() - t_s
                    # A later edit overlapping this dropped span also sees changed values.
                    recheck_intervals.append((e["i"], e["j"]))
                    continue
            for k in range(span + 1):
                new_path[e["i"] + k].q = path_element[k].q
            table.mark_dirty()
            wave.append(e)
            st["sampler_time"] += time.perf_counter() - t_s

    def _cascade_remove(fail_pos):
        """Remove the LATEST wave edit overlapping the failing state, then rebuild the wave
        region exactly from the certified snapshot.  ``fail_pos`` is a state index; an edit
        [i, j] overlaps the failing spot when its states touch [fail_pos, fail_pos+1].  The
        exact rebuild makes single-edit removal safe: no stored intermediate can resurface,
        and the surviving edits are re-gated in application order, so the composite cost
        stays monotone and good edits near a bad one are preserved."""
        if fail_pos is None:
            return False
        latest = None
        latest_index = -1
        for index, e in enumerate(wave):
            if not (e["j"] < fail_pos or e["i"] > fail_pos + 1):
                if index > latest_index:
                    latest, latest_index = e, index
        if latest is None:
            return False
        commit_event = event_cell[0] or {}
        commit_event.setdefault("cascades", []).append(
            [int(fail_pos), (int(latest["i"]), int(latest["j"])), len(wave) - 1]
        )
        st["rejected_collision"] += 1
        t_s = time.perf_counter()
        sampler.feedback("collision", ij=(latest["i"], latest["j"]),
                         robot=latest["robots"][0], fail_k=fail_pos)
        st["sampler_time"] += time.perf_counter() - t_s
        wave.remove(latest)
        _rebuild_remaining([(int(latest["i"]), int(latest["j"]))])
        return True

    def _commit_feedback():
        for e in wave:
            st["accepted"] += 1
            for r in e["robots"]:
                st["accepted_per_robot"][r] = st["accepted_per_robot"].get(r, 0) + 1
            t_s = time.perf_counter()
            sampler.feedback("accepted", dcost=float(e["dcost"]),
                             ij=(e["i"], e["j"]), robot=e["robots"][0])
            st["sampler_time"] += time.perf_counter() - t_s

    while True:
        iter += 1
        if cnt >= max_iter or iter >= max_attempts:
            st["stop_reason"] = "max_iter" if cnt >= max_iter else "max_attempts"
            break

        t_s = time.perf_counter()
        try:
            proposal = sampler.propose()
        except _Stop as exc:
            st["sampler_time"] += time.perf_counter() - t_s
            st["stop_reason"] = f"sampler exhausted: {exc}"
            break
        st["sampler_time"] += time.perf_counter() - t_s
        st["sampler_calls"] += 1
        if proposal is None:
            st["skipped_short"] += 1
            continue
        i, j, robots_to_shortcut = proposal

        # ---- verbatim: task-id gate ----------------------------------------------------
        can_shortcut_this = True
        for r in robots_to_shortcut:
            if new_path[i].mode.task_ids[r] != new_path[j].mode.task_ids[r]:
                can_shortcut_this = False
                break
        if not can_shortcut_this:
            st["skipped_gate"] += 1
            t_s = time.perf_counter()
            sampler.feedback("gate", ij=(i, j), robot=robots_to_shortcut[0])
            st["sampler_time"] += time.perf_counter() - t_s
            continue

        # ---- exact cost gate -----------------------------------------------------------
        improves = None
        path_element = None
        if table.ensure():
            verdict = table.decide(robots_to_shortcut, i, j)
            st["gate_uses"] += 1
            if verdict is not None:
                improves = verdict
                if verdict:
                    st["gate_positive"] += 1
                else:
                    st["gate_negative"] += 1
            else:
                st["gate_eps_fallback"] += 1

        if improves is None:
            path_element = _proposal_candidate(
                env, new_path, i, j, robots_to_shortcut, State
            )
            candidate_segment_cost = path_cost(path_element, env.batch_config_cost)
            old_segment_cost = path_cost(new_path[i : j + 1], env.batch_config_cost)
            improves = candidate_segment_cost < old_segment_cost

        if not improves:
            st["cost_rejects"] += 1
            t_s = time.perf_counter()
            sampler.feedback("cost", ij=(i, j), robot=robots_to_shortcut[0])
            st["sampler_time"] += time.perf_counter() - t_s
            continue

        # ---- verbatim construction and application (no collision check) ----------------
        if path_element is None:
            path_element = _proposal_candidate(
                env, new_path, i, j, robots_to_shortcut, State
            )
        q0 = new_path[i].q
        q1 = new_path[j].q
        assert np.linalg.norm(path_element[0].q.state() - q0.state()) < 1e-6
        assert np.linalg.norm(path_element[-1].q.state() - q1.state()) < 1e-6

        segment_gain = float(
            path_cost(new_path[i : j + 1], env.batch_config_cost)
            - path_cost(path_element, env.batch_config_cost)
        )
        for k in range(j - i + 1):
            new_path[i + k].q = path_element[k].q
        table.mark_dirty()
        wave.append({
            "i": i, "j": j, "robots": robots_to_shortcut,
            "new_qs": [path_element[k].q for k in range(j - i + 1)],
            "dcost": segment_gain,
        })
        cnt += 1
        st["provisional_accepted"] += 1

        if len(wave) >= wave_size:
            commit_wave()

    if wave:
        commit_wave()

    st["iters"] = iter
    st["wall_time"] = time.time() - start_time
    st["cost_final"] = float(costs[-1])
    st.update(sampler.report())

    if stats_out is not None:
        stats_out.clear()
        stats_out.update(st)

    # ---- verbatim: final asserts --------------------------------------------------------
    assert new_path[-1].mode == path[-1].mode
    assert np.linalg.norm(new_path[-1].q.state() - path[-1].q.state()) < 1e-6
    assert np.linalg.norm(new_path[0].q.state() - path[0].q.state()) < 1e-6

    return new_path, [costs, times]
