"""Observational instrumentation for shortcut replacement and collision-check reuse.

The shortcutter keeps a fixed number of path states and rewrites robot configurations on an
accepted interval.  This module labels the configurations that an accepted shortcut actually
changed.  A later accepted shortcut replaces those labels where it writes new configurations,
which lets us measure partial/full overwrite, shortcut lifetime, and the successful collision
checking effort spent on edits that do not survive to the final path.

It also hashes every cost-passing candidate presented to the collision checker.  Repeated hashes
are exact repeated collision queries under the same mode/configuration sequence; no inference is
made between merely nearby chords.  The class is deliberately observational and contains no
proposal or acceptance policy.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


def _counter_value(counter: Any, name: str) -> int:
    return int(getattr(counter, name, 0)) if counter is not None else 0


class ReplacementAudit:
    """Collect replacement and exact-repeat evidence for one shortcutting call."""

    def __init__(self, counter: Any = None, change_tol: float = 1e-12):
        self.counter = counter
        self.change_tol = float(change_tol)
        self.overhead_s = 0.0
        self._owners: Dict[tuple[int, int], int] = {}
        self._events: list[dict] = []
        self._checks: list[dict] = []
        self._seen_candidates: Dict[str, list[int]] = defaultdict(list)

    def snapshot(self) -> dict:
        """Cheap counter snapshot taken immediately around the native collision call."""
        return {
            "configs": _counter_value(self.counter, "configs"),
            "edges": _counter_value(self.counter, "edges"),
            "paths": _counter_value(self.counter, "paths"),
        }

    @staticmethod
    def _digest(path_element: Sequence, robots: Iterable[int], i: int, j: int) -> str:
        """Exact candidate identity: endpoints/robots plus every q and task-id mode."""
        h = hashlib.blake2b(digest_size=16)
        h.update(np.asarray([int(i), int(j), *map(int, robots)], dtype="<i8").tobytes())
        for state in path_element:
            q = np.ascontiguousarray(np.asarray(state.q.state(), dtype="<f8"))
            tids = np.ascontiguousarray(
                np.atleast_1d(state.mode.task_ids).astype("<i8", copy=False)
            )
            h.update(q.tobytes())
            h.update(tids.tobytes())
        return h.hexdigest()

    def describe_candidate(
        self,
        current_path: Sequence,
        path_element: Sequence,
        i: int,
        j: int,
        robots: Iterable[int],
    ) -> dict:
        """Capture candidate identity and the slots it would genuinely change.

        This must be called before the accepted candidate is copied into ``current_path``.
        Endpoints normally do not appear in ``changed_slots`` because shortcutting preserves them.
        """
        t0 = time.perf_counter()
        robots = tuple(int(r) for r in robots)
        changed = []
        for r in robots:
            for k in range(int(i), int(j) + 1):
                before = np.asarray(current_path[k].q[r], dtype=float)
                after = np.asarray(path_element[k - i].q[r], dtype=float)
                if float(np.linalg.norm(after - before)) > self.change_tol:
                    changed.append((r, k))
        out = {
            "digest": self._digest(path_element, robots, i, j),
            "changed_slots": changed,
            "robots": robots,
            "i": int(i),
            "j": int(j),
        }
        self.overhead_s += time.perf_counter() - t0
        return out

    def record_check(
        self,
        descriptor: dict,
        *,
        free: bool,
        proposed_gain: float,
        accepted_gain: float,
        collision_wall_s: float,
        before_counts: dict,
        after_counts: dict,
        elapsed_s: float,
        check_index: int,
        fail_k: Optional[int] = None,
    ) -> None:
        """Record one native path-collision call and, on success, update edit ownership."""
        t0 = time.perf_counter()
        digest = descriptor["digest"]
        previous = self._seen_candidates.get(digest, [])
        check_id = len(self._checks)
        record = {
            "check_id": check_id,
            "check_index": int(check_index),
            "i": descriptor["i"],
            "j": descriptor["j"],
            "span": int(descriptor["j"] - descriptor["i"]),
            "robots": list(descriptor["robots"]),
            "digest": digest,
            "free": bool(free),
            "proposed_gain": float(proposed_gain),
            "accepted_gain": float(accepted_gain),
            "collision_wall_s": float(collision_wall_s),
            "config_checks": int(after_counts["configs"] - before_counts["configs"]),
            "edge_checks": int(after_counts["edges"] - before_counts["edges"]),
            "path_checks": int(after_counts["paths"] - before_counts["paths"]),
            "elapsed_s": float(elapsed_s),
            "fail_k": None if fail_k is None else int(fail_k),
            "exact_repeat": bool(previous),
            "previous_exact_checks": list(previous),
        }
        self._checks.append(record)
        self._seen_candidates[digest].append(check_id)

        if free:
            event_id = len(self._events)
            changed = list(descriptor["changed_slots"])
            event = {
                "event_id": event_id,
                "check_id": check_id,
                "check_index": int(check_index),
                "i": descriptor["i"],
                "j": descriptor["j"],
                "span": int(descriptor["j"] - descriptor["i"]),
                "robots": list(descriptor["robots"]),
                "gain": float(accepted_gain),
                "accepted_elapsed_s": float(elapsed_s),
                "successful_check_wall_s": float(collision_wall_s),
                "successful_config_checks": record["config_checks"],
                "changed_states": len(changed),
                "retained_states": len(changed),
                "replaced_by": {},
                "fully_overwritten_at_event": None,
                "fully_overwritten_at_check": None,
                "fully_overwritten_elapsed_s": None,
            }
            self._events.append(event)

            # A later edit takes ownership only where it genuinely changes the current value.
            # If it uses an older shortcut's point as an endpoint, that point is unchanged and
            # remains owned by the older edit, correctly retaining that dependency.
            touched_previous: Dict[int, int] = defaultdict(int)
            for slot in changed:
                previous_owner = self._owners.get(slot)
                if previous_owner is not None:
                    touched_previous[previous_owner] += 1
                self._owners[slot] = event_id

            for previous_owner, count in touched_previous.items():
                old = self._events[previous_owner]
                old["retained_states"] -= int(count)
                key = str(event_id)
                old["replaced_by"][key] = old["replaced_by"].get(key, 0) + int(count)
                if old["retained_states"] == 0 and old["fully_overwritten_at_event"] is None:
                    old["fully_overwritten_at_event"] = event_id
                    old["fully_overwritten_at_check"] = int(check_index)
                    old["fully_overwritten_elapsed_s"] = float(elapsed_s)

        self.overhead_s += time.perf_counter() - t0

    def finalize(self, *, elapsed_s: float, check_index: int) -> dict:
        """Return JSON-serializable raw events and aggregate measurements."""
        t0 = time.perf_counter()
        for event in self._events:
            n = event["changed_states"]
            event["retained_fraction"] = float(event["retained_states"] / n) if n else 1.0
            event["fully_overwritten"] = event["fully_overwritten_at_event"] is not None
            stop_t = (event["fully_overwritten_elapsed_s"]
                      if event["fully_overwritten"] else float(elapsed_s))
            stop_c = (event["fully_overwritten_at_check"]
                      if event["fully_overwritten"] else int(check_index))
            event["lifetime_s"] = float(stop_t - event["accepted_elapsed_s"])
            event["lifetime_checks"] = int(stop_c - event["check_index"])
            # A transparent proxy for anytime contribution.  It is not claimed to be an exact
            # causal decomposition for a non-additive global cost.
            event["gain_time_exposure"] = float(event["gain"] * event["lifetime_s"])

        accepted = self._events
        overwritten = [e for e in accepted if e["fully_overwritten"]]
        touched = [e for e in accepted if e["retained_states"] < e["changed_states"]]
        total_gain = float(sum(e["gain"] for e in accepted))
        total_success_wall = float(sum(e["successful_check_wall_s"] for e in accepted))
        total_success_cfg = int(sum(e["successful_config_checks"] for e in accepted))
        repeats = [c for c in self._checks if c["exact_repeat"]]
        repeated_failures = [
            c for c in repeats
            if not c["free"] and any(not self._checks[p]["free"]
                                     for p in c["previous_exact_checks"])
        ]
        changed_total = int(sum(e["changed_states"] for e in accepted))
        retained_total = int(sum(e["retained_states"] for e in accepted))

        summary = {
            "checks": len(self._checks),
            "accepted_events": len(accepted),
            "partly_or_fully_overwritten_events": len(touched),
            "fully_overwritten_events": len(overwritten),
            "fully_overwritten_event_fraction": (
                float(len(overwritten) / len(accepted)) if accepted else 0.0
            ),
            "retained_changed_state_fraction": (
                float(retained_total / changed_total) if changed_total else 1.0
            ),
            "accepted_gain_sum": total_gain,
            "fully_overwritten_gain_sum": float(sum(e["gain"] for e in overwritten)),
            "fully_overwritten_gain_fraction": (
                float(sum(e["gain"] for e in overwritten) / total_gain)
                if total_gain > 0 else 0.0
            ),
            "successful_collision_wall_s": total_success_wall,
            "fully_overwritten_successful_collision_wall_s": float(
                sum(e["successful_check_wall_s"] for e in overwritten)
            ),
            "successful_config_checks": total_success_cfg,
            "fully_overwritten_successful_config_checks": int(
                sum(e["successful_config_checks"] for e in overwritten)
            ),
            "exact_repeated_collision_checks": len(repeats),
            "exact_repeated_failed_collision_checks": len(repeated_failures),
            "exact_repeat_collision_wall_s": float(sum(c["collision_wall_s"] for c in repeats)),
            "exact_repeat_config_checks": int(sum(c["config_checks"] for c in repeats)),
            "audit_overhead_s": float(self.overhead_s),
        }
        self.overhead_s += time.perf_counter() - t0
        summary["audit_overhead_s"] = float(self.overhead_s)
        return {"summary": summary, "checks": self._checks, "accepted": self._events}

