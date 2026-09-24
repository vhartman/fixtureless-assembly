"""Physical-contact execution experiment on a saved assembly motion plan.

This reuses sim.py's tested MuJoCo model, arm tracker, and skill servos. Only the
physical support transition changes: a mate does not release a pinching hand.
The hand remains contact-driven and monitored until symbolic release or transfer.

Initial scope: all two-finger arms with an upfront physical grasp certificate,
2 ms physics, and actuated arms. The default uses the existing collision-audited
independent phase schedule; lockstep is only a diagnostic baseline.
Part--part mates remain capture-checked datum welds; insertions receive the saved
nominal process wrench. This is not full contact-only peg dynamics or an MPCC.

    python3 sim_contact.py run PLAN.json --mp-path MP.json --record REPLAY.npz

sim.py has no scene-factory hook, so this entry point replaces its scene class only
for one in-process run, restoring it in finally. sim.py itself is untouched.
"""

from __future__ import annotations

import sys

import mujoco

import sim
from plan_io import read_plan


def _desired_grasps(state):
  return sim._symbolic_grasps(state)


class ContactScene(sim.SimScene):
  """Preserve actual pad contact through a mate and release only on state change."""

  def _physical_grasps(self):
    out = set()
    for robot, grasp in self.active_grasps.items():
      if grasp[0] >= 0:
        raise RuntimeError(
            f"sim_contact requires physical two-finger contact; {robot} has a grasp weld")
      contacted = self._physical_contact_target.get(robot)
      if contacted is None:
        contacted = (self._hold_info.get(robot) or {}).get("grasp_part", grasp[1])
      out.add((robot, contacted))
    return out

  def _release_absent_grasps(self):
    if self.expect_state is None:
      raise RuntimeError("sim_contact requires persisted symbolic states at every attach")
    desired = _desired_grasps(self.expect_state)
    for robot, _part in sorted(self._physical_grasps() - desired):
      self.release(robot)

  def weld_mate(self, part):
    """Retain physical inserter contacts while installing the mate datum.

    Legacy weld_mate removes every grasp of part. A physical pinch has no grasp
    equality to remove: its pads already exert genuine MuJoCo forces. Save only
    those contact registrations across the existing mate operation. The usual
    sustained bilateral-contact monitor then detects a real loss of hold.
    """
    preserved = {}
    for robot, grasp in self.active_grasps.items():
      if grasp[1] == part and grasp[0] < 0:
        preserved[robot] = (
            grasp, self._physical_contact_target.get(robot),
            self._hold_info.get(robot), self._contact_grasp_misses.get(robot, 0))
    super().weld_mate(part)
    for robot, (grasp, target, info, misses) in preserved.items():
      self.active_grasps[robot] = grasp
      if target is not None:
        self._physical_contact_target[robot] = target
      if info is not None:
        self._hold_info[robot] = info
      self._contact_grasp_misses[robot] = misses
      for geom_id in self._holdex.get(robot, ()):
        self.model.geom_contype[geom_id] = 0
        self.model.geom_conaffinity[geom_id] = 0
    if preserved:
      mujoco.mj_forward(self.model, self.data)
      print(f"[sim_contact/contact] mate {part}: retained physical hand(s) "
            + ", ".join(sorted(preserved)))

  def attach(self, parent_frame, child, grasp_relpose=None,
             preserve_existing_grasps=False):
    if not self.contact_grasps:
      raise RuntimeError("sim_contact is a physical-contact experiment, not weld replay")
    if self.expect_state is None:
      raise RuntimeError("sim_contact requires persisted symbolic states at every attach")
    desired = _desired_grasps(self.expect_state)
    current = self._physical_grasps()
    if parent_frame in self.assembly.objects:
      # Legacy attach first detaches child, releasing its physical hand.
      self.weld_mate(child)
      self._release_absent_grasps()
      return True
    if not preserve_existing_grasps and not (desired - current):
      # A release/place may have a gripper parent only to re-root the planner tree.
      self._release_absent_grasps()
      return True
    # New pick or transactional handover: retain the bilateral contact check.
    ok = super().attach(parent_frame, child, grasp_relpose=grasp_relpose,
                        preserve_existing_grasps=preserve_existing_grasps)
    if ok:
      self._release_absent_grasps()
    return ok


def _check_scope(cfg, args):
  kinds = [x.strip() for x in (cfg.get("grippers") or "").split(",") if x.strip()]
  if not kinds:
    kinds = ["vacuum" if cfg.get("vacuum") else "two_finger"]
  if set(kinds) != {"two_finger"}:
    raise SystemExit("[sim_contact] initial physical-contact scope is all two-finger robots")
  if not cfg.get("grasp_physics_check", False):
    raise SystemExit("[sim_contact] plan must be solved with --grasp-physics-check")
  if not args.mp_path:
    raise SystemExit("[sim_contact] supply the saved motion plan with --mp-path")
  if args.execution_timing not in ("phase", "lockstep"):
    raise SystemExit("[sim_contact] choose phase execution or diagnostic lockstep")
  if not args.actuated or args.render_kinematic:
    raise SystemExit("[sim_contact] use actuated physics, not kinematic rendering")
  if abs(float(args.physics_step) - 0.002) > 1e-12:
    raise SystemExit("[sim_contact] calibrated pad contacts require --physics-step 0.002")


def main(argv=None):
  argv = list(sys.argv[1:] if argv is None else argv)
  if not argv or argv[0] != "run":
    raise SystemExit("usage: python3 sim_contact.py run PLAN.json --mp-path MP.json [sim flags]")
  if not any(x.startswith("--execution-timing") for x in argv):
    argv += ["--execution-timing", "phase"]
  # The physical-pad Cross calibration stayed attached at 0.5rad/s but slipped
  # at the legacy 1rad/s default. Keep one conservative contact-mode default;
  # a caller can still set an experimentally justified rate explicitly.
  if not any(x.startswith("--joint-rate") for x in argv):
    argv += ["--joint-rate", "0.5"]
  if not any(x in argv for x in ("--apply-process-wrench", "--no-apply-process-wrench")):
    argv.append("--apply-process-wrench")
  if "--no-contact-grasps" in argv:
    raise SystemExit("[sim_contact] --no-contact-grasps contradicts physical contact")
  # A failed physical acquisition or seat must not spawn a cascade of misleading
  # failures. Report the first broken action and stop the execution.
  if "--abort-on-action-failure" not in argv:
    argv.append("--abort-on-action-failure")
  argv.append("--contact-grasps")
  args = sim.parse_args(argv)
  cfg, _plan = read_plan(args.path)
  _check_scope(cfg, args)
  print(f"[sim_contact] physical two-finger execution; timing={args.execution_timing}, "
        "saved path only, nominal process wrench, first failure stops")
  old_scene = sim.SimScene
  try:
    sim.SimScene = ContactScene
    sim._cli_run(args)
  finally:
    sim.SimScene = old_scene


if __name__ == "__main__":
  main()
