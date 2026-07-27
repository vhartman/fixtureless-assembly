"""END-TO-END SIM OUTCOMES -- the regression net the unit tests do not provide.

WHY THIS FILE EXISTS. On 2026-07-26 a bare `NameError` killed EVERY `sim.py run --v2`
invocation while all 126 unit tests stayed green, because nothing in the suite ever built a
SimScene. The same blind spot hid a `--mass-scale` StopIteration and a missing
`<compiler angle="radian"/>` that made MuJoCo read rai's joint ranges as degrees. All three were
found by running the sim by hand.

WHAT THESE ASSERT, AND WHY NOT MILLIMETRES. A final part-pose error is almost always a CASCADE:
one mate welds to a wrong datum and everything after inherits the displacement, so the headline
mm number describes the wreckage rather than the fault (measured: a 33.7mm first miss became a
1894mm final error). These tests assert the FIRST FAILURE's IDENTITY -- which action, which
failure kind -- plus the ok/total count. That is the thing that stays stable when the physics is
re-tuned, and the thing that actually changes when something breaks.

The scene/path artifacts are untracked working files; every test SKIPS if its pair is absent
rather than failing, so a fresh checkout stays green.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _have(*files):
  return all(os.path.exists(os.path.join(REPO, f)) for f in files)


def _run(capsys, argv):
  """Run a sim CLI invocation and return its printed output."""
  import sim as S
  cwd = os.getcwd()
  os.chdir(REPO)
  try:
    S._cli_run(S._build_cli().parse_args(argv))
  finally:
    os.chdir(cwd)
  return capsys.readouterr().out


def _summary(out):
  """(n_ok, n_total, first_failure_line or None) from a run's printed summary."""
  m = re.search(r"(\d+)/(\d+) actions ok", out)
  assert m, f"no summary line in output:\n{out[-2000:]}"
  first = None
  lines = out.splitlines()
  for i, ln in enumerate(lines):
    if "FIRST FAILURE" in ln:
      first = lines[i + 1].strip() if i + 1 < len(lines) else ""
      break
  return int(m.group(1)), int(m.group(2)), first


CAR = ["run", "car-mini-new.json", "--v2", "--mp-path", "mp-car-mini-new.json"]


@pytest.mark.skipif(not _have("car-mini-new.json", "mp-car-mini-new.json"),
                    reason="car-mini plan/path pair not present")
def test_car_mini_actuated_smooth_executes_cleanly(capsys):
  """The compliant-arm reference case. Catches anything that breaks --actuated end to end:
  the arm model, the servos, the velocity feedforward, the settle gate, the weld wrench."""
  out = _run(capsys, CAR + ["--actuated", "--smooth"])
  n_ok, n_tot, first = _summary(out)
  assert first is None, f"expected a clean run, first failure was: {first}"
  assert n_ok == n_tot, f"{n_ok}/{n_tot}"


@pytest.mark.skipif(not _have("car-mini-new.json", "mp-car-mini-new.json"),
                    reason="car-mini plan/path pair not present")
def test_car_mini_mocap_corridor_executes_cleanly_and_verifies(capsys):
  """The DEFAULT path plus corridor retiming. Also pins the corridor's own verification: with
  the legitimate-contact filter in place car-mini's schedule is genuinely collision-free, so a
  regression that reintroduces false positives (a gripper vs the part it HOLDS, a seated mate)
  shows up here rather than as mysterious refinement passes."""
  out = _run(capsys, CAR + ["--corridor", "--force-limits", "--inertia-factor", "1"])
  n_ok, n_tot, first = _summary(out)
  assert first is None, f"expected a clean run, first failure was: {first}"
  assert n_ok == n_tot, f"{n_ok}/{n_tot}"
  assert "certified collision-free" in out, "corridor verification did not run or did not pass"


@pytest.mark.skipif(not _have("car-mini-new.json", "mp-car-mini-new.json"),
                    reason="car-mini plan/path pair not present")
def test_a_mismatched_plan_and_path_pair_is_refused(capsys, tmp_path):
  """A saved path indexes tasks by INTEGER id, so replaying it against a different task
  structure relinks every attach at the wrong moment. The count-only guard could not see a
  same-count/different-semantics pair; the signature can."""
  import json
  states = json.load(open(os.path.join(REPO, "mp-car-mini-new.json")))
  bad = tmp_path / "mp-bad.json"
  json.dump({"states": states, "task_signature": "deadbeefdeadbeef"}, open(bad, "w"))
  out = _run(capsys, ["run", "car-mini-new.json", "--v2", "--mp-path", str(bad), "--smooth"])
  assert "TASK-SIGNATURE MISMATCH" in out
  assert "actions ok" not in out, "it executed anyway despite a mismatched pair"


@pytest.mark.skipif(not _have("cross8.json", "mp.json"),
                    reason="cross8 plan/path pair not present")
def test_cross8_warns_that_its_assembles_have_no_funnel(capsys):
  """cross8 was planned with pre_insertion=0, so every assemble drives the held part straight
  to a pose already in contact and it arrives LATERALLY -- 0.213mm of penetration while the
  target was still at rest, then ~800 steps of shoving and a weld fired 32mm off. The warning
  is what turns an hour of probing into a line of output, so it must keep firing."""
  out = _run(capsys, ["run", "cross8.json", "--v2", "--mp-path", "mp.json"])
  assert "NO INSERTION FUNNEL" in out
  m = re.search(r"NO INSERTION FUNNEL on (\d+)/(\d+) assemble", out)
  assert m and m.group(1) == m.group(2), "expected EVERY cross8 assemble to be unfunnelled"


def test_main_refuses_to_silently_do_nothing():
  """`main.py --motion-plan --save x` without --solve/--load-plan used to walk to the end and
  return None in 0.5s -- indistinguishable from a crash or a very fast solve."""
  import main
  cfg_args = ["--assembly", "cross", "--parts", "3", "--num-robots", "2",
              "--motion-plan", "--save", "/tmp/should-not-be-written.json", "--no-view"]
  with pytest.raises(SystemExit) as e:
    main.main(cfg_args)
  assert "need a plan" in str(e.value)
