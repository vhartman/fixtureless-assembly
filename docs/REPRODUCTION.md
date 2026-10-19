# Full reproduction and experiment record

This document contains the detailed working notes that previously lived in the root README.
For the project overview and repository map, see the [root README](../README.md).

This repository searches for multi-robot assembly plans, exports real-arm motion paths with
`mr_bench`, and validates the paths in MuJoCo. See [PIPELINE.md](PIPELINE.md) for the general
workflow and the other solver modes. The opt-in physical-contact execution
experiment and its current limitations are described in [SIM_CONTACT.md](SIM_CONTACT.md).
The new execution-program prototype is described in [SIMV2.md](SIMV2.md).

## Setup

Use Python 3.12 and the repository environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Motion planning also needs the `mr_bench` source tree. The default location is the sibling
checkout `../mr_bench/src`. Set `MRBENCH_SRC` if it is elsewhere:

```bash
export MRBENCH_SRC=/absolute/path/to/mr_bench/src
```

Use writable caches and a fixed Python hash seed for scripted runs:

```bash
export PYTHONHASHSEED=0
export NUMBA_CACHE_DIR=/tmp/assembly-numba-cache
export MPLCONFIGDIR=/tmp/assembly-mpl-cache
export ASSEMBLY_GRASP_RAW_CACHE_DIR=/tmp/assembly-grasp-raw-cache
mkdir -p experiments
```

The last directory stores only content-addressed, scene-independent surface proposals and
finite-pad contact scores. Scattered poses and collision/closure verdicts are never reused.
Deleting it is safe; the next sampled-grasp run rebuilds it.

Do not run two search or motion-planning jobs at the same time. Viewer flags are blocking, so
use `--no-view` for search and motion-planning scripts.

## Robust keyframes: current contract

The current defaults make the robustness settings explicit in every fresh search:

- `--witness-margin 0.005` is one hard 5 mm non-intended-pair gate used by the discrete
  search and by every later IK/fusion/funnel solve. A post-pass never applies a stricter gate
  than the search that admitted the action. The ordinary skill KOMO starts with its sparse
  collision model; if its nominal optimum fails this certificate, the exact violating
  frame-pair (or hardware-limit side) is added as an inequality and the same problem is
  warm-reoptimized. This lazy cutting-plane loop avoids both the old blind random-restart
  behaviour and an expensive dense all-pairs margin objective. Saved plans record
  `_search_witness_margin`;
  loading a plan with a larger requested margin is refused. Older plans without that marker
  are certified at 0 m and must be re-searched, or replayed from an exact `--mp-export-load`
  artifact without resolving IK.
- `--witness-sets 4` keeps up to four same-grasp IK branches. Pre-skill and post-release goals
  are conditional pairs: entering member A can only exit through member A, never jump to a
  different IK branch.
- `--object-over-table [--object-over-table-margin M]` is an optional workcell constraint for
  real IK. At every witness, the reference datum of each held rigid component moved by the
  current skill must lie inside the table frame's local XY footprint. Payloads on uninvolved
  arms are excluded because isolated IK parks those arms at an artificial EXIT pose. Positive
  `M` insets that footprint; negative
  `M` permits a bounded overhang, which is useful when the intent is only to exclude extreme
  outside-workcell solutions. The bounds use RAI's standard `1e1` workspace scaling and are
  checked exactly again on the accepted witness. Table-supported parts are unchanged. The flag
  constrains IK endpoints, not arm links, complete payload geometry, or the continuous
  motion-planning path, so physical workcell collision geometry remains useful.
- `--pre-pick 0.05 --post-pick 0.05 --pre-place 0.05 --pre-add-grasp 0.05
  --pre-insertion 0.05 --post-release 0.05` complete both sides of the contact skills. A pick
  now explicitly follows `pre-grasp -> attach -> held-object retreat`: its two standoffs use
  the same paired IK member, while the ordinary motion planner checks the departure after the
  payload becomes attached. This replaces the old execution-only lift whose temporary offset
  could decay back toward the table. An assembly mouth is a hard boundary: it must achieve the
  full configured `--pre-insertion` displacement on the same IK branch or that witness is
  rejected during search. It never shrinks toward the seated pose. Release departure tries a
  deterministic 100/80/60/40/20/10% distance ladder under the same gate; if none works, the
  exported zero-length fallback is reported explicitly.
- `--skill-funnel-collision-res 0.002` makes an assembly witness certify more than two clear
  endpoints: search samples the exact held-gripper interpolation from its pre-insertion mouth
  to its seated keyframe under the same collision gate. The accepted pair is persisted in the
  plan JSON. Motion export reuses it verbatim; a missing mouth is an artifact error, never a
  request to rerun IK.
- `add_grasp -> pre_assemble` shares one all-robot rendezvous keyframe, and adjacent release
  events reuse the assembly endpoint. This removes a pointless extra transport goal. It does
  **not** constrain BiRRT's path interior or run a controller in the planning loop; detours
  during insertion are allowed. `--mp-cograsp` is therefore a diagnostic of interior
  closed-chain drift, not a requirement that ordinary BiRRT project onto that manifold.
- `--optimize-witnesses floor`, `--wrench`, `--grasp-model asymmetric`,
  `--sig-collapsed-h`, and `--sig-components press` remain enabled for the runs below. The
  fresh search writes the floor-optimized witnesses and branch sets; a later `--load-plan`
  reuses them. `--fuse-cograsp` may still re-solve eligible paired keyframes before the final
  export.

### Named search profiles

Use `--search-profile standard` for the recommended fresh-search baseline. It expands to the
robust settings above plus the measured greedy local search, and makes the run
non-interactive:

```text
--weight 1 --milestones --milestone-oracle structural
--fill-local-h --fill-greedy --no-fill-exact --no-fill-policy-guide --no-fill-skeleton-first
--no-fill-action-widening
--milestone-boundaries 4 --fill-scope-widening --no-fill-touch-all
--center-scatter
--sig-collapsed-h --sig-components press --useful-grasps
--real-ik --ik-positive-reuse --wrench --grasp-model asymmetric
--optimize-witnesses floor
--witness-margin 0.005 --witness-joint-margin 0.02
--witness-branch-metric motion --witness-branch-budget 0 --witness-sets 4
--tight-fingers --strict-collisions 1e-6 --witness-robot-collisions --always-contact
--mate-eq-weight 100 --place-eq-weight 100 --part-float 0.002
--skill-funnel-collision-res 0.002
--grasp-sampler
--grasp-sampler-keep 20 --grasp-sampler-candidates 96 --grasp-sampler-scenes
--grasp-sampler-roll-refine
--component-grasp-keep 32
--grasp-sampler-wrap 0.015
--grasp-sampler-strict  # optional: make margin/pad-support hard gates
--no-grasp-refine
--grasp-widen --grasp-widen-batch 8 --grasp-widen-max 96
--placement-yaws 4 --placement-yaw-widening
--open-witness --open-witness-batch 7 --open-witness-epochs 2
--lazy-edge-selector forward
--quiet-rai --no-view
```

The profile also pins analytic IK seeding, lazy milestone fills, bracing, handovers, placement,
the hands-off goal, and release hoisting. These are not decorative defaults: they respectively
avoid repeated stochastic IK work, retain the scalable fill path, preserve wrench-feasible
support actions, keep regrasp routes available, require a physically completed result, and free
arms before later motion. `--witness-robot-collisions` and `--always-contact` ensure each search
scene includes parked inactive-arm geometry and all contact geometry. Search witnesses remain
per-edge certificates. The short, semantically prescribed insertion continuation is now part of
the IK certificate; arbitrary free-space transit between skills remains the motion planner's
responsibility and may detour.

The witness post-pass keeps the existing normalized clearance-floor formulation. It first finds
the minimum number of large bounded-joint transitions available with no additional clearance
floor, then selects the highest tested per-step clearance fraction that does not add any such
transition (`--witness-branch-budget 0`). “Motion” counts the joint displacement the planner must
actually traverse; it does not quotient a bounded joint by 2 pi or assume that joint 6 is a
harmless UR wrist roll. This prevents *discretionary* branch changes but does not claim that every
large transition is removable. For redundant arms, deterministic Jacobian-nullspace seeds are
projected back onto the exact tool pose by the same collision-aware KOMO and passed through the
ordinary search clearance/joint-margin certificates before joining the dynamic-programming
candidate set. The search's 5 mm gate is unchanged; no stronger absolute post-search clearance
gate is introduced.
Standard lazy symbolic candidates are validated in execution order. Four non-pruning selector
ablations are available. `terminal-first` checks the candidate's last unresolved `assemble`
endpoint and then resumes execution order; `reverse` works backward from the terminal skill;
`fail-first` checks an exact skill-local physical query with prior failed effort before falling
back to reverse. `defer-failed` retains forward validation, but after a query fails it filters
every symbolic alias of that same physical query for the rest of the current fairness epoch.
The next epoch clears the filter and advances the shared effort cursor, so a finite miss never
becomes a permanent negative verdict. None of the selectors changes the finite action graph.
On the 4-robot cube scaling cases, this removed the repeated-alias pathology: matched 4- and
10-part searches fell from 141.12 to 15.88 seconds and from 740.02 to 258.25 seconds,
respectively (one seed; witness optimization disabled in both rows). A matched cross-scene
follow-up showed why it remains an explicit ablation: OpenArm Cross improved 90.91 -> 75.57 s,
OpenArm Car Mini was wall-time neutral (25.60 -> 25.44 s), and mixed-gripper Chair regressed
32.25 -> 65.14 s. In Chair's final module, filtering the failed family redirected search from a
promising retry into a larger symbolic frontier (1,289 -> 4,383 expansions and 14 -> 48 binds).
Standard therefore retains forward validation until physical-query reuse can be a soft
scheduling preference rather than an all-or-nothing per-epoch exclusion. On UR5 Car Mini,
fail-first reduced
binds 386 -> 293 but
was wall-time neutral (78.76 vs 79.23 s). On OpenArm Car Mini it made costly mate bundles
dominate: obj_1 exhausted two node-budget epochs after 5,537 binds/462 s of IK, versus the
forward run's 3,490 binds/214 s of IK and successful 287 s fill. Standard therefore remains
`forward`; alternative selectors are research ablations until selection accounts for check cost.

`--open-witness-positive-order` is a separate positive-only scheduling experiment. It stably
prefers exact certified edges, previously successful compatible co-grasps, and grasps already
reached by the same robot, but retains every successor and never treats a finite IK failure as a
no-good. It is intentionally not part of `standard`: on matched plugged-in runs it improved
UR5 Cross-7 (5.71 to 4.00 s search, 51 to 41 IK solves), but regressed mixed-gripper Chair-8
(16.33 to 21.37 s, 37 to 43 IK solves). Positive evidence alone is therefore not a reliable
global priority across different local fills.

The standard profile also enables the physical IK-query invariant
(`--ik-positive-reuse`; use `--no-ik-positive-reuse` for the matched ablation). A canonical key
contains only the physical slice used by that skill: picks use the acting robot/grasp and fixed
table scene; handovers use the participating robot/grasp assignments on their payload;
table-supported `add_grasp` also uses the fixed table scene; assemble uses the moving and target
payloads plus the table scene for a table-anchored mate; place uses its held payload and actual
destination table scene. Symbolically different histories that build the same skill program
share the successful joint certificate directly and continue the same open-witness effort cursor.
A revisit skips deterministic seeded/unseeded NLP attempts and advances to fresh structured-bias
or random slots. It currently still rebuilds the scene/KOMO, analytic seed enumeration and branch
probe, and relaxed reach precheck; those are not yet memoized. Under the standard open-witness
scheduler, a failed stochastic query remains open rather than turning into an infeasibility
verdict. With the standard `forward` selector, a symbolic alias may revisit it immediately but
starts at the next fresh effort slot. With `defer-failed`, aliases wait until the next fairness
epoch. Only a successful certificate eliminates the IK call entirely. The `[ik-reuse]` report
distinguishes candidate-touched queries, actually probed queries, retained positive certificates,
and effort-cursor revisits.

The experimental `--fill-policy-guide` is deliberately separate from those selectors. It is
available only to the ordinary `--milestones` solver, requires `--fill-local-h`, and needs a
finite fibre to schedule: `--grasp-local-widen` for raw grasp columns and/or
`--placement-yaw-widening` for table-yaw orbits. The separate whole-search `--grasp-widen`
restart is not its raw-column interface. The guide leaves the compact local fill and its normal
lazy-validation schedule intact. Only after that compact pass and, when local columns are active,
the one historical numeric `K -> K+B` incoming-grasp expansion have failed, it queries the
reverse PDB over the relevant finite master.

The PDB contributes only an ordering. The guide traverses the settled Bellman-optimal policy DAG
breadth-first, subject to a finite optional traversal guard, rather than committing to one
tie-broken relaxed path.
Its atomic demands are raw grasp-signature fibres and stable-rest yaw orbits. A grasp fibre can
be a `loose` label for any relevant free part or a `support` label on the settling component;
`support` is neutral about whether the concrete contact is inherited, acquired by `add_grasp`,
or reached through a handover. Each distinct fibre receives at most one bounded,
stable-master-ID batch in the rescue pass. If its forward probe fails, those columns remain
active while the next Bellman branch receives effort. A yaw atom exposes the complete configured
finite world-z orbit of one stable rest class; yaw probes likewise retain earlier policy-selected
raw columns and rest classes. This monotone accumulation can represent a relaxed route that
requires more than one missing coordinate without eagerly opening the entire Cartesian product.
Search then returns to its compact yaw prefix; a yielded candidate temporarily publishes the
full finite yaw table so immediate replay, witness collection, and `--mp-open-plans` can resolve
a policy-selected placement id.

The reverse computation performs no IK or KOMO and no policy-specific analytic/SSIK ranking or
gate. Raw members are considered in stable master-ID order; all ordinary collision, closure, and
IK checks remain in the existing lazy forward validator. Each newly exposed batch/orbit receives
one bounded forward probe (one open-witness epoch when that mechanism is enabled). A failed probe
advances to another distinct Bellman demand; after the finite frontier is exhausted, the ordinary
progressive schedule resumes. Later members of a large raw fibre remain the responsibility of
ordinary numerical and final finite-master widening, so the guide cannot drain a 96-grasp column
before considering another branch.

This refinement removes arbitrary tie-breaking, but deliberately does not introduce a focal
slack such as “abstract cost +5.” If every Bellman-optimal abstract route is concretely spurious
and the realizable manipulation has larger relaxed action cost, the guide exhausts its frontier
and falls through to generic widening. Feeding scene-specific IK counterexamples back into the
abstract graph would be a separate counterexample-guided refinement step; finite stochastic IK
failure is not currently promoted to a permanent abstract no-good.

If the probe fails, generic grasp/yaw/scope widening, milestone repair, and the later eager
finite-master session remain authoritative. Thus the guide is additive finite effort scheduling,
not pruning or a concrete-feasibility certificate. It preserves coverage of the configured finite
action alphabet only subject to the existing node, repair, endpoint-boundary/window, deadline,
and finite witness/IK budgets; it is neither a global-completeness nor an infeasibility claim.
The standard profile deliberately keeps `--fill-policy-guide` off until this optional ordering
has broader benchmark evidence.

A matched guarded run on OpenArm Car Mini-5, OpenArm Cross-7, and mixed-tool Chair-8 confirmed
the intended non-interference seam: all six local-column runs succeeded, and enabling the guide
changed neither plan, cost, IK call/failure count, nor peak memory. None of those compact fills
failed, so the policy frontier was never queried; their wall-time differences are continuous
solver/system variance and are not evidence of a speedup. The focused regression suite therefore
includes forced compact misses: it verifies alternative Bellman-branch advancement, monotone
column retention, combined grasp-plus-yaw demands, replay-safe cap publication, and fall-through
to ordinary finite-master recovery. The option remains an ablation until a natural benchmark
regularly reaches this post-failure regime.

`terminal-first` is the minimal concrete-disassembly experiment: it reuses the exact symbolic
pre-assembly scene already present in a complete candidate, rather than introducing a second
inverse action model or a terminal-mode score. Successful joint vectors enter the ordinary exact
scene witness pool; failure refutes only that candidate edge for the current fairness epoch. A
2026-08-28 matched UR5-limited Cross-7 run completed with both selectors. Terminal-first reduced
IK binds/failures from 1121/828 to 935/656, but search time increased from 93.5 s to 112.9 s after
repair selected a different realization. A mixed-tool UR5-limited Chair-8 run also completed
(35 actions, 187.2 s, 1.99 GB peak). On the difficult OpenArm Car-mini run it greatly improved
the early fills but the final fill exhausted even four witness epochs and its subsequent search
hit the 4 GiB address-space guard. It is therefore deliberately opt-in:

```bash
--lazy-edge-selector terminal-first
```

A bounded focal-queue prototype was also rejected rather than retained as another tuning flag:
selecting minimum local-PDB `h` within `1.5 f_min` expanded the Cross fill to the node/memory
guard before performing any IK. The reverse PDB is useful as the admissible anchor, but its large
equal-progress plateaus are not a useful focal secondary objective.
The same 5 mm witness margin keeps carried assemblies above the table at free-pose
`handover`/`add_grasp` keyframes. Pick, place, and constrained assembly contacts are unchanged;
this is enforced during search and all witness passes, not introduced only at export.
At an `assemble` endpoint, the acting hand versus the action's rigid mate-target subassembly uses
the strict zero-penetration gate rather than the 5 mm free-space margin: close-fitting mates can
necessarily put a finger within millimetres of the named target or a part already attached to it.
This exemption does not cover arm links, another robot, clutter, or unrelated parts.
Likewise, hands participating in the same `handover`, `add_grasp`, or `assemble` rendezvous use
strict collision without the 5 mm inter-robot buffer at the endpoint. Bystander robots retain
the buffer. This matches motion planning's existing task-own-robots margin ramp rather than
giving search and planning incompatible endpoint definitions.
The 0.02 rad joint margin serves the same purpose at hardware stops for coupled
`handover`/`add_grasp`/`assemble` endpoints: such an endpoint must have an open joint-space
neighbourhood for sampling-based motion planning. Ordinary single-arm picks and places are not
filtered by it. It does not enlarge the robot limits; it rejects boundary branches and lets
open-witness/search try another realization. Arm-less `--floating-ee` coordinates are merely a
six-DOF pose parameterization and are therefore exempt from this hardware-limit gate. The test
only covers robots participating in the coupled endpoint; an uninvolved arm resting at a home
limit cannot veto another pair's task.
`--wrench` also makes every backward heuristic wrench-aware, including per-tool facts for a
mixed vacuum/two-finger fleet. There is no separate mixed-wrench opt-in; use
`--no-wrench-aware-h` only for an explicit heuristic ablation.

Thus the ordinary command is now:

```bash
python3 main.py --solve --search-profile standard \
    --assembly ASSEMBLY --parts N --num-robots R --robot ROBOT \
    --seed SEED --save experiments/PLAN.json
```

#### Calibrated Husky duo

`--robot husky` loads the calibrated Husky carrier and exposes its embedded right/left UR5e
arms as the ordinary `a1`/`a2` robots. Assemblies are then created by the same scenario path as
every other robot; there is no assembly-specific Husky `.g` file. The current model has two
fixed Robotiq grippers, so use exactly two robots and do not request vacuum or `--floating-ee`.
For example, inspect the standard peg-in-hole initial scene with:

```bash
python3 main.py --assembly peg-hole --parts 2 \
    --robot husky --num-robots 2 --show-init
```

The table's working surface is 0.44 m above the floor. The table, calibrated carrier, both arm
subtrees, collision proxies and grippers are retained
when the scene is reconstructed or exported to MuJoCo. SSIK uses two independently generated
artifacts (`husky_right` and `husky_left`) because the calibrated chains differ slightly. Both
use an identity RAI-joint map and the measured tool0-to-Robotiq-centre transform: a +90-degree
roll about tool0 followed by the 130mm centre offset. The ordinary `ur5`/`ur5_limited`
calibration is not modified or aliased.

Rebuild the two SSIK modules from the current calibrated scene after changing either arm,
gripper centre, or joint convention:

```bash
python3 tools/calibration/calibrate_husky_ssik.py /tmp/husky-ssik
.venv/bin/ssik build /tmp/husky-ssik/husky_right.mjcf \
    --base a1_ur_base_link --ee a1_ur_tool0 \
    --out ik_artifacts/husky_right_ik.py --validate-poses 100
.venv/bin/ssik build /tmp/husky-ssik/husky_left.mjcf \
    --base a2_ur_base_link --ee a2_ur_tool0 \
    --out ik_artifacts/husky_left_ik.py --validate-poses 100
```

The verified end-to-end peg-in-hole recipe is:

```bash
python3 main.py --solve --search-profile standard \
    --motion-plan --motion-profile standard \
    --assembly peg-hole --parts 2 --num-robots 2 --robot husky \
    --seed 11023 --mp-seed 11023 \
    --save experiments/husky-peg-hole-2r-standard-plan.json \
    --mp-save experiments/husky-peg-hole-2r-standard-mp.json

python3 sim.py run experiments/husky-peg-hole-2r-standard-plan.json \
    --mp-path experiments/husky-peg-hole-2r-standard-mp.json \
    --record experiments/husky-peg-hole-2r-standard.npz \
    --report-json experiments/husky-peg-hole-2r-standard-report.json
```

The standard motion profile intentionally spends its complete 300-second RRT* budget after
finding the first feasible path. Use an explicit shorter `--mp-runtime` only for a smoke test;
the simulator still consumes the exact saved path and never re-solves IK.

For local-fill performance diagnostics, prefix the same command with
`FILL_SEARCH_PROFILE=1`. Each `[open-witness ...]` line then reports cumulative forward-search
time and counts for successor generation (`succ`), heuristic cache misses/requests (`h`), the
cheap feasibility gate, queue/alternate bookkeeping, and deleted-edge repair. The timers are
opt-in because per-edge timing is measurable on million-edge fills; compare operation counts or
matched profiled runs, not a profiled wall time against an ordinary run.

#### OpenArm v2 duo

`--robot openarm` uses the native seven-axis OpenArm model and pinch hand. Two robot slots are
one physical torso/duo (`a1,a2`); four slots create two opposing duos (`a1,...,a4`). The model
keeps the exported hardware joint limits. Its workcell is instead calibrated for the shorter
arm: the two torso rows sit 10 cm closer to the workbench than the FR3 rows.

OpenArm deliberately has distinct **HOME** and **EXIT** joint configurations. HOME is the
user-facing pose (`joint1=0` on both sides): the physical trajectory starts there and the
terminal task returns there. EXIT is the tucked manipulation parking pose (`joint1=-3`): every
uninvolved arm is collision-checked there while skill IK is solved. Motion export inserts no
forced `exit` task in the physical motion graph. Search/IK parks otherwise-idle arms at EXIT,
while execution starts at HOME and collision-checks a direct connection from each robot to its
first required task; the terminal task returns directly to HOME. This avoids a global four-arm
rendezvous and the corresponding 3 rad OpenArm detour without weakening motion validation.
Full pre-keyframe certificates likewise use EXIT only as an IK background: the BiRRT hint copies
the active task slices onto the current mode entry and never inserts idle EXIT slices into the
path. The witness continuity objective starts at HOME, follows consecutive active appearances,
and includes the return to HOME.
UR/FR3/floating scenes currently use HOME=EXIT and retain their previous behavior exactly.

OpenArm grasp filtering uses `src/rai/openarm/openarm_pinch_floating.g`, not the generic
Robotiq floating gripper. Its tool centre, approach axis, palm, and open-finger sweep match the
real OpenArm hand. In `left_openarm.g` and `right_openarm.g`, the single-`#` collision candidates
are active and calibrated; the deliberately disabled `##joint4` and `##joint6` guesses remain
disabled. Analytic IK is side-specific and seven-DOF. The runtime RAI limits are transformed
through the calibrated joint-sign map and passed to SSIK's feasible-swivel interval resolver;
the generated solver's original URDF wrist limits are not used. This matters because a 7R target
has a continuous one-dimensional self-motion manifold: filtering only SSIK's default 16 swivel
samples caused false empty results even for poses generated by valid in-limit configurations.
Interval representatives seed KOMO and are cached by physical target and hardware box. They are
not treated as an exhaustive finite collision proof, and seeded 7R solves retain their normal
KOMO restart budget.

With that correction, the formerly pathological centred-scatter Cross-7 search completes all
seven parts under the production limits in 13.45 s (23 actions, 158 IK binds, 292 MiB peak RSS,
seed 11023, geometry-only diagnostic, witness post-pass excluded). Cached SSIK interval work is
0.98 s of the 8.7 s IK bind wall. Dense inspection also explains the remaining rejected
grasps: some exact target poses put the gripper 90--124 mm through the table; another has at
least 34.5 mm wrist/torso and 19.2 mm gripper-base/torso penetration over 512 sampled swivel
values. Seven joints provide one redundant scalar, not freedom to repair a tool pose that itself
intersects fixed workcell geometry.

The OpenArm planning bounds for the final wrist axes are joint 6 in `[-1.8, 1.8]` rad and
joint 7 in `[-1.85, 1.85]` rad. The original calibrated bounds (`+/-0.7854` and
`+/-1.5708`) lose every SSIK branch for 8 of 50 demonstrated Car Mini tool poses. A fresh
standard search and optimized 44-task motion plan succeeded under the stricter
`+/-1.8`/`+/-1.8` test model. The production joint-7 bound adds 0.05 rad because that path
touched -1.8 rad; its joint-6 margin is 0.088 rad. This replaces the temporary `+/-pi`
workaround with bounds established by re-solving rather than by clipping an old trajectory.

Table placement yaw is available as a separate, geometry-principled reachability experiment:

```text
--placement-yaws N
```

`N=1` is the historical action space and remains the bare-CLI compatibility default. The
`standard` search profile now selects `N=4` together with `--placement-yaw-widening`. For
`N>1`, each stable rest pose is augmented by the uniform world-z rotations `2 pi k/N`. This is
the free `S1` fibre of an
object resting on a horizontal table: support stability, table collision, gravity, and the
object-frame wrench problem are unchanged, while fixed-base arm reachability can change. The
old placement ids remain the first candidates; additional rotations receive deterministic
quaternion-derived ids and are stored in `State.place`, saved in the plan config, and restored
by replay. Thus two different yaws cannot alias under the state hash, and an old plan missing
the setting reconstructs with `N=1`.

The backward signature/PDB models quotient these yaw copies back to their common rest class.
That quotient is exact for the facts represented there and avoids multiplying the abstract
database by `N`; concrete search and real IK still see every rotation. A focused test with the
normal OpenArm workcell and the original joint-6/joint-7 hardware limits found three held Cross
prefixes for which the upright set-down had no fixed-arm IK but a yaw copy did (one `a2` grasp
succeeded only at 270 degrees). So the degree of freedom can recover real reachability without
moving the table or robot bases.

Eager enumeration remains available by selecting `--no-placement-yaw-widening`; it is useful as
an ablation but is not the standard. A matched original-wrist OpenArm Cross-7 run gives its
trade-off: `N=1` found 30
actions in 125.10 s (596 MiB peak), `N=2` found 29 actions in 136.86 s and used a new 180-degree
placement, while `N=4` found 31 actions in 291.44 s (927 MiB peak) and also used a non-legacy
placement. All three passed the standard wrench, 5 mm witness margin, strict collision, witness
set, and clearance-floor pipeline; the yaw-quotiented PDB remained small.

The production scheduler instead uses finite nested action spaces. Every fill starts with the
historical yaw. If all cheap local unbraced or PDB-certified-brace phases fail, it exposes the
180-degree copy and then both quarter turns. Only after the complete local yaw lattice fails does
it enable unrestricted bracing and all-part manipulation. Placement IDs and successful IK
witnesses remain stable across levels, and the last pass contains exactly the eager `N=4` action
space; this changes evaluation order, not the finite problem being solved. A matched one-seed
suite (post-search optimization disabled in both columns) measured:

| scene | historical N=1 | widened N=4 | actions / cost | peak RSS N=1 / N=4 |
|---|---:|---:|---:|---:|
| OpenArm Car Mini-5 | 24.76 s | 24.82 s | 21 / 26 | 336 / 338 MiB |
| OpenArm Cross-7 | 142.49 s | 142.44 s | 30 / 40 | 590 / 587 MiB |
| mixed-tool Chair-8 | 117.27 s | 117.54 s | 34 / 43 | 1683 / 1681 MiB |
| mixed-tool Peg Stool-5 | 38.70 s | 38.71 s | 19 / 24 | 945 / 944 MiB |

Thus unused rotations are effectively free on the current suite. They are not a universal cure:
with the original narrow OpenArm wrists, Car Mini wheel 2 exhausted the 1-, 2-, and 4-yaw cheap
local families before entering the same unrestricted-bracing wall. That experiment diagnoses a
reach/holding bottleneck rather than a missing table orientation.

The design is a finite manipulation-specific instance of **iterative broadening**, with the
parameter family rather than an arbitrary child count defining each nested graph. It is also
close to focused TAMP methods that keep continuous values outside the symbolic graph and expose
them only when a candidate plan needs refinement. Relevant precedents are Ginsberg and Harvey's
[iterative broadening](https://doi.org/10.1016/0004-3702(92)90059-7), Garrett et al.'s
[conditional-sampler formulation](https://www.roboticsproceedings.org/rss13/p39.html) and
[PDDLStream](https://arxiv.org/abs/1802.08705), and Lertkultanon and Pham's
[high-level grasp-placement table](https://arxiv.org/abs/1509.00600). The next principled step,
if finite yaw widening still spends too much time on failed fills, is a focused placement stream:
search over the rest/yaw **class**, instantiate only a small nested batch for the selected
skeleton, and fairly alternate new values with other symbolic skeletons. It is not to add more
eager pose actions or assembly-specific ordering rules.

`--robot openarm --floating-ee` uses that same native pinch model in the compact OpenArm
workcell, with one free six-DOF chain per requested robot and no arm links, reach limits, or
SSIK. It does **not** fall back to the generic Robotiq hand. This is the geometry-preserving
ablation for separating grasp/search feasibility from fixed-arm IK; saved floating plans rebuild
the same model in replay.

The validated seven-part Cross command is the standard search with three OpenArm-specific scene
and candidate-set overrides:

```bash
python3 main.py --solve --search-profile standard \
    --assembly cross --parts 7 --num-robots 4 --robot openarm \
    --grid-init 4 --grid-gap 2.0 \
    --no-grasp-refine --grasp-sampler-keep 16 \
    --seed 11023 \
    --save experiments/openarm-cross-4r-standard-ee180.json
```

This command retains the central grid as a relocation-stress layout. The standard ordinary
scatter is now a compact centred lattice at the requested spacing rather than the historical
x=-0.5 one-row line, so it no longer places opposite parts outside OpenArm's workspace.
`--no-grasp-refine` exposes the complete hybrid grasp set immediately: the
generic conventional prefix alone lacks off-centre transfer grasps needed by the shorter arm,
so a conventional-only epoch is mostly wasted for this scene. Keeping 16 candidates retained
the orientation diversity needed by the Cross while bounding the sampled action space. The
corrected `gripper_center` uses the common approach convention through a 180-degree tool-frame
rotation; the side-specific SSIK calibration includes that same transform. The floating pinch
proxy mirrors the three active EE capsules plus the open-finger collision union, and a focused
test now fails if either its tool frame or those capsules drift from the real hand.

The fingers remain rigid/open in the search and IK models. The standard profile therefore uses
the independent closing certificate (`--grasp-closure-check`) by default: for each two-finger
pick, handover, or `add_grasp`, a
throw-away collision scene virtually sweeps the receiving finger capsules to a common,
object-dependent aperture. A grasp is rejected if the jaws cannot contact both sides or the sweep
crosses another gripper, part, arm, or the table; vacuum grasps are unchanged. With the
scene-reserved sampler, the same per-kind scene/closure certificate labels the initial `pick`,
the actual mandatory-predecessor `assemble` endpoint, and support grasps for every intermediate
rigid component created at an authored assembly boundary (including the final product). This is
important because a grasp that clears a partial assembly may collide with a part added later;
singleton and final-product labels alone can therefore discard the useful transport grasp. The
bounded database reserves these roles before the hybrid tail. Pairwise handover compatibility
remains a lazy action check rather than a quadratic precomputed table. Failed candidates remain
as fallback instead of being silently deleted.

Every newly saved plan also carries a versioned contract over its contact-bearing RAI geometry,
joint order/limits, sampled-grasp database, and closure implementation. Every two-finger grasp
edge stores its certified object-dependent aperture alongside the action and witness. Replay and
simulation fail closed on a missing/mismatched contract; `--refresh-witnesses` is the explicit
legacy recertification route. Visual-only mesh/color edits do not invalidate the contract.
New real-IK plans also store the exact symbolic state after each action and only the grasp
transforms actually selected by the plan. Loading one under matching settings for motion
planning or simulation reuses
those records, so it does not regenerate the full candidate grasp database. Older plans still
use the original reconstruction path; the collision-model contract remains enforced.

Closure-aware sampling also **generates valid grasps instead of relaxing the gate**. Antipodal
point contacts on a CAD mesh need not centre the gripper's finite pads: a flange may make one jaw
touch first. The closure solve measures both first-contact travels and translates the tool along
its closing axis by half their difference; the translated pose is retained only after the full
bilateral close is re-certified. Likewise, if an otherwise valid source has a finger/table
overlap, the sampler retracts along `-tool-z` by the minimum distance needed for 0.2 mm positive
clearance (bounded by half the pad length). These are new persisted candidates, tagged
`jaw-recenter` and `approach-retract`. The original remains a free-space fallback, while clutter
and other-hand collisions are never projected. On the flat 12 mm Car Mini wheels, the OpenArm
candidates require about 1.8 mm retraction; their approximately 20 mm per-finger closing travel
is unchanged.

Inspect one certificate without running search using the scrubbable closure viewer:

```bash
python3 main.py --search-profile standard \
    --assembly car-mini --parts 5 --num-robots 4 --robot openarm \
    --grid-init 4 --grid-gap 2.0 --grasp-sampler-keep 16 \
    --grasp-closure-check --viz-grasp-closure a1:obj_3:0 \
    --viser --viser-port 8080
```

Open `http://localhost:8080` and scrub the 21 samples from fixed-open to the inferred common
aperture. Each label reports per-finger signed distance to the target and names the first
non-target penetrating pair when the sweep is blocked. Replace `a1:obj_3:0` with
`ROBOT:PART:GRASP`. The command uses the same floating scene and certificate as the action gate;
it changes neither the search nor the saved model.

To inspect every grasp that passes the isolated floating-hand collision gate and the
object-dependent virtual jaw closure, use:

```bash
python3 main.py --search-profile standard \
    --assembly car-mini --parts 5 --num-robots 4 --robot openarm \
    --grid-init 4 --grid-gap 2.0 --no-grasp-refine --grasp-sampler-keep 16 \
    --seed 11 --viz-initial-grasps --viser --viser-port 8080
```

The target remains at its actual initial table pose, but every other part is parked outside the
workcell: the catalog therefore measures intrinsic pickability rather than removable clutter.
The timeline contains one accepted grasp per step with the fingers at their inferred common
closed aperture. Labels report the part and grasp id,
candidate family, any generated approach retraction, closure travel, closest collision pair,
and the cheap gate's 5 mm penetration tolerance. A displayed acceptance has additionally passed
the strict closure certificate; the 5 mm number is not used to forgive its open finger/table
geometry. SSIK, KOMO, and wrench feasibility are not run. Add a part name, for example
`--viz-initial-grasps obj_0`, to restrict the catalog. Use `--viz-grasp-closure` separately to
scrub all 21 samples of one candidate's closing sweep. The catalog automatically enables
closure-aware scene ranking while building the sampled database, so its grasp IDs match a
search run with `--grasp-closure-check` even when that flag is omitted from the viewer command.

### Stable-placement grid viewer

The placement catalog can be inspected without running search. The grid viewer emits one Viser
step per singleton, connect-module, direct-base group, and full assembly; within a step, every
accepted placement is shown next to the others on the table. The marker and GUI label identify
the exact placement id stored in search states and saved plans.

For the full Chair using geometry-derived stable orientations:

```bash
python3 main.py --assembly chair --parts 8 \
  --stable-rest-poses --placement-yaws 1 \
  --viz-rest-poses-grid --viser --viser-port 8080 --no-view
```

Open `http://localhost:8080` and use the step slider to move between parts and subassemblies.
Use `--placement-yaws 4` to inspect the complete four-yaw placement alphabet used by the
standard search profile. The older `--viz-rest-poses` remains available when one orientation
per timeline step is preferable. Both viewers call the same `rest_faces` and placement-pose
functions as search; they do not infer or repair poses for display.

Every fresh `--solve` now runs the corresponding isolated singleton check before symbolic search
or IK. If a part has no isolated pick, the run prints per-kind tested/open-clear/closure counts
and aborts with `[grasp-preflight] ABORT`; widening or decluttering cannot repair an intrinsic
failure. Closure participates in this preflight when `--grasp-closure-check` is enabled.

Real-robot model onboarding also fails early on two structural contracts that previously
appeared only as large IK searches. The manipulation EXIT vector must be finite, dimensionally
consistent, and at least 1e-3 inside every hardware limit; OpenArm must declare an EXIT distinct
from HOME and its collective EXIT scene must be collision-free. Hand collision geometry is
derived from the configured tool centre's kinematic subtree, not from names such as `palm`,
`finger`, or `ee_coll`. A hand subtree with no active collision shape is a model error. Thus a
new gripper naming convention cannot silently lose the intended-contact margin exemptions, and
removing OpenArm's workspace-clearing EXIT fails at startup rather than during assembly fill.

The current five-part OpenArm Car Mini command is:

```bash
python3 main.py --solve --search-profile standard \
    --assembly car-mini --parts 5 --num-robots 4 --robot openarm \
    --grid-init 4 --grid-gap 2.0 \
    --no-grasp-refine --grasp-sampler-keep 16 \
    --grasp-closure-check --seed 11 \
    --save experiments/openarm-car-mini-4r-standard-clearance-plan-vh-grasp-closure.json
```

The strict generated-grasp smoke at K=16 currently retains 4/1/1/2/5 closure-certified
initial candidates for `obj_0` through `obj_4`. A finite standard real-IK sweep finds at least
one clean reachable candidate for the body and for every wheel (the wheel winners are
`a2:g0`, `a1:g0`, `a1:g0`, and `a2:g1` respectively; other arms/IDs also win). The older
24-action, 193.1 s artifact predates approach-retracted candidate generation and therefore has
a different grasp-database hash; do not treat it as a replay certificate for this version.

With the distinct HOME/EXIT model, the guarded five-part command completes all four wheels. The
matched diagnostic run found a 24-action plan in 56.7 s of search (617 IK binds, 503 failed),
using the strict closure, 5 mm clearance, 0.02 rad endpoint joint-margin, wrench, and inactive-arm
collision contracts. The artifact is `experiments/openarm-car-mini-home-exit-plan.json`. This
run deliberately disabled the later witness-set/floor/fusion passes to measure search itself;
it is a search/keyframe validation, not yet a motion-planning or simulator artifact.

After removing the unnecessary physical EXIT rendezvous, the current persisted 24-action plan
exports 43 motion tasks (previously 44). The standard 100 s optimized planner produced a
181-state path of cost 58.756; all three co-grasp spans have zero measurable relative drift.
Standard simulation with the established preliminary OpenArm calibration (`--joint-rate 0.5
--mass-scale 0.25`) completed 24/24 actions in 10,822 recorded frames, with worst final part
error 0.4 mm / 0.3 deg. The current artifacts are:

```text
experiments/openarm-car-mini-wrist-tight-no-exit-export.json
experiments/openarm-car-mini-wrist-tight-no-exit-mp.json
experiments/openarm-car-mini-wrist-tight-no-exit-standard.npz
```

For experiments at reach and joint-limit boundaries, an opt-in SSIK initialization shell is
available:

```text
--ssik-seed-neighborhood 0.01 --ssik-neighborhood-cap 6
```

It samples two deterministic Cartesian shells around a fixed tool target and filters/ranks the
resulting seven-axis branches by the real hardware limits. These poses are **only KOMO seeds**:
KOMO still solves the original target and the ordinary residual, collision, closure, clearance,
and joint-margin gates certify the result. Empty exact or nearby sets never refute an action.
Under `--open-witness`, shell generation is lazy and consumes part of an edge's existing reopen
budget (roughly half near-target, half fresh random basins), so the successful first epoch remains
unchanged. On the matched HOME/EXIT Car Mini run, the shell produced the same 24-action plan,
617 binds, and effectively identical 55 s search time; it is therefore not in the standard
profile.

The measured reason hard empty-SSIK rejection remains off is unchanged: the current soft-pose
KOMO tolerance has produced successful picks with no exact SSIK branch. Fast reach/collision
evidence is safe as ordering or initialization guidance, not as a general infeasibility proof.

Closure-enabled UR5-limited regressions also pass: standard Cross-7 produced 32 actions in
27.3 s (435 MiB peak RSS), standard Car Mini produced 19 actions in 10.7 s (308 MiB), and the
established mixed vacuum/two-finger tight-chamfer peg stool produced 19 actions in 33.8 s
(640 MiB). All completed with zero achieved-witness closure rejects. The saved plans are
`experiments/ur5-limited-cross-7p-4r-standard-closure-plan.json`,
`experiments/ur5-limited-car-mini-5p-4r-standard-closure-plan.json`, and
`experiments/ur5-limited-peg-stool-4-tight-chamfer-closure-plan.json`.

The archived pre-calibration run completed a 27-action plan in 85.9 s of search and 121 s
end-to-end, including
wrench checking, the shared 5 mm witness gate, strict collisions, open-witness search, witness
sets, and floor witness optimization. Peak RSS was about 526 MiB. Its largest local fill
(`obj_5`) contained 9,216 unique symbolic states and about 214k edges; compact incremental-search
bookkeeping stores only one pending/alternate copy per incoming edge and interns value-equal
states, preventing the repair history from growing without bound.

Those artifacts carry the older grasp-database hash. After the OpenArm gripper-centre and
closure-aware candidate corrections, they are useful historical/self-contained replay data but
are **not** current-model certificates. A normal load fails closed on the hash mismatch. An
explicit `--refresh-witnesses` may use an old action list only as a symbolic skeleton; every
successor and IK witness is then recertified under the current database. The three archived
Cross skeletons currently fail that symbolic replay at different mid-assembly grasp actions, so
none was silently promoted to a new plan.

The historical matching motion plan and replay command were:

```bash
python3 main.py \
    --config experiments/openarm-cross-4r-standard-ee180.json \
    --load-plan experiments/openarm-cross-4r-standard-ee180.json \
    --motion-plan --motion-profile standard --mp-seed 11023 \
    --mp-export-save experiments/openarm-cross-4r-standard-ee180-export.json \
    --mp-save experiments/openarm-cross-4r-standard-ee180-mp.json

python3 sim.py run experiments/openarm-cross-4r-standard-ee180.json \
    --mp-path experiments/openarm-cross-4r-standard-ee180-mp.json \
    --joint-rate 0.5 --mass-scale 0.25 \
    --record experiments/openarm-cross-4r-standard-ee180-calibrated.npz
```

That 300-second optimized planner pull reached all 48 exported tasks and produced a 251-state
shortcut path; all six co-grasp spans had zero measurable relative drift. `mr_bench` reserves
two 100,000-by-DOF arrays for every mode by default (about 43 MiB per four-OpenArm mode). The
local wrapper now starts each array at 1,024 rows and uses the upstream lossless doubling path;
sampling, collision predicates, and accepted paths are unchanged. This prevents the long mode
chain from exhausting several GiB merely in empty capacity.

The current seven-part OpenArm Cross result uses the motion-distance witness selector and the
denser swept-path contract. Search found a 30-action plan in 147.44 s; the floor witness pass
selected from 283 candidates over 24 IK steps and persisted 18 witness sets. The first ordinary
0.01-rad motion artifact was correctly rejected by simulation: a thin
`a2_left_finger1_coll`/`obj_0` collision occupied an interval between the coarse edge samples.
Replanning with the now-standard 0.002-rad shortcut certificate produced a 303-state path of
cost 84.788. A 29,541-sample base-path audit passed, the phase scheduler refined from 10 ms to
4 ms, and the 0.002-rad final audits accepted the smoothed/batched schedule. MuJoCo completed
30/30 actions in 14,545 recorded frames. Current artifacts:

```text
experiments/openarm-cross-4r-motion-witness-current-plan.json
experiments/openarm-cross-4r-motion-witness-current-mp-fine.json
experiments/openarm-cross-4r-motion-witness-current-standard.npz
```

The OpenArm simulator currently needs the same *explicitly labelled execution-model
calibration* as the preliminary FR3 validation: half-rate motion and quarter-mass placeholder
parts. With the robot-independent profile alone at full placeholder mass, the strict run fails
when a support pinch carries 0.45 Nm against its saved 0.40 Nm scalar cap; the later misses are
downstream. With the two overrides above it completes 27/27 actions, recording 13,815 frames,
with worst final part error 1.0 mm / 0.4 deg. Do not fold those values silently into the common
profile: full-mass validation still requires calibrated OpenArm grasp/actuator parameters or a
co-grasp load-sharing controller consistent with the search wrench LP.

The candidate count remains a measured calibration, not a cosmetic speed knob. With the same seed,
12 candidates did not finish in 200 s; 14 found a plan but took 150 s of search and grew its
last fill to about 402k edges; 20 candidates after standard grasp widening crossed a 1.3 GiB
RSS guard. The required transfer/mate diversity begins around this boundary (the successful
plan uses grasp IDs 12 and 15), so 16 is currently the safe OpenArm Cross setting. A tested
action-level empirical failure ordering was also slower (106 s search versus 85.9 s): an action
that fails in one holder/support scene can be feasible in another, so such evidence must not be
treated as a global blacklist or priority. Do not silently disable wrench or collision gates to
make a smaller grasp set pass.

Before the compact regenerating frontier fix (2026-08-29), a fresh current-hash K=16 Cross
search certified parts 1--5 but the final exact LazySP fill grew to roughly 350k pending entries
and 430k cached successor edges. Under the required 4 GiB process guard, the following live
phase exhausted address space; this was correctly reported as resource exhaustion rather than
an infeasibility certificate. Releasing completed fill engines and using
`PYTHONMALLOC=malloc MALLOC_ARENA_MAX=1` returned dead pages (measured RSS reductions around
50 MiB per wide phase), but could not compact that still-live frontier. The current 30-action
artifact above is the successful post-fix run.
K=14 reduced some frontiers but removed an easy predecessor realization and spent 290 s/3,329
IK binds on part 5, so it is not a replacement recipe. The correct next fix is a compact or
regenerating LazySP frontier, not weaker collision/wrench gates or a larger unguarded run.

The profile deliberately does **not** guess assembly-specific grippers, insertion force/noise,
layout, special bracing restrictions, or part count. Put those after the profile in the command.
It also does not run the motion-only `--fuse-cograsp` pass; request that when exporting a motion
plan.

The standard profile uses the scene-reserved hybrid grasp database. Primitive parts retain a
conventional-plus-sampled candidate alphabet; mesh and compound parts use sampled candidates
because they have no meaningful six-face database. With `--grasp-sampler-scenes`, the sampler
first puts deterministic representatives for each available per-kind pick, mandatory-mate, and
intermediate/final component-support fact in a K-independent master (one row may satisfy several
facts), then appends the hybrid tail. Conventional poses therefore remain stable source
candidates but are no longer guaranteed to occupy the positional prefix. A requested K only
contains the reservations that fit in that prefix. These labels only order the finite database;
exact IK, collision, and wrench checks remain authoritative. The database content hash is saved with the plan, and replay fails closed if
positional grasp ids would be interpreted against different content. Current-format artifacts
that still contain the former version marker `4` remain readable; obsolete formats 1--3 fail
explicitly and must be regenerated.

The standard profile generates one scene-ranked finite grasp master before search. Its
candidate family includes the nested 45-degree pinch rolls. Search starts with K=20 and
reveals stable prefixes in batches of 8 up to `--grasp-widen-max 96` after a no-plan return;
scene/closure classification is not repeated at every K. The maximum also controls how many
raw proposals are evaluated before selection, so increasing it can expose a grasp previously
hidden by the old 32/64-proposal scene-probe cap. `--grasp-sampler-candidates` instead changes
the number of sampled surface contact points. Larger maxima cost more preprocessing and may
make the search much larger; a timeout is not a proof that the finite master is infeasible.
Old IDs never change during a run, and successful IK witnesses remain valid. The `fabrica`
profile inherits the roll family but sets both the initial and maximum prefix to K=64 by
default; `--grasp-widen-max N` alone requests a larger finite master on CAD scenes. A saved
plan persists its profile, effective `--grasp-sampler-keep`, database hash, and widening policy.

To profile this preprocessing without changing the candidate order or search, prefix a normal
solve with `ASSEMBLY_GRASP_PROFILE=1`. The `[grasp-prof]` lines report each part’s raw proposal
time (and whether it was built or cached), finite-master selection time, counts and time for
closure/scene/retraction checks, and view materialization. They also report signature/master
access and preflight timing. The latter stages include some of the per-part work, so do not sum
all displayed lines as independent phases. Profiling is optional and is not saved in the plan.

An experimental finer-grained alternative is available as `--grasp-local-widen`. With a named
standard profile, that explicit flag replaces (and logs that it replaces) the global
`--grasp-widen` restart. It constructs one immutable finite sampler master up to
`--grasp-widen-max`, while the first resumable milestone session initially exposes only
`--grasp-sampler-keep` IDs per part. A failed concrete fill may expose one `K -> K+B`
incoming-part batch before ordinary milestone repair continues; finite IK failures remain open.
If that whole compact session exhausts, the milestone iterator starts exactly one fresh eager
session with every part exposed to its actual finite master cap. This completion session is used
by direct solves and `--mp-open-plans` alike, except when a deadline ends the compact session.
It restores the complete configured finite grasp action space without promoting an omitted column
or failed IK attempt to a no-good. The saved plan records the master hash and final per-part
active prefixes.

The standard profile uses `--component-grasp-keep 32` for grasps on an already rigid
multi-part component. Singleton pickup tables remain unchanged. For each larger component,
the per-part diversity-ranked masters are merged round-robin by stable grasp ID and candidates
are checked against the geometry of the *whole* component; only the first `N` collision-free
rows are exposed.  Thus the component has a bounded assembly-level alphabet instead of blindly
inheriting up to `N` rows from every constituent.  The action label remains the ordinary
`(contacted_part, grasp_id)`, so witnesses, wrench contacts, IK reuse, and saved plans need no new
identity convention.  These bounded component tables are also constructed lazily: the finite
dependency-closed key set is known up front, but geometry and collision rows are built only when
preflight, the local reverse PDB, or concrete search requests that component.  Explicit
`items()`/`values()` consumers still materialize the full finite table.  If the restricted
search returns no plan, the CLI automatically performs one completion retry with the exact
historical full constituent union. A finite compact miss is therefore not reported as
infeasibility. The resumable `--mp-open-plans` stream uses the same completion rule: it first
enumerates restricted plans, then reopens the full union while retaining suspended plans and
exact witness evidence. A winning saved plan records `32` when the restricted alphabet produced
it and `0` when the full-union completion did. Use `--component-grasp-keep 0` for the matched
historical-union ablation.

The promotion check used seed 11023 on the established OpenArm Car Mini-5, OpenArm Cross-7,
and mixed-tool Chair-8 scenes. All six matched runs succeeded and every K=32 run solved without
the completion retry. Car Mini retained the same 26-action cost-36 plan and 208 IK binds while
wall/search time changed from 46.30/21.28 s to 36.53/19.59 s. Cross improved from 30 actions at
cost 40 to 29 actions at cost 38, from 734 to 590 IK binds, and from 109.00/81.03 s to
77.15/55.02 s. Chair retained the same 36-action cost-47 plan and 77 IK binds while wall/search
time changed from 76.18/32.58 s to 28.38/13.39 s; peak RSS fell from 651 to 446 MiB. Reloading
all three compact artifacts reused their persisted witness blocks with zero IK calls. The
manifest, raw logs, plans, and summaries are in
`benchmarks/results/component-grasp-standard-regression/`.

This component cap is different from increasing every singleton part prefix. In the historical
K=20 study, eagerly starting OpenArm Car Mini with 32 singleton grasps increased search time from
41.2 s to 89.2 s and failed binds from 405 to 658, for only one additional initially feasible
wheel grasp. A forced-omission regression (all IDs 0--19 unavailable) failed with K=20, widened
once, and solved with grasp 23. Those measurements motivate retaining prefix widening as an
explicit runtime-oriented ablation; the current standard instead accepts the measured overhead
of fixed K=64 to give every search comparison one common finite action alphabet.

`--grasp-refine` remains an explicit anytime ablation. It widens nested action spaces in
order—conventional/local/unbraced, then full-grasp/local/unbraced, then bracing, then all-part
scope—so its final epoch retains the complete configured action space; early misses are not
infeasibility claims. Use
`--no-grasp-sampler` for the original primitive-only database. Assembly-specific compliant
grasps still require an explicit
`--grasp-sampler-wrap W` override when the standard 15 mm two-finger pad-deformation allowance
is inappropriate. This wrap flag does not claim that a vacuum cup seals on curved geometry;
the separate `--allow-curved-grasps` switch remains an explicit idealized override.

`--grasp-sampler-strict` is an opt-in diagnostic/quality mode. It omits generated rows below
`--grasp-sampler-margin` or `--grasp-sampler-pad-support` instead of appending them as the
finite fallback used by the normal widening sampler. Conventional face grasps are unchanged;
if a part has no remaining isolated candidate, the preflight fails explicitly and asks for a
less restrictive margin/support or a corrected scene. This is intentionally not part of the
standard profile because a physically small or table-near part can legitimately need a lower
threshold.

`--grasp-physics-check` is the first exact-finger grasp-generation experiment. It requires
`--grasp-sampler --grasp-sampler-scenes` and, for each two-finger candidate in the bounded
master, reuses the exported MuJoCo gripper with its actual rotating linkage. In an isolated
world the jaws close from open, must establish bilateral pad contact without moving the object
more than the ordinary 10 mm capture allowance, and must retain it under gravity plus a fixed
load in each of the six tool-frame directions. The common load magnitude is selected with
`--grasp-physics-force` (10 N by default); it is a qualification load, not a fitted capacity.
Unary results are cached per `(part, pose)` and failing candidates are omitted from two-finger
grasp tables; vacuum candidates are unchanged.

The same flag also eagerly builds a two-hand compatibility table before search. Each candidate
pair first passes the calibrated geometric close sweep with the giver already closed. Only those
survivors run MuJoCo: each actual articulated hand must acquire and retain the complete rigid
component under the same six loads. Component/grasp results are cached, so dynamics is linear in
the number of component grasps even though the final compatibility lookup is pairwise. Search
never invokes MuJoCo; fused `handover` successors only query the finite Boolean table. Genuine
two-hand bracing remains governed by the wrench model rather than this one-hand transfer test.
IK and the wrench LP remain independent gates, and every accepted grasp continues to use the
one envelope selected by `--grasp-model/--wrench-profile`. The flag and load magnitude are
saved and participate in the grasp-database signature.

The sampler has one supported implementation: an immutable hybrid master with deterministic
per-gripper pick/assemble scene-role reservations. Exact IK and wrench checks remain
authoritative. Earlier task-conditioned sampler experiments were removed rather than retained
as production modes.

The two supported profiles share exactly the same geometry and witness-clearance contract:

- `--search-profile standard` includes bounded fair open-witness scheduling. Exhausting its
  two epochs means “not found within this effort,” not geometric infeasibility.
- `--search-profile fabrica` is `standard` with a fixed dense K=64 grasp alphabet and no
  grasp-prefix widening. It is intended only for imported CAD assemblies.

Profiles are opt-in so old experiment commands retain their exact behavior. Resolution order
is `explicit individual flag > explicitly selected profile > --config file > built-in
defaults`; both profile UI choices can be reversed explicitly with `--view` and
`--no-quiet-rai`. A plain `--config` invocation therefore remains an exact reproduction, while
adding a profile deliberately retunes only that profile's named fields. Unnamed scene-specific
values still come from the file.

New plan files persist the resolved search semantics (`weight`, milestone oracle/local PDB,
boundary and scope widening, open-witness policy, analytic-seed/failure policy, wrench-aware
signature settings, positive IK reuse, and the goal condition). The profile *label* is intentionally not replayed:
if `standard` changes later, `--config OLD_PLAN.json --solve` keeps the values actually used for
that artifact rather than silently adopting the new profile definition.

The standard search has three nested levels. Structural milestones enumerate dependency-valid
part or module additions and chronologically backjump when a chosen addition makes the suffix
fail. Each addition builds a goal-conditioned backward PDB in the uniformly projected grasp
space, then greedy best-first search uses that PDB to find validated local glue quickly.
Candidate paths are validated lazily with real IK; a failed stochastic query is suspended and
revisited with increasing bounded effort instead of being permanently rejected immediately.

Chair's dependency graph and detached back/post module are supported: the milestone driver builds
the module internally, joins it as one composite milestone, and can backtrack among outer
dependency-valid milestone choices. A downstream dead end now rejects only the exact symbolic
settled endpoint that produced it and lazily requests another endpoint from the same exact local
fill. The standard tries up to four, including recursively inside detached modules; the no-good
key also contains the exact parent state, so a suffix failure from one boundary does not poison a
different one. This remains bounded rather than globally complete: module-internal part order is
fixed, joint-space alternatives for one symbolic endpoint are handled separately by witness sets,
open-witness effort is capped at two epochs, and node/repair/PDB guards are finite.

### Full-state backward reference

`--full-backward` is the unprojected exact-search ablation. It retains the complete discrete
`State` identity (robot and raw-grasp identities, every support, placement id, and kinematic
parent), materializes the cheap-feasible graph reachable from the authored initial state, and
runs multi-source reverse Dijkstra from all reachable goal states. If real IK is enabled, the
shortest candidate is validated lazily; a failing IK edge is removed and reverse Dijkstra is
repeated. It does **not** use the signature-collapsed backward PDB or silently fall back to it.

The implementation materializes edges with the existing forward operators rather than
maintaining a second set of inverse operators that could drift semantically. Consequently this
is an exponential reference suitable for small ablations, not a replacement for the standard
milestone search. The default hard guards are 200,000 states and 2,000,000 edges; exceeding one
prints the exact reason and aborts. For example:

```bash
python3 main.py --solve --full-backward \
  --full-backward-max-states 50000 --full-backward-max-edges 500000 \
  --assembly cube-sequence-2x2x3 --parts 3 --num-robots 1 \
  --floating-ee --vacuum --no-real-ik --wrench \
  --no-grasp-sampler --placement-yaws 1 --no-placement-yaw-widening \
  --no-bracing --no-handovers --quiet-rai --no-view
```

This remains a small-problem diagnostic and is not part of the paper comparison runner. The
runner instead uses `ours-fine-projection`, which keeps the standard milestone solver and local
reverse Dijkstra construction but refines its grasp signature from `press` to
`press,pickface,solohold,braceside`. `--fill-raw-grasp-h` is the still finer diagnostic that
retains raw grasp ids in the local reverse table; it commonly reaches the 200,000-state guard
and is therefore also excluded from the routine paper matrix.

### Named motion-planning profile

Motion planning has its own profile because its settings do nothing during search and should not
be confused with search feasibility:

```bash
python3 main.py \
    --config experiments/PLAN.json --load-plan experiments/PLAN.json \
    --motion-plan --motion-profile standard --mp-seed SEED \
    --mp-export-save experiments/FINAL_EXPORT.json \
    --mp-save experiments/MP.json
```

`--motion-profile standard` expands to:

```text
--pre-pick 0.05 --pre-place 0.05 --pre-add-grasp 0.05
--post-pick 0.05 --pre-insertion 0.05 --post-release 0.05
--fuse-cograsp --fuse-pre-skill
--mp-sequence --mp-optimize
--mp-runtime 300  (no --mp-stall-timeout)
--mp-plan-candidates 4 --mp-plan-epochs 6 --mp-plan-initial-runtime 15
--mp-shortcut-iters 100 --mp-shortcut-strategy native --mp-shortcut-res 0.002
--mp-collision-tolerance 1e-4
--mp-robot-margin 0 --mp-margin-ramp 0 --mp-margin-mode linear --mp-table-margin 0
--no-mp-wrench-gate
--mp-cograsp --quiet-rai --no-view
```

For an execution-clearance stress test, select the otherwise identical profile:

```bash
--motion-profile execution
```

It changes only the auxiliary clearance policy to
`--mp-robot-margin 0.05 --mp-margin-ramp 0.05 --mp-margin-mode binary`. Thus different robots
must keep 5 cm clearance during ordinary transit, while the auxiliary margin becomes exactly
zero whenever the relevant end effector is within 5 cm of either endpoint of its current
segment. The underlying collision-tolerance and joint-limit predicates remain active there;
"zero" removes only the extra 5 cm shell. Relaxing at both endpoints is important: relaxing
only at the goal would make the same physical keyframe legal before a mode switch and illegal
immediately afterward. `--mp-margin-mode linear` retains the previous continuous interpolation
for experiments that need it.

Both coupling passes are therefore always enabled by the standard motion profile. Their scopes
are different: `--fuse-cograsp` jointly resolves a literal `(add_grasp, assemble)` witness pair;
it correctly reports zero pairs for plans that contain only picks and handovers.
`--fuse-pre-skill` couples each generated pre-pick/pre-place/pre-assemble/pre-add-grasp endpoint
to its skill endpoint. In addition, every actor-only pre-skill solve now has a hard 0.35 rad
joint-space corridor to the actual skill witness. If no requested backoff rung stays on that IK
branch, export uses the skill witness itself (a zero-length funnel) instead of accepting a
geometrically nearby but completely different arm configuration. Thus a standard export may
shorten a 5 cm pre-skill funnel in a tight scene, but it may not introduce an IK-branch jump.
For assemble, the standard search has already certified and persisted the exact mouth/seat pair,
so this pass reuses it and performs no assemble IK. The post-pass remains useful for pick, place,
add-grasp, and handover funnels, which are export conveniences rather than search-feasibility
conditions.
The `--pre-add-grasp` funnel covers both `add_grasp` and `handover`: these have identical
incoming-hand geometry and differ only in the ownership update at the endpoint. This keeps a
fast receiver at a certified standoff while the giver approaches, instead of letting fixed
finger collision proxies press into the payload before the handover fires.

The `1e-4` motion tolerance is therefore now pinned rather than inherited accidentally. The
accepted shortcut geometry is sampled at 0.002 rad, and the standard simulator independently
preflights and audits at the same resolution. This closed a measured OpenArm Cross failure in
which the old 0.01-rad grids skipped a thin finger/object collision interval while a controller-
rate sample landed inside it. RRT*
optimization is also part of the production contract: once a solution exists, the planner keeps
improving it for the remainder of that pull, and the certified shortcut pass runs afterward. The
standard disables the old 30-second frontier-stall cutoff and permits the full 300-second budget;
this covers the historically difficult Chair transit between its legs instead of assuming that a
temporarily stationary mode frontier is hopeless. The
profile intentionally leaves `--mp-seed`, `--mp-save`, and `--mp-export-save/load` to the
caller. A final export remains important: it makes retries zero-IK and records the effective
collision/margin certificate.

`--mp-wrench-gate` optionally extends the same monkey-patched feasibility predicate with
quasi-static gravity equilibrium at every sampled configuration. It uses the task planner's
physical grasp/table capacities and support state, canonicalizes each contact model in the
carried body's frame, and reuses the warm LP while the sampled arm pose changes the gravity
direction. It deliberately does not apply insertion or screw loads during free-space transit;
those belong to their task/skill certificates. The gate currently requires `--wrench` and the
standard fixed-sequence export. It is off in both named motion profiles so a topological
`--no-wrench` baseline stays genuinely force-blind; use it explicitly for the gravity-aware
motion-planning ablation. Its enabled/disabled value is persisted in the final export and may
be explicitly overridden on export replay.

Standard now adds no universal robot-clearance shell: the former
10 cm hard margin substantially changed free-space connectivity in compact workcells even though
its participant ramp relaxed to zero at task endpoints. Positive `--mp-robot-margin`,
`--mp-margin-ramp`, `--mp-margin-mode`, and carried-part `--mp-table-margin` remain explicit
experiment settings (or are selected together by the explicit `execution` profile);
they should not be inferred from endpoint clearance because that does not certify connectivity
through the segment. A future portable robustness objective should optimize clearance softly
after finding a feasible path. Tight peg/stool exports should continue overriding the auxiliary
funnels explicitly while retaining the 5 cm insertion funnel.

When the caller has not set them, the motion profile creates and selects
`NUMBA_CACHE_DIR=/tmp/assembly-numba-cache` and
`MPLCONFIGDIR=/tmp/assembly-mpl-cache` before importing the planner. This prevents the external
`mr_bench` Numba functions from failing with “no locator available” and avoids Matplotlib trying
to write under an unwritable home directory. Explicit environment values are preserved.

For a **fresh search whose result must also admit a motion path**, use the open-plan driver:

```bash
python3 main.py --solve --motion-plan --search-profile standard --motion-profile standard \
    --assembly ASSEMBLY --parts N --num-robots R --robot ROBOT --seed SEED \
    --mp-open-plans --mp-plan-candidates 4 --mp-plan-epochs 6 \
    --mp-plan-initial-runtime 15 --mp-seed MP_SEED \
    --save experiments/PLAN.json \
    --mp-export-save experiments/FINAL_EXPORT.json \
    --mp-save experiments/MP.json
```

This is deliberately plan-level feedback, not an action blacklist. The milestone search is a
resumable generator. Each complete discrete realization receives one short MP pull; a miss
suspends that plan and asks the same search session for another exact milestone boundary. After
the candidate pool has been probed once, every suspended plan is reopened fairly at doubled
runtime tiers (15, 30, 60, 120, 240, and 300 seconds under the standard profile). The MP seed is
deterministic per candidate and tier. Every pull has `--mp-optimize`; when it finds a solution,
the remainder of that tier improves the path before promotion.
The open-plan driver preserves the resolved search sampler schedule. Under the current standard
this is the fixed scene-reserved K=64 hybrid database with no whole-search grasp widening, not a
conventional-first `--grasp-refine` stage. Explicit `--grasp-refine` remains a separate ablation;
the driver still skips duplicate complete plans across its normal widening/reopening schedule.

Each candidate is certified and exported once, then later pulls use `--mp-export-load` and do
zero IK. MP runs execute in child processes because export changes global planner/model state;
where `prlimit` is available each child also has a 6 GiB address-space guard. The parent search
installs the same soft limit itself. Four GiB was measured insufficient for Chair: its mixed-arm
witness export reached 3.87 GiB before RRT* needed another tree. Set `MP_OPEN_PARENT_AS` or
`MP_OPEN_CHILD_AS` explicitly (bytes) to change either guard. Intermediate plans, exports, paths,
seeds, statuses, and the machine-readable `manifest.json` live under
`<MP stem>-open/<timestamp>-<pid>/`; only a successful plan/export/path triple is promoted to
the requested output names.

Exhausting the configured tiers prints a failure and keeps every candidate suspended in that
directory. It does **not** report symbolic or geometric infeasibility and does not silently fall
back to an unplanned plan. Increase `--mp-plan-epochs`, `--mp-runtime`, or the candidate count for
more finite effort. This initial implementation is restricted to a fresh ordinary milestone
solve (`--milestones --real-ik --motion-plan`), which is what the standard search profile uses;
plain `--load-plan` retries remain the separate zero-IK command above.

Chair validation (2026-08-25): the historical
`chair8-4r-mixed-robust-seed1-final-export.json` is no longer a valid stress artifact under the
limited UR10 model: its fixed `06_pre_place` keyframe has wrist-2 at 0.586 rad while the current
upper limit is 0.283 rad. Both tested seeds correctly rejected it; additional MP time cannot fix
an invalid goal. A fresh standard search produced a 39-action, 71-task export with nine
multi-witness GoalSets. With seed 1, the first optimized 15-second tier found a complete path at
6.09 s, improved RRT* cost from 549.23 to 169.61, and the certified shortcut pass produced a
380-state path of cost 125.86. Thus the lowest tier solves the hardest current valid scene; the
30--300 s tiers and alternative complete plans remain headroom for unlucky seeds/realizations.
The run also measured why the memory guard is 6 GiB: its witness/export process reached 3.87 GiB
before allocating the planner trees. Current artifacts are
`chair8-4r-mp-open-standard-seed1-{plan,export,mp}.json`.

Pre-retuning profile audit (2026-08-24): fresh searches under the then-current `standard`
completed on Cross-7 (27 actions),
Car-mini (24 actions), mixed-gripper Chair-8 (38 actions), and sampled-mesh Peg-hole (3 actions),
each with complete persisted witnesses and floor optimization; the larger three also exported
witness sets. The freshly generated Cross plan completed all 50 motion modes under
`--motion-profile standard`; its 100-shortcut result had 683 serialized states and the
final-export retry performed zero IK solves. Cube-8 vacuum was a failure of those former
profiles: both grew repeated 500--1500-bind milestone fills and were killed under memory
pressure before the sixth/seventh addition. Do not describe the profile as universal until that
fill-state growth regression is fixed; changing collision tolerance or silently falling back to
an older plan would not address it.

### Experimental fixed-order chain heuristic

`--optimistic-chain-h` is the scalable middle ground between plain optimistic fills and the
global milestone abstraction. It runs one backward signature solve over only
`base -> prefix 1 -> ... -> complete` in `assembly_order`, and permits only the next part to be
loose at each prefix. Rest faces, grasp classes, holders/braces, wrench facts and the entire
remaining fixed order are retained; alternative order ideals and combinations of unrelated
loose parts are not. The narrow fact-guided phases are only tried first: unrestricted local
fill phases remain the completeness fallback.

```bash
python3 main.py --config experiments/PLAN.json --search-profile standard --solve \
    --optimistic --optimistic-chain-h --no-optimistic-fallback \
    --placement-yaws 1 --no-placement-yaw-widening \
    --optimize-witnesses off --witness-sets 1 --no-fuse-cograsp \
    --no-hoist-releases --quiet-rai --no-view
```

Single guarded measurements on noisy Car-mini (three limited UR10s, 20 N insertion force,
10 mm insertion noise) gave 2,485 chain states in 0.64 s and a complete hands-off 25-action plan
in 22.76 s search / 30.73 s process time. Milestones built 9,076 states in 4.40 s and found a
25-action plan in 30.57 s / 44.85 s. Plain optimistic exceeded three minutes and exhausted its
compact grasp epoch. On the mixed-gripper 10-part fixture the chain abstraction built 25,627
states in 1.38 s; the global signature heuristic exceeds its 2,000,000-state guard.

This flag does **not** support detached connect modules. Full Chair-8 contains the back-plus-posts
module, so the chain path now fails immediately with an instruction to use `--milestones`
instead of expanding the impossible main-component order to a memory failure. Chair-8 milestones
completed the current standard comparison in 38 actions and 101.70 s search. The optimistic
solver also now runs `_settle_to_goal` after its last fill; it no longer reports a still-held
complete assembly as a successful hands-off plan.

For reproducible experiments, spell those defaults out in the search command:

```bash
python3 main.py --solve \
    --assembly ASSEMBLY --parts N --num-robots R --weight inf \
    --wrench --grasp-model asymmetric \
    --sig-collapsed-h --sig-components press \
    --witness-margin 0.005 --witness-sets 4 \
    --pre-pick 0.05 --pre-place 0.05 --pre-add-grasp 0.05 \
    --pre-insertion 0.05 --post-release 0.05 \
    --quiet-rai --no-view --save experiments/PLAN.json
```

Generate motion from that persisted plan, then retry BiRRT from the exact final export without
ever resolving witnesses again:

```bash
python3 main.py \
    --config experiments/PLAN.json --load-plan experiments/PLAN.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill \
    --mp-sequence --mp-optimize --mp-runtime 300 \
    --mp-seed 11023 --mp-shortcut-iters 100 --mp-shortcut-res 0.002 \
    --mp-collision-tolerance 1e-4 --mp-robot-margin 0.1 --mp-margin-ramp 0.1 \
    --mp-cograsp --quiet-rai --no-view \
    --mp-export-save experiments/FINAL_EXPORT.json --mp-save experiments/MP.json

python3 main.py \
    --config experiments/PLAN.json --load-plan experiments/PLAN.json \
    --motion-plan --mp-export-load experiments/FINAL_EXPORT.json \
    --mp-sequence --mp-optimize --mp-runtime 300 \
    --mp-seed 11023 --mp-shortcut-iters 100 --mp-shortcut-res 0.002 --mp-cograsp \
    --quiet-rai --no-view --mp-save experiments/MP_RETRY.json
```

## Standard simulation reproduction

Use one common, fail-closed execution profile for current saved motion paths, independent of
assembly and robot family:

```bash
python3 sim.py run experiments/PLAN.json \
    --mp-path experiments/MP.json \
    --record experiments/REPLAY.npz

python3 sim.py replay experiments/REPLAY.npz
```

For visible two-finger opening/closing, add `--animate-grippers` (optionally
`--gripper-motion-time 0.25`). The renderer commands the object-dependent aperture persisted
with the selected grasp witness; it does not infer a new aperture from the live/rendered pose.
It then moves the fingers concurrently with the arm trajectory.
Add `--wait-for-gripper` when that overlap is undesirable. With the standard phase scheduler,
closing and opening become collision-checked, robot-local reservations: only the robots involved
in that event park, while unrelated robots may continue along their paths. A pick closes before
grasp activation and post-pick retreat; a release opens before departure; and a handover executes
receiver-close, ownership transfer, then giver-open. Each commanded phase takes
`--gripper-motion-time`. The phase smoother and motion batching preserve these reservations, and
simulation fails closed if a later postprocessing pass removes one. Without `--phase-schedule`,
`--wait-for-gripper` retains the legacy global blocking behavior. The flag is opt-in because the
historical concurrent animation remains useful when execution time matters.
These are the grippers' actual hinge/linkage kinematics driven by MuJoCo position actuators
(six synchronised mimic joints for Robotiq 2F-85, two opposed hinges for OpenArm), but they are
introduced only in the simulation export and therefore do not change IK or motion-planning
dimensions. The moving manufacturer meshes are presentation geometry; the planner-certified
open-pose finger collision proxies remain rigidly attached to the gripper base. This is
intentional: allowing approximate capsules to pinch an already weld-attached object would add
a second, uncalibrated grasp controller and count its internal reaction against the grasp wrench
limit. The existing weld remains the sole grasp/load model. Because the finger joints are
ordinary MuJoCo qpos, their articulated motion is embedded in the same NPZ and needs no special
replay command. Vacuum hands have no moving geometry and are left unchanged.

An experimental physical-pinch mode is available separately:

```bash
python3 sim.py run experiments/PLAN.json --mp-path experiments/MP.json \
    --contact-grasps --record experiments/contact-grasps.npz
```

`--contact-grasps` applies physical contact to each two-finger hand in the workcell. It keeps the
finger collision proxies on the articulated linkage, commands the Robotiq master joint with a
bounded actuator, and declares a grasp only after both inner pads carry load. No robot--part weld
is created for those hands; bilateral frictional contact alone transports the part. Vacuum hands
in a mixed workcell retain the established suction weld model. It currently requires
`--physics-step 0.002`; the 4 ms mode
softens the calibrated pad contact through MuJoCo's `refsafe` clamp and permits numerical creep.
Part--part mate welds are unchanged. Generate the task plan with `--grasp-physics-check`: its
ordered handover table first rejects geometric overlap of the two closed hands and then certifies
each articulated pinch under the six signed tool-frame loads. At execution, a handover is a
transactional physical event: the giver keeps supporting the payload while the receiver closes;
only verified bilateral receiver contact commits the transfer and starts opening the giver. A
failed receiver acquisition leaves the giver active and fails the action rather than introducing
an unsupported interval or silently replacing contact by a weld. This mode is not a new default.

Two end-to-end contact examples are checked in under `experiments/contact-grasp-handover/`:

```bash
# Full 7-part Cross, three physical parallel-jaw robots and physical handovers.
python3 sim.py replay \
  experiments/contact-grasp-handover/cross7-ur5-3r-contact-v6-pad-slow.npz 2

# Five-part Chair: vacuum staging, physical vacuum-to-jaw handovers, and four jaw-held pegs.
python3 sim.py replay \
  experiments/contact-grasp-handover/chair5-ur5-4r-contact-pinch.npz 2
```

The Cross uses `--joint-rate 0.5`: at the nominal 1 rad/s, dynamic finger/object slip can occur
despite the same quasistatically certified path. The Chair example deliberately disables the
task-level wrench model to isolate the physical grasp/handover/peg-execution experiment.

`sim.py run` pins the following contract, so a reproduction does not change when
individual built-in defaults do:

```text
--actuated --skill-servo --arm-arm-contacts --no-spline
--phase-schedule --phase-servo-time 0.15
--phase-smooth --phase-smooth-time 0.12
--phase-batch --phase-batch-min-motion 0.08 --phase-batch-max-gap 0.35
--phase-audit-resolution 0.002
--settle-tol 0.002 --settle-max 1.0 --servo-pick
--servo-force 6 --servo-max-dev 0.01 --servo-torque-guard 0.6 --servo-tol 0.002
--servo-place-seat
--force-limits --inertia-factor 1 --wrench-margin 1
--path-preflight --path-preflight-tolerance 1e-4 --path-preflight-resolution 0.002
--joint-rate 1 --mass-scale 1 --coll-scale 1
--no-corridor --no-phase-intent --no-c2-spline
--require-mp-path --require-success
```

The profile means:

- when paired with `--mp-path`, execute the persisted real-arm path without resolving IK or
  rerunning BiRRT;
- reject an invalid saved path during strict collision/joint-limit preflight;
- use the collision-audited offline phase scheduler, intrinsic phase-rate smoothing, and
  certified short-island batching;
- preserve the planner-certified geometric paths. `--spline` and `--c2-spline` remain explicit
  spatial-smoothing experiments: their nominal collision certificate does not provide the
  millimetre-scale execution slack needed around a part that has physically settled on the
  table;
- run actuated MuJoCo execution with compliant skill/seat servos, settling gates, normalized
  grasp force/torque guarding, acquisition touch termination, and physical cross-arm contacts;
- model grasp interfaces as compliant (12 ms weld time constant), while keeping already-mated
  assembly connections near-rigid (4 ms), matching the rigid components used by task and motion
  planning. `--weld-timeconst` remains a shared experimental override and
  `--mate-weld-timeconst` can override only the latter;
- embed the generated robot, carrier, table, result list, and recorded states in the NPZ.
- refuse to run a fresh stochastic BiRRT when the persisted `--mp-path` is omitted, and return a
  nonzero process status if physics reports any failed action. The NPZ is written first and now
  embeds the resolved profile, input artifact names, success bit, and final pose errors.

When diagnosing a pause near a table or mate, set `SIM_SERVO_TRACE=1` for a behavior-neutral
trace of force-guard transitions (measured force, threshold, and contact pair). The standard
`--servo-place-seat` skill treats a table place as sustained physical support and stops the
descent there; it prevents the schedule from repeatedly tracking a slightly lower CAD table
datum. Part-to-part insertion uses the complete child-in-parent transform selected by the
witness. Motion planning owns the full free-space approach and reaches/fires the explicit
`pre_assemble` funnel-mouth waypoint before handing over; the earlier proximity handoff could
visibly skip the end of that approach. From the mouth onward, insertion controls the measured
child pose relative to the measured parent rather than a fixed world pose. With both sides held
it uses the cooperative
relative Jacobian `[J_child, -J_parent]`; the persisted assembly witness supplies only a
null-space posture bias. Unrelated robots may keep following their certified phase schedule.
The controller aligns to 0.5 mm/0.5 degrees at fixed depth, advances on the authored mate axis,
slows to 20 mm/s after guide/socket contact (and 10 mm/s after a retry), and gives a stalled or
wrench-guarded insertion one bounded 3 mm backoff/re-alignment retry. The guard evaluates
the normalized wrench on both the incoming and target holders. Ordinary guide-wall contact
before full depth is not itself classified as a jam. Contact is
accepted only inside the 10 mm/2 degree seat gate; the wider 50 mm capture range merely prevents
a failed diagnostic rollout from exploding when a datum weld is installed and is never reported
as success. Newly exported paths also contain an explicit held-object `post_pick` retreat. For a
table pick this is a separately collision-checked IK endpoint along the table normal (+world z),
with grasp and EE orientation unchanged; other acquisition contexts retain the grasp-approach
direction. This prevents an execution-side lift from decaying back to the table-level grasp
reference while leaving the free-space planner free to detour around nearby parts.

### Exporting an open-loop robot trajectory

Current recordings also carry a per-frame controller-mode trace and exact logical gripper-command
events. Convert one into a hardware-oriented joint artifact with:

```bash
python3 tools/exporters/export_open_loop.py experiments/REPLAY.npz experiments/OPEN_LOOP.npz
python3 tools/exporters/export_open_loop.py experiments/REPLAY.npz experiments/OPEN_LOOP.json \
    --dt 0.05
```

The exporter refuses a replay whose simulator outcome is failed. For inspection only, a
failed rollout can be exported explicitly with `--allow-failed`; such an artifact is not a
hardware-ready command trajectory.

The compressed NumPy (`.npz`) output uses the channel-oriented `assembly-open-loop-v2` schema.
Portable
JSON uses the row-oriented `assembly-open-loop-json-v3` schema: top-level robot/joint ordering is
followed by a `samples` list whose records contain exactly `q`, `qd`, `qdd`, `time`,
`opening_gripper`, `closing_gripper`, `servo_controller`, and `attached_objects` (a per-robot
object name or `null`). JSON is uniformly resampled at `--dt 0.05` (20Hz) by default.
`--grasp-wait-mode auto` preserves the already certified robot-local waits in recordings made
with phase scheduling and `--wait-for-gripper`; it does not insert another dwell or warp the
timing, and an unrelated robot can therefore keep moving during a close. The attachment trace
changes at the recorded grasp-activation time, not when closing starts. Use
`--grasp-wait-mode preserve` to require this behavior and fail on an older recording.

For recordings without scheduled waits, `auto` retains the legacy export behavior: each close
inserts a stationary `--grasp-wait 0.5` second global dwell. Simultaneous closures share that
dwell, and `closing_gripper` remains asserted throughout it. For a handover, the receiver closes
throughout this dwell while the giver remains attached; only at the end does the giver receive
its one-sample opening command. A zero grasp wait is rejected when the trajectory contains a
handover because it cannot represent that ordering. Other opening commands occupy one output
sample. `q` is the joint motion that actually occurred in simulation—not the nominal
motion-planner reference—and its velocity/acceleration are finite differences after resampling.
`servo_controller=true` means that the
source sample depended on an online execution hook (`settle`, `skill_servo`, `approach`, or
`wrench_guard`), which hardware must implement or deliberately replace.

The NPZ carries the same attachment trace as an `attached_objects[N, R]` string array (empty
string means no object), aligned with its `q`, `qd`, and `qdd` rows. The trace applies close/open
events in recording order. The JSON expansion deliberately represents a handover as a temporary
co-grasp during the close dwell, followed by the giver detaching at its delayed opening sample.

Recordings created before these annotation channels are still convertible, but the exporter
marks every affected JSON sample as `servo_controller=true` and prints a warning. New recordings
log exact logical opening and closing independently of visual finger animation. For older
recordings, the exporter reconstructs closes plus unambiguous opens from the saved action marks
(`assemble`, `place`, `release`, and single-giver `handover`) and labels those events as inferred;
it never guesses among multiple possible handover givers.

The current UR scenes use the profile exactly as written: the visual cruise remains the
profile-pinned `--joint-rate 1.0` and parts retain `--mass-scale 1.0`. The current FR3 calibration
adds exactly two visible overrides:

```bash
--joint-rate 0.5 --mass-scale 0.25
```

These FR3 flags are an execution-model calibration, not part of the robot-independent safety
contract. Half-rate motion prevents inertial grasp overloads in the current FR3 servo model;
quarter-mass parts compensate for placeholder grasp/load parameters. Do not silently apply
quarter mass to UR experiments or present it as full-physics validation. The target is to
remove the FR3 overlay after its actuator, grasp-capability, and mass calibration supports
`--mass-scale 1.0`.

A verified four-FR3 reproduction is:

```bash
python3 sim.py run experiments/fr3-cross-4r.json \
    --mp-path experiments/fr3-cross-4r-mp.json \
    --joint-rate 0.5 --mass-scale 0.25 \
    --record experiments/fr3-cross-4r-phase-batched.npz

python3 sim.py replay experiments/fr3-cross-4r-phase-batched.npz
```

This completed 26/26 actions. The initial 10 ms phase grid exposed a cross-arm collision and
automatically retried at 4 ms; the exact final audit then passed. The NPZ contains both FR3
carrier bases and the RAI-derived 1.2 x 0.9 x 0.06 m workbench.

Do not add the following to the standard profile:

- `--limit-rate`: phase scheduling deliberately ignores this speed boost and uses
  `--joint-rate` as the visual cruise;
- `--c2-spline`: an independent, opt-in geometric smoothing experiment;
- `--phase-intent`: an experimental alternative to the established `--phase-batch` stage;
- `--corridor`, `--corridor-online`, or `--pace`: older scheduling paths, retained for
  ablations and artifact forensics;
- a non-unit `--coll-scale` or `--no-path-preflight`: geometry/certificate debugging overrides,
  never a standard validation. The profile pins the certificate-preserving no-op
  `--coll-scale 1` so a future global default cannot silently shrink geometry.

As with the search and motion profiles, an explicit flag wins over the sim profile. For example,
the FR3 rate/mass calibration above overrides the two neutral standard values. Use
`--no-servo-torque-guard` to disable that profile-owned guard explicitly. `--mp-path`,
`--record`, and `--gui` are outputs/inputs rather than algorithm defaults and are never guessed.

Artifact-specific commands later in this README reproduce the historical artifact named next
to them. They are not the current baseline when they contain one of these old or experimental
flags.

### 2026-08-14 robust evaluation

All four fresh searches used wrench checking, asymmetric grasps, collapsed-h `press`, floor
witness optimization, four-branch witness sets, and the shared 5 mm search/IK gate. All saved
paths reached their terminal mode at `--mp-collision-tolerance 1e-4`.

The assembly-specific search arguments were:

```text
Car:   --assembly car-mini --parts 5 --num-robots 3 --robot ur10_limited
       --seed 11023 --min-sep 0.3 --insert-force 20 --insertion-noise 0.01
       --tf-close-torque 0.5 --ignore-gravity --random-init
Cross: --assembly cross --parts 7 --num-robots 4 --robot ur10 --seed 0
Cube:  --assembly cube --parts 8 --num-robots 4 --robot ur10 --vacuum
       --seed 11023 --check-init
Chair: --assembly chair --parts 8 --num-robots 4 --robot ur10_limited --seed 1
       --grippers vacuum,vacuum,two_finger,two_finger --allow-curved-grasps
       --scatter-spacing 0.2 --min-sep 0.3
```

| assembly | search | MP result | co-grasp diagnostic | artifacts |
|---|---:|---:|---:|---|
| Car-mini, 3 limited UR10 | 25 actions, 12 witness sets | 44 modes, 522 serialized states | 16/16 pairs drift internally; 4/4 eligible keyframe pairs fused | `car-mini-3r-robust-seed11023-*` |
| Cross-7, 4 UR10 | 28 actions, 16 witness sets | 51 modes, 760 states | 0/3 pairs drift | `cross7-4r-robust-seed0-*` |
| Cube-8, 4 vacuum UR10 | 40 actions, 26 witness sets | 70 modes, 995 states | 3/4 pairs drift | `cube8-4r-vacuum-robust-seed11023-*` |
| Chair-8, mixed UR10 | 38 actions, 21 witness sets | 64 modes, 903 states | 1/4 pairs drift | `chair8-4r-mixed-robust-seed1-*` |

Current-audit note (2026-08-21): the car artifact in this table predates the strict
interpolated joint-limit preflight. It is now rejected at stretch 4 because
`a2_ur_wrist_2_joint=0.286711` exceeds its `0.28319` upper limit by 3.52 mrad. Regenerate its
motion path before using it for simulation/controller evaluation; `--no-path-preflight` only
turns it into an unsafe diagnostic and produced a large actuator tracking error in testing.

Chair requires the established mixed fleet and curved vacuum contacts:

```bash
--grippers vacuum,vacuum,two_finger,two_finger --allow-curved-grasps \
--scatter-spacing 0.2 --min-sep 0.3
```

The Cube rollout used the requested simulation flags and completed 40/40 actions:

```bash
python3 sim.py run \
    experiments/cube8-4r-vacuum-robust-seed11023-plan.json \
    --mp-path experiments/mp-cube8-4r-vacuum-robust-seed11023.json \
    --actuated --corridor --corridor-rate-clamp --coll-scale 0.8 \
    --spline --limit-rate \
    --record experiments/cube8-4r-vacuum-robust-seed11023-actuated.npz

python3 sim.py replay \
    experiments/cube8-4r-vacuum-robust-seed11023-actuated.npz
```

Strict preflight checked 35,008 samples and the scheduled corridor verified 5,232 samples with
no collision. The NPZ has 34,791 time samples: the motion planner did not return 34k states.
The 995-state saved path was compacted to 322 geometric waypoints, splined to 2,017 waypoints,
then rate/force/corridor retiming produced the recording samples. `--coll-scale 0.8` remains a
debug geometry override, so this rollout is not a certificate for the unscaled robot geometry.

This Cube rollout is retained as a dated artifact result. Its corridor scheduler and
`--coll-scale 0.8` debug geometry do not define the standard simulation profile above.

### Experimental C2 geometric smoothing

`--c2-spline` is a separate, opt-in geometry postprocessor. It replaces the default C1
Catmull--Rom stage for that run; it does not modify phase scheduling, phase-rate smoothing,
controllers, or event execution. Each mode-local free-motion stretch is fitted in composite
joint space by a clamped cubic B-spline. Its two task-witness endpoints are hard equalities,
while ordinary motion-planner waypoints are soft observations. The curve may therefore leave
and shorten the saved polyline instead of merely retiming it.

Safety is fail-closed. The emitted samples and the linear interpolation between every adjacent
sample are checked at `--c2-resolution` (default 0.01 rad maximum per-joint step) with the
mode's complete motion-planner collision predicate and the robot joint limits. Accepted fits
also obey `--c2-max-deviation` (default 0.20 rad in composite configuration norm). The fitter
tries progressively more faithful curves; if none certifies, that entire stretch stays on the
original preflight-certified polyline. Stretches containing a rigid co-hold or handover are
always protected because collision checking alone cannot enforce their closed-chain relative
tool transform. C2 continuity is consequently claimed within accepted free-motion stretches,
not across mode switches or protected/fallback stretches.

The implementation is isolated in `assembly_planner/c2_smoothing.py` and passes ordinary sampled waypoints to
the existing postprocessing pipeline. The established version remains the default. A current
car-mini evaluation command is:

```bash
python3 sim.py run experiments/car-mini-ur10-new.json \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-batch --c2-spline --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-c2-phase-batched.npz

python3 sim.py replay experiments/car-mini-ur10-new-c2-phase-batched.npz
```

Useful experimental controls are `--c2-control-ratio` (fewer controls smooth more),
`--c2-smooth-weight`, `--c2-samples-per-seg`, `--c2-max-deviation`, and
`--c2-resolution`. Defaults should be treated as a conservative starting point; rejection does
not require per-assembly tuning because local fallback is automatic.

### Visual-quality phase scheduling in simulation

A paper-style formulation, pseudocode, certification argument, and current limitations are in
[PHASE_SCHEDULING.md](PHASE_SCHEDULING.md).

Use `--phase-schedule` for the offline coordination-space scheduler. It keeps each robot on its
already planned joint path and chooses only its monotone phase along that path. The objective is
a calm reference—preferred cruise motion, fewer crawl/stop episodes, and ordinary-speed
asynchronous arrival at the final pose—not minimum makespan. Mode order, acting/holding robot
rendezvous, and monotonicity remain hard constraints.

Every proposed phase step is checked with the rebuilt motion planner's complete predicate,
including the saved robot-margin gates and the strict `--path-preflight-tolerance`. The complete
schedule is then checked again at bounded joint-path variation and on both sides of every mode
switch. The default `--phase-audit-resolution 0.01` rad matches the motion planner's edge scale;
`--phase-audit-resolution 0` restores the old check at every 2 ms controller sample. A
coordination deadlock falls back to the synchronized path, which is audited too; an invalid
fallback aborts rather than being simulated. Runtime tick-level collision gating is not the
planner. `--phase-schedule` also defaults physical arm-arm contacts on as a backstop. The
MuJoCo masks use one bit per robot, so cross-robot contacts are enabled without the old bug that
also enabled enormous same-arm self-contact forces.

Phase search runs first on a 10 ms coordination grid, then interpolates back to the 2 ms
controller rate. Acceptance uses the configuration-resolution audit above. A counterexample
automatically retries the phase search at 4 ms and 2 ms; all grids are exact multiples of the
controller tick. Audit samples are selected using accumulated maximum-joint variation, not
endpoint distance, so an out-and-back motion cannot be skipped. First/last samples and every
event pose are unconditional. This avoids rechecking long stationary runs without changing the
resolution contract used by motion planning.

Phase speed is measured in each robot's own joint-space arc, rather than the composite arc of
all robots. A robot therefore moves at the requested visual cruise whenever it is actually
moving, and spans where only another robot moves are skipped as zero-duration bookkeeping.
Every configuration produced by those skips remains subject to the same collision predicate
and final configuration-resolution audit.

`--phase-smooth` applies a second, time-domain postprocessor to the resulting per-robot phase
rates. It does **not** fit another joint-space spline: that would cut geometric corners and
invalidate the motion path. Each robot's shared coordination phase is first converted to that
robot's intrinsic accumulated joint-space arc. This is essential when a robot is stationary
over a long span of the composite path: smoothing the composite coordinate would erase the
scheduler's per-arm speed normalization and can amplify physical acceleration dramatically.
The intrinsic non-negative increments are Gaussian-smoothed over `--phase-smooth-time` (default
0.12 s), then each fire-to-fire interval receives a least-change non-negative projection onto
its exact displacement. The complete shared phase vector remains pinned at every mode switch;
actor stops, intentional sharp-corner slowdowns, and rigid co-hold/collision-coupling intervals
remain hard constraints. A per-robot guard rejects any interval that would exceed that robot's
certified source peak rate. The complete result is then
audited at 0.01 rad joint-path resolution in both pre- and post-switch modes;
collision-sensitive intervals revert locally
to their original certified samples. Thus smoothing can reduce acceleration and jerk but can
never become a reason to execute an unaudited shortcut. A collision-free proposal is also
rejected if either acceleration or jerk RMS becomes more than 2% worse than its source.
`--phase-smooth` is strict: if no independent phase trace exists, if the quality guard rejects
the proposal, or if collision repair cannot certify it, the run aborts before physics and no
recording is written. To execute a raw or synchronized schedule, omit `--phase-smooth`
explicitly.

`--phase-batch` is the final, geometry-independent cleanup for short move-stop-move patterns.
It runs after `--phase-smooth`, so it works identically with the default Catmull--Rom geometry
and with `--c2-spline`. For each robot it finds adjacent motion islands separated by at most
`--phase-batch-max-gap` (default 0.35 s). If either island contains less than
`--phase-batch-min-motion` (default 0.08 rad joint-space arc), it tries to pack that afterthought
next to the larger motion as one near-cruise burst. It deliberately leaves one clean stationary
period rather than spreading the same displacement into a long crawl.

An arm's own task stop, a co-grasp/handover equality interval, the global event-passage rule,
and the source peak joint rate remain hard constraints. Crossing another robot's mode switch is
allowed only if the candidate is valid in both the pre- and post-switch worlds. Each local
proposal is checked, followed by the complete configuration-resolution audit. A late counterexample restores only
the connected offending merge windows. The log reports total motion islands, small islands,
and internal slow gaps before and after; acceleration/deceleration at the two outer ends of one
continuous motion are not counted as internal gaps. Mandatory task stops are therefore not
misreported as removable visual defects.

`--phase-intent` is a separate experimental alternative to `--phase-batch` for the stronger
visual objective “move decisively, then be clearly idle.” It approximately minimizes the number
of control samples on which any robot is moving, with fixed duration and event times. For each
robot, its own task events partition time into robot-owned intervals. The complete intrinsic
joint-space path arc in each eligible interval is packed into one delayed motion with 150 ms
quintic velocity ramps and a peak path rate no greater than `--phase-intent-rate` (default
1.5 rad/s). Using intrinsic rather than shared composite arc removes bookkeeping stops where only
another robot moves. Several temporal placements are tried.

Only the robot's own events are rest anchors. A foreign mode switch is a passage deadline and a
pre/post-mode collision constraint, so the robot can cross it at nonzero velocity. True rigid
co-hold/handover spans stay byte-exact. Scheduler-created collision couplings are not physical
equalities: intent is allowed to explore them because its candidates are collision-screened and
the complete result receives the 0.01 rad pre/post-switch audit. Monotonicity and every event
witness remain exact.

Pre-skill GOTO tasks are geometric passage waypoints, not rest events: they change no attachment
and execute no skill hook, so their old task timestamp is not a hard phase deadline. However, an
own-event interval containing such a waypoint is a contact-skill funnel: the already continuous
phase-smoothed timing is retained there because it also encodes coordinated holder timing and the
terminal low-rate approach expected by the seat servo. `--phase-intent` therefore compacts free
transport and does not independently repack insertion/grasp funnels. For each eligible interval, several
continuous durations are considered. A candidate may not increase either total all-robot idle or
the squared lengths of its idle runs; among the survivors the shortest motion is placed latest.
A second coordinate-refinement pass shifts earlier robots later after the other robots' compact
motions are known. Thus maximizing individual idle cannot create a longer scene in which nothing
moves.

This objective is intentionally different from minimum makespan and minimum jerk. In practice,
the ramped bursts can improve derivatives too, but that is measured rather than assumed.
`--phase-intent` is fail-closed: if no certified motion-support reduction survives, simulation
does not start. It is mutually exclusive with `--phase-batch`, but normally follows
`--phase-smooth`. Compact timing explores farther from the synchronized diagonal, so a narrow
coordination corridor naturally rejects many placements and limits the achievable idle gain.

Current saved-path validation commands (no IK or motion replanning) are:

```bash
python3 sim.py run experiments/car-mini-ur10-new.json \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-phase-intent-rad-audit.npz

python3 sim.py run experiments/chair-no-vacuum-assemble.json \
    --mp-path experiments/mp-chair-no-vacuum.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/chair-phase-intent-rad-audit.npz
```

The car completed 19/19 actions in 4,423 replay frames; the chair completed 34/34 in
9,172 frames. The car audit retained its known 10 ms-grid collision counterexample and accepted
the 4 ms retry. Final-audit collision calls fell from 34,840 to 3,798 on car and from 72,635 to
9,271 on chair. Use `python3 sim.py replay FILE.npz` to inspect either recording.

`--phase-servo-time SEC` is the fixed nominal pre-fire reservation used while forming the
offline phase schedule (default 0.15 s per settle or seat operation, matching the measured
car-mini mean rather than its 1 s worst-case cap). It is a guess made before execution, never
an EMA or per-tick phase correction. At the event, convergence is checked only for its acting
robots and holders; unrelated robots are intentionally moving and their tracking residual must
not create a global stop. The old offline velocity-funnel/wrench loop is not applied underneath
a phase schedule: it used the minimum factor as a second global clock and froze every robot
when one placed part touched the table. The actor still gets its bounded compliant offset,
event-owned settle/seat hooks, physical arm contacts, and the grasp breakaway monitor; the
other independently scheduled arms keep flowing. A genuine event-servo overrun can still hold
at that event, but no longer turns every contact tick into another global timeout.

The actuated settle gate now defaults to `--settle-tol 0.002`. Car-mini's loaded a3 servo has
a repeatable approximately 1.9 mrad steady-state residual; the former 1 mrad threshold waited
the complete `--settle-max 1.0` s without reducing it. Two mrad is about 2 mm at a one-metre
reach and matches the terminal seat-servo scale, while remaining far below the transient errors
that the gate is intended to keep out of a newly activated weld. More generally, a residual
below twice the requested tolerance that improves by less than one quarter of that tolerance
over 0.1 s is recognized as the finite-gain servo's steady-state plateau; large or still-moving
errors retain the normal gate and one-second safety bound.

Current car-mini validation (saved plan/path only; no IK or motion replanning):

```bash
python3 sim.py run experiments/car-mini-ur10-new.json \
    --mp-path experiments/car-mini-ur-10-new-mp.json \
    --phase-schedule --phase-smooth --phase-batch --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/car-mini-ur10-new-phase-batched.npz

python3 sim.py replay experiments/car-mini-ur10-new-phase-batched.npz
```

This completed 19/19 actions. With the established C1 geometry, batching kept 52 certified
merges and reduced motion islands from 207 to 155, sub-0.08-rad islands from 127 to 73, and
internal slow gaps from 700 to 620. The
phase reference has 37,976 control samples plus 34
post-switch poses, all accepted by the exact final audit; recording every eighth physics tick
produced 4,778 NPZ frames. The duration is printed only for context. Motion-quality reporting
measures each robot from its first motion through its own final arrival and reports the
harmless terminal idle tail
separately, so an early-finishing arm is not encouraged to creep merely to end in synchrony.
Car-mini's tight coordination and rigid-coupling windows leave relatively little freedom for
postprocessing: reference acceleration/jerk RMS fell by 1.5% and the recorded physical values
by approximately 0.8%, with the physical peak joint-space rate unchanged.

With `--c2-spline`, the current run kept 26/28 proposed merges after two local audit
restorations: motion islands fell 144 to 118, sub-0.08-rad islands 76 to 49, and internal slow
gaps 373 to 332. It also completed 19/19 actions. Use the earlier C2 command to reproduce or
replay `experiments/car-mini-ur10-new-c2-phase-batched.npz`.

Current peg-stool chamfer evaluation uses the supplied saved plan and path directly. Both the
established C1 geometry and opt-in C2 geometry now produce strict smoothed rollouts:

```bash
# Established C1 geometry
python3 sim.py run experiments/peg-stool-4r-new-plan.json \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-batch --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-phase-smoothed.npz

# C2 geometry: strict, collision-audited intrinsic-arc smoothing
python3 sim.py run experiments/peg-stool-4r-new-plan.json \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-batch --c2-spline --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-c2-phase-smoothed.npz

python3 sim.py replay experiments/peg-stool-4r-new-phase-smoothed.npz
python3 sim.py replay experiments/peg-stool-4r-new-c2-phase-smoothed.npz
```

For C1, the 45,285-sample asynchronous reference and smoothed result pass the exact 2 ms
collision audit. Intrinsic-arc smoothing reduces joint acceleration RMS by 26.7% and jerk RMS
by 26.8%. Batching keeps 23/23 certified merges: motion islands fall 160 to 137,
sub-0.08-rad islands 78 to 53, and internal slow gaps 459 to 435. Simulation completes 21/21
actions and records 5,735 frames.

An experimental intentional-motion rollout replaces `--phase-batch` with `--phase-intent`:

```bash
python3 sim.py run experiments/peg-stool-4r-new-plan.json \
    --mp-path experiments/mp-peg-stool-chamfer.json \
    --phase-schedule --phase-smooth --phase-intent --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/peg-stool-4r-new-phase-intent-v6.npz

python3 sim.py replay experiments/peg-stool-4r-new-phase-intent-v6.npz
```

At the default 1.5 rad/s intent rate, moving support falls from 67,590 to 62,701 robot-samples
(idle fraction 59.1% to 62.1%), motion islands fall 171 to 121, and internal slow gaps fall 512
to 363. Conditional moving speed rises from 0.68 to 0.73 rad/s. All-robot idle falls from 9.9 s
to 7.8 s and its longest interval from 1.4 s to 1.2 s. The 150 ms quintic ramps reduce
acceleration RMS by another 16.0% and jerk RMS by 16.7% relative to the already smoothed input.
The actuated recording peaks around 1.72 rad/s instead of the former approximately 2.4 rad/s.
All 13 accepted robot-owned intervals and the complete trajectory pass the exact audit; four
intervals retain certified smoothed timing after no alternative can be certified, 19 intersect a
true rigid span, and the coordinate-refinement pass shifts three intervals later. The wider
foreign-stop diagnostic falls from 11 to 9. Simulation completes 21/21 actions and records 5,281
frames.

For C2, 14 stretches fit, 15 closed-chain stretches remain protected, and 3 unsafe fits retain
their certified polylines. Its 45,575-sample asynchronous reference and smoothed result also
pass the exact audit. Smoothing reduces acceleration RMS by 20.4% and jerk RMS by 20.5%.
Batching keeps 17/17 certified merges, reducing motion islands 142 to 125, sub-0.08-rad islands
63 to 44, and internal slow gaps 282 to 259. Simulation completes 21/21 actions and records
5,772 frames.

This artifact-specific Chair run validates phase scheduling and intrinsic smoothing on a longer
mixed-EE UR5 plan. It predates the canonical batching command above (the historical filenames
say "no-vacuum", but the saved config contains two vacuum and two two-finger grippers):

```bash
python3 sim.py run experiments/chair-no-vacuum-assemble.json \
    --mp-path experiments/mp-chair-no-vacuum.json \
    --phase-schedule --phase-smooth --limit-rate \
    --servo-torque-guard 0.6 \
    --record experiments/chair-no-vacuum-phase-smooth.npz

python3 sim.py replay experiments/chair-no-vacuum-phase-smooth.npz
```

This completed 34/34 actions. The first 10 ms phase search succeeded, and all 79,060 expanded
2 ms samples plus 60 post-switch poses passed the exact audit. The recording has 9,932 frames.
Three small finite-gain residuals plateaued at up to 3.3 mrad; plateau detection reduced total
settling from 3.002 s (three full timeouts) to 0.306 s while preserving the successful outcome.
In the previously conspicuous `obj_1` handover-to-table-place interval, the repeated global
force-guard freeze is gone: the longest all-robot stop fell from roughly 9.5 seconds to 0.53
seconds, while the whole interval now takes 7.41 seconds and still completes successfully.
The phase smoother required no collision-driven interval fallback on Chair. It reduced reference
acceleration/jerk RMS by 14.5%/13.4% and recorded physical acceleration/jerk RMS by
13.5%/16.0%, without increasing the recorded peak joint-space rate or changing the placement
pause above.

`--limit-rate` is deliberately ignored by this scheduler: it keeps the visual cruise at
`--joint-rate` (1 rad/s by default) rather than racing the UR joint limits. The existing
`--force-limits` tables are computed and reported but are not yet consumed by the phase
scheduler; the runtime normalized wrench guard and break monitor remain active. This is the
next safety-model integration, not a hidden claim.

The older `--corridor-online --pace` artifacts are retained only for debugging. Their earlier
“19/19” result was not a collision-free validation: arm-arm physics was masked, and the sparse
three-tick preventive gate could advance unchecked from an already colliding state. In the
recorded paced car run, two vacuum robots overlap by roughly 9 cm near frame 3544. Do not use
that command or NPZ as the current baseline.

## Peg-in-hole examples

### Atomic screw-in skill

`peg-hole-screw` is the first helical assembly example.  Its symbolic action is still one
ordinary `assemble(robot, screw, socket)` action: the task planner chooses one grasp and that
same grasp must remain feasible from the helical entry pose through the seated pose.  It never
inserts release/regrasp actions into the task plan.  If a future multi-turn skill needs finger
rewinding, that is an execution-controller implementation hidden inside the same atomic skill,
not a larger symbolic skeleton.

The scene is intentionally a virtual-thread test.  It has a real concave socket and a raised
asymmetric screw head so the rotation is visible, but no contact thread mesh.  The skill
prescribes a constant-pitch helix and the wrench model applies the axial
insertion load plus a pure screw-torque couple.  This isolates planning and control from thread
contact simulation.  The reproducible UR10 experiment lets table friction react a modest
0.05 Nm torque:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --search-profile standard --assembly peg-hole-screw --parts 2 \
    --num-robots 2 --robot ur10_limited \
    --grippers two_finger,two_finger \
    --table-friction 0.5 --screw-torque 0.05 --seed 11023 \
    --no-grasp-widen --no-view \
    --save experiments/peg-hole-screw-ur10-2r-standard-plan.json
```

The resulting task plan is exactly `pick`, `assemble`, `release`.  The saved witness block also
contains the complete 17-state, single-grasp helix certificate.  Loading the plan therefore
does not rerun screw IK:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/peg-hole-screw-ur10-2r-standard-plan.json \
    --load-plan experiments/peg-hole-screw-ur10-2r-standard-plan.json \
    --motion-profile standard --motion-plan --mp-runtime 90 --mp-plan-candidates 1 \
    --mp-save experiments/mp-peg-hole-screw-ur10-2r-standard.json \
    --no-view --quiet-rai

python3 sim.py run experiments/peg-hole-screw-ur10-2r-standard-plan.json \
    --mp-path experiments/mp-peg-hole-screw-ur10-2r-standard.json \
    --record experiments/peg-hole-screw-ur10-2r-standard.npz

python3 sim.py replay experiments/peg-hole-screw-ur10-2r-standard.npz
```

`--screw-angle DEG`, `--screw-depth M`, `--screw-torque NM`, `--screw-speed RAD_S`, and
`--screw-samples N` override the scene specification.  The final mate remains the authored
assembly datum; entry is backed off by `depth` and counter-rotated by `angle`.  Intermediate
poses are warm-started in reverse from the accepted seated witness, branch-continuity gated,
collision checked, and persisted.  In simulation the skill owns only its assemble stretch:
the free-space planner segment is replaced by this certificate, strict path preflight runs,
and the ordinary smooth retimer caps its helical cruise at `screw-speed`.  All other motion
continues to use the standard simulation profile unchanged.

`peg-stool-4-screw` is a separate five-part environment; none of the existing peg-stool
scenes are changed.  Four round 19mm shafts each execute a 180-degree, 20mm helix into a
chamfered 30mm square socket, leaving 5.5mm radial clearance.  The 25mm socket depth leaves a
5mm blind-end gap, so the authored 20mm helix starts at the mouth instead of beginning part-way
inside the socket.  The screw scene's 110x85mm socket rectangle also leaves a real corridor for
the rotating gripper; ordinary stool scenes retain their 110x70mm grid.  A small eccentric,
yellow, contact-disabled indicator near each exposed foot makes the rotation visible without
changing collision, grasp, wrench, or mass geometry.  A normal handover may reorient a horizontally
scattered leg *before* insertion—one robot cannot realistically perform both that pickup and
the vertical screw pose—but there is no release, symbolic regrasp, or hand change between a
screw's entry and seated keyframes.  Any future finger rewind for a multi-turn screw belongs
inside the execution skill, not in the task skeleton.

The tested setup uses one vacuum arm for the fixture and two finger arms for leg transfer and
screwing.  Its capability scales are explicit, persisted physics parameters: x5 fingers cover
the measured handover transient, while x2 vacuum covers the off-centre completed stool during
the final set-down.  The new scene alone uses a 0.50kg seat; the stock 0.75kg press-stool mass
is retained everywhere else.

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --search-profile standard --assembly peg-stool-4-screw --parts 5 \
    --num-robots 3 --robot ur5_limited \
    --grippers vacuum,two_finger,two_finger \
    --allow-curved-grasps --vacuum-no-assemble \
    --table-friction 0.5 --insert-force 0.1 --screw-torque 0.05 \
    --vacuum-scale 2 --two-finger-scale 5 --seed 11023 \
    --save experiments/peg-stool-4-screw-ur5-3r-round-plan.json \
    --quiet-rai --no-view

PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/peg-stool-4-screw-ur5-3r-round-plan.json \
    --load-plan experiments/peg-stool-4-screw-ur5-3r-round-plan.json \
    --motion-profile standard --motion-plan --mp-runtime 30 \
    --mp-seed 11024 --mp-margin-ramp 0.4 \
    --mp-save experiments/mp-peg-stool-4-screw-ur5-3r-round.json \
    --quiet-rai --no-view

python3 sim.py run experiments/peg-stool-4-screw-ur5-3r-round-plan.json \
    --mp-path experiments/mp-peg-stool-4-screw-ur5-3r-round.json \
    --record experiments/peg-stool-4-screw-ur5-3r-round.npz

python3 sim.py replay experiments/peg-stool-4-screw-ur5-3r-round.npz
```

By default, MuJoCo only sees a mate reaction when the insertion geometry actually makes
contact.  The virtual screw and weld-on-seat abstractions can therefore validate the kinematics
without transmitting the process load into the holders.  Add `--apply-process-wrench` for the
physical support test:

```bash
python3 sim.py run experiments/peg-stool-4-screw-ur5-3r-round-held-plan.json \
    --mp-path experiments/mp-peg-stool-4-screw-ur5-3r-round-held.json \
    --apply-process-wrench \
    --record experiments/peg-stool-4-screw-ur5-3r-round-held-loaded.npz
```

The flag is intentionally opt-in while the force capabilities are being calibrated.  It applies
the planner's nominal axial insertion force as an equal-and-opposite internal pair at one physical
interface point.  An ordinary press engages smoothly over the final 10mm by default, rather than
applying a fictitious long-range force throughout the collision-free approach.  A screw receives
its declared pure torque throughout the certified helix.  All loads ramp on over 50ms to avoid a
one-tick impulse.  Forces are converted to body-COM wrenches with the correct interface lever arm,
so receiving-part holders, inserted-part grasps, and actuated arms must react the load even when
the idealized mate geometry generates no collision contact.  Pre-engagement and post-mate motion
are unloaded.  Override the physical envelope with `--process-engagement-distance` and
`--process-wrench-ramp` when the authored skill provides better measurements.

The saved `--table-friction` is also the simulation coefficient.  A missing value retains the
historical MuJoCo bench (`mu=0.5`); an explicit low value scales sliding, torsional and rolling
friction together.  Thus `--table-friction 0.02` is a useful nearly-frictionless experiment
without the numerical idealization of exactly zero, while `0` really is frictionless.  Previously
the simulator silently used `0.5` regardless of the saved value and retained torsional friction at
zero; recordings made before this fix do not establish a frictionless counterfactual.

`--insertion-noise` is a robust set, not intrinsically a probability distribution.  Simulation
therefore applies no tick-wise white noise.  Select one fixed mate-frame realization per assembly
action explicitly:

```bash
# One exact robust-box vertex (four lateral-only vertices; sixteen with insertion offset).
python3 sim.py run PLAN.json --mp-path MP.json \
    --apply-process-wrench --process-noise corner --process-noise-corner 0 \
    --record corner-0.npz

# One reproducible uniform draw inside the box per assembly action.
python3 sim.py run PLAN.json --mp-path MP.json \
    --apply-process-wrench --process-noise sample --process-noise-seed 11023 \
    --record sample-11023.npz
```

The NPZ metadata stores the mode, seed/corner and all realized coefficients.  For a robust-set
claim, sweep every corner; seeded samples are appropriate for an empirical success-rate study.
The lateral coefficients multiply `insertion_noise * insert_force` along the two transverse axes;
the optional last two coefficients shift the contact point by `insertion_offset`.

Initial object-placement uncertainty is a separate, opt-in experiment. It samples one rigid
planar displacement per initially disconnected table-supported group, preserves the authored
support face and height, and rejects the complete draw unless the initial RAI scene is collision
free:

```bash
python3 sim.py run PLAN.json --mp-path MP.json \
    --object-perturb-xy 0.005 --object-perturb-yaw 3 \
    --object-perturb-seed 11023 --object-perturb-observation observed \
    --report-json noisy-11023.json --record noisy-11023.npz
```

`observed` models a pose estimate available to the controller: immediately before each pick, a
bounded 6-D task-space servo applies the measured part displacement to the exact saved gripper
anchor pose, and the terminal mate servo follows the displaced parent's full datum pose. It does
not replan the path or reconstruct the grasp transform, so robot-specific EE calibration remains
intact. `unobserved` does not expose the initial displacement to this retargeter; the standard
skill's existing contact/admittance feedback remains active. In both modes, the grasp weld is
activated only within `--object-grasp-capture` (10mm) and
`--object-grasp-capture-angle` (10deg); a larger miss is reported as `not_grasped`, rather than
teleporting the object into the hand. Physical collisions remain active during the correction.
This is an execution robustness test, not a claim that the nominal motion-path certificate covers
the offset trajectory.

Run a sequential, resource-guarded Monte Carlo matrix with:

```bash
python3 benchmarks/run_noisy_sim.py \
    --plan PLAN.json --mp-path MP.json --output benchmarks/results/noisy-example \
    --seeds 0 1 2 3 4 --xy 0.002 0.005 --yaw-deg 1 3 --observation both \
    --apply-process-wrench --process-noise sample
```

Each cell runs in a fresh process with a 4GiB/900s guard by default. The runner checkpoints
`summary.json` and `summary.csv`, retains all failures plus one representative success per cell,
and stores the simulator's full per-action report and perturbation realization. Use
`--keep-recordings all|failures|none`, `--memory-gb`, and `--timeout` to change experiment
hygiene; add controlled simulator overrides after a literal `--`.

The example's 0.05Nm torque is intentionally an easy-start demonstration load.  It is comparable
to a tiny M1.6 fastener, not a 19mm furniture-leg tightening torque.  Keep it for controller
debugging; use roughly 0.25--0.5Nm for a modest plastic/low-preload fastening experiment, and
calibrate a real threaded leg from measured running and seating torque rather than shaft diameter.

For a matched geometry-only counterfactual, generate the same plan with `--no-wrench` and leave
all physical role restrictions unchanged.  In particular, `--vacuum-no-assemble` remains active:
`--no-wrench` disables equilibrium/capacity feasibility, not the declared set of tools allowed to
perform an assembly action.  Then run that saved plan with the same motion and simulation profiles
and add `--apply-process-wrench`.  This deliberately asks whether a skeleton selected without the
load model survives the load in dynamics; simulation failure is a valid experimental outcome, and
the standard profile records the failed NPZ before returning a non-zero status.

The initial UR5 car-mini 10N comparison was additionally contaminated by applying the load over
the entire pre-insertion approach.  Rerun old counterfactual recordings with the engagement-gated
implementation above. MuJoCo already distributes a connected body's physical reaction over all
simultaneously active grasp welds and table contacts; the simulator must not run a second LP and
redistribute that measured reaction synthetically. Before the mate weld exists, however, the
target and inserter are two bodies, and a lone holder on either side must carry that side's load.

The wrench-aware plan can still expose two separate model discrepancies. Search checks static group
equilibrium with the full directional grasp polytope, while simulation's default breakaway guard
uses conservative scalar force/torque caps per grasp. `--directional-caps` makes the break monitor
use the planner's component-wise box. More fundamentally, the discrete wrench gate currently uses
the state's stored placement orientation; a free-space IK witness may rotate a held component
without feeding that realized pose back into the LP. In the current car action 14, the target has
only one holder. The symbolic LP predicts `T_close=0.238Nm` and `T_lat=0.297Nm` against `0.400Nm`
component caps. At the realized near-engagement orientation, gravity raises the corresponding
quasi-static closing-axis demand to about `0.340Nm`, and tracking/inertial load takes MuJoCo to
`0.41Nm`. Thus this case is not incorrect LP load allocation: it is a missing witness-conditioned
wrench recheck followed by missing dynamic reserve. Report support selection, cap model, realized
pose, and dynamic reserve separately until those two checks and the gripper calibration are fixed.

For a declared Robotiq 2F-85 hardware-ceiling experiment, add the orthogonal profile:

```bash
python3 main.py --solve --search-profile standard \
    --wrench-profile max_robotiq \
    --assembly cross --parts 7 --num-robots 2 --robot husky \
    --seed 11023 --save experiments/husky-cross-max-robotiq-plan.json
```

`--wrench-profile max_robotiq` replaces only the two-finger capability caps: 50 N in each
force direction, 5 Nm about grasp-local x/y, and 3 Nm about grasp-local z (the approach axis).
The profile is persisted in the plan, so replay, motion export, and simulation restore it
without repeating the flag. It leaves vacuum capabilities and all search settings unchanged;
`--two-finger-scale` and `--tf-close-torque` remain explicit post-profile overrides.
Simulation automatically selects the component-wise breakaway monitor for this anisotropic
profile; `--no-directional-caps` is the explicit conservative scalar-cap ablation. Historical
plans and provisional caps retain their previous scalar-monitor default.

These are the manufacturer's **absolute moment and force limits**, not measured object-slip
limits or a safety-rated operating envelope. The manual requires acceleration and an
integrator-selected safety factor to be included. Moreover, Mx/My are specified about the
fingertip base; the present grasp box applies the corresponding free-couple components at the
modeled grasp datum and does not yet couple fingertip-base moment to the force lever arm. Thus
the profile is a more realistic upper ceiling for experiments, while the later pad/object
calibration must supply working caps and a safety factor. For the isotropic-model ablation,
which cannot represent axis-dependent moments, the profile conservatively uses 3 Nm on every
moment axis.

The opt-in search-side alternative is `--orientation-robust-wrench`. For every freely held
component, it requires the current contact set to support gravity at **every** rigid-body
orientation. It does not enumerate poses or call IK. In the component frame, all orientations
are the unit sphere of gravity directions; its convex hull is a ball. The implementation checks
exact containment of that gravity-wrench ellipsoid in the bounded contact-wrench zonotope using
the zonotope's cached 6-D facet normals. Multiple contacts may choose a different feasible load
allocation for each orientation, matching the existing fully-adjustable robust insertion model.
Insertion-noise corners are crossed with the continuous gravity set, so each corner must survive
every orientation. Table-supported components are exempt because their discrete rest pose fixes
their orientation.

```bash
python3 main.py --solve --search-profile standard \
    --assembly car-mini --parts 5 --num-robots 4 --robot ur5_limited \
    --orientation-robust-wrench --seed 11023 \
    --save experiments/car-mini-orientation-robust-plan.json
```

This is deliberately not part of `standard`: it is sound but can be substantially conservative,
for example forcing an extra brace even when a real controller would keep a one-grasp carry
upright. `--ignore-gravity` makes the orientation uncertainty empty and therefore supersedes this
flag. The check is a cheap symbolic/contact test; IK remains downstream and is never run merely to
reject a gravity orientation.

This motion artifact retains the standard 100mm robot margin but uses a 0.4m participant ramp.
The longer ramp matches the closing geometry of the intentionally near-contact vacuum-to-finger
handovers; strict penetration, composed screw-path, and 0.002rad full-path audits remain enabled.

For a screw stretch, unrelated robots may already have useful motion in the saved composite
path.  Simulation preserves that planner polyline and replaces only the screw task's joints
with the persisted helix.  The union of both paths' knots is then checked at the standard
0.002rad resolution in the exact pre-attach mode.  An unsafe composition fails preflight;
there is no fallback to an ordinary insertion or an unsmoothed path.

#### Imported real stool

`real-stool` is the five-part assembly imported from `part_meshes/260904_YandV_obj_v2`;
`real-bench` is its three-part, two-hole variant. The old `yandv-stool` and `yandv-bench`
spellings remain aliases for saved-plan compatibility. The exact meshes are display-only, while
planning and simulation share a non-overlapping compound collision model: a 35mm base plate
with 25mm blind pockets, the authored 40mm rear upstand, and 30x22x105mm legs. The four leg
bodies use their distinct LF/RF/LB/RB CAD meshes in MuJoCo; they are not four copies of LF.
The physical model is deliberately lightweight: the seat is 100g and each leg is 50g; display
meshes are inert, so those values are identical in the wrench model and MuJoCo.
Regenerate the deterministic centred, metre-scale meshes after editing the source OBJ files:

```bash
python3 part_meshes/yandv_stool/make_visual_meshes.py
python3 tools/visualization/view_yandv.py --assembly real-stool --backend mujoco --assembled
```

#### Exporting assembly parts

`tools/exporters/export_assembly_parts.py` exports the authored geometry of any built-in assembly
as individual
part files, using the same convention as the earlier chamfered-stool export: GLB in metres and
STL in millimetres. Compound GLBs retain separately named child solids. A manifest records the
object IDs, bounds, mate parents, and assembled part-frame positions; a README and zip archive
are generated alongside the meshes.

```bash
python3 tools/exporters/export_assembly_parts.py --assembly cross
python3 tools/exporters/export_assembly_parts.py --assembly cube
python3 tools/exporters/export_assembly_parts.py --assembly chair
```

The default destination is `exports/<assembly>/`, with `exports/<assembly>.zip` as the portable
archive. Use `--parts N`, `--output DIR`, or `--no-zip` when needed. Collision-only rounded-box
corner sweeps are deliberately represented by their nominal sharp boxes, matching the previous
stool export and the geometry used for grasp sampling.

`--mate-symmetry` is an opt-in discrete mate-goal set. For every real-stool leg, the one
symbolic `assemble` edge tries the two exact collision-symmetry members (0 and 180 degrees
about the leg's long axis) as fixed IK targets. The rectangular 30x22mm leg is not invariant
under a quarter turn, so 90/270 degrees are deliberately excluded. No member label is stored:
the already-required robot witness contains the achieved configuration, while every later
symbolic scene may use the canonical representative because the declared members are physically
identical. This does not create task states or ask KOMO to optimize a free angle. The cheap
on-table mate fact is existential over the same finite set, and MuJoCo derives its weld from the
achieved body pose. The flag is off by default, including for older saved plans.

#### Finite internal mate wrenches

An assembly may give a child-to-parent mate a finite, bilateral component box through
`Assembly.mate_wrench_limits`:

```python
self.mate_wrench_limits["obj_1"] = {
    "force": [Fx, Fy, Fz],       # N in the parent mate frame
    "torque": [Mx, My, Mz],      # Nm in the parent mate frame
}
```

With `--mate-wrench-limits` (the default for newly generated plans), the wrench gate no longer
treats such a connected assembly as one infinitely rigid body. It contracts only the *uncapped*
mates into rigid supernodes, writes one six-dimensional equilibrium equation for every resulting
body, and connects the two sides of each capped mate with equal-and-opposite bounded wrench
variables. Gravity, table contacts, robot grasps, and the assembly process wrench are assigned to
the body on which they physically act. This is still one polyhedral feasibility LP; it does not
add a post-IK rejection or a new search action. Assemblies with no finite-mate metadata take the
literal former single-rigid-body code path.

The real-stool legs currently use a deliberately low demonstrator box of 1N per force component
and 0.03Nm per moment component. This makes the search retain or reacquire support on the seat
during a 10N insertion instead of transmitting that load through an ideal leg joint. Run the
matched ideal-weld baseline with `--no-mate-wrench-limits`. The switch and authored values are
persisted in new plan JSON files; legacy plans load with the old ideal-weld behavior.

```bash
# Finite-mate model (standard search enables it, because real-stool declares caps).
python3 main.py --solve --search-profile standard --wrench-profile max_robotiq \
    --assembly real-stool --parts 5 --num-robots 2 --robot husky \
    --insert-force 10 --mate-symmetry --seed 11023 \
    --save experiments/real-stool-husky-finite-mates-plan.json

# Exact baseline: same assembly and load, but every completed mate is an ideal weld.
python3 main.py --solve --search-profile standard --wrench-profile max_robotiq \
    --assembly real-stool --parts 5 --num-robots 2 --robot husky \
    --insert-force 10 --mate-symmetry --no-mate-wrench-limits --seed 11023 \
    --save experiments/real-stool-husky-ideal-mates-baseline-plan.json
```

Finite mates presently support the fixed-orientation LP and nominal/corner process loads. They
fail closed with a clear error for `--orientation-robust-wrench` and affine robust noise, rather
than silently falling back to the ideal-weld approximation. Simulation's scalar
`--mate-force-cap`/`--mate-torque-cap` breakaway weld is a separate dynamics diagnostic and is not
the planner's per-axis internal LP. The reference replay therefore applies the persisted process
wrench while the search certificate remains authoritative for internal transmission.

The calibrated Husky has two Robotiq grippers. Its provisional default slip caps cannot support
the first loaded insertion in this assembly, so a wrench-on hardware-envelope run must name the
documented `max_robotiq` profile (this is an absolute manufacturer ceiling, not the pending slip
calibration):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --search-profile standard --wrench-profile max_robotiq \
    --assembly real-stool --parts 5 --num-robots 2 --robot husky \
    --seed 11023 --save experiments/real-stool-husky/plan.json
```

The reference run produced a 20-action plan with 16 persisted exact witnesses and three witness
sets: 11.4s to the initial plan and 13.1s for witness-set/floor optimization. For geometry and
reachability diagnosis only, replace `--wrench-profile max_robotiq` with `--no-wrench`; that is
not an equivalent force-aware result.

The saved wrench-aware motion plan and successful standard-profile dynamics recording can be
inspected independently:

```bash
# Scrub the geometric motion plan in Viser (no IK or motion planning is rerun).
python3 tools/visualization/mp_viser_saved.py experiments/real-stool-husky/plan.json \
    experiments/real-stool-husky/mp.json 8080

# Replay the recorded MuJoCo execution (no controller or planning is rerun).
python3 sim.py replay experiments/real-stool-husky/replay-standard.npz
```

The NPZ was generated with the standard execution contract and persisted motion path. Exact CAD
meshes are renderer-only: their MuJoCo mass is explicitly zero, while the shared simplified
collision geoms carry the masses authored by the assembly model.

#### Fabrica benchmark imports

The `fabrica-*` environments import Car, Beam, Cooling Manifold, Duct, Gamepad, Plumbers
Block, and Circular Stool from the neighbouring Fabrica repository. Fabrica stores every OBJ
in centimetres and directly in a shared assembled frame. The deterministic importer converts
to metres, recentres each part, and records source hashes, centres, and decomposition settings
in a manifest. Exact meshes are used for rendering and grasp sampling. Search, motion planning,
and MuJoCo collision share the named convex components under one rigid part body. The
`tetra-inner` importer checks these cells against the original CAD so they do not bridge
holes. Earlier VHACD assets did bridge mating holes; their successful centre-point tests
were insufficient. See [collision import and validation](../part_meshes/fabrica/README.md)
for the method, fidelity/cost tradeoff, and each manifest's representation metadata.

Fabrica Car source `2.obj` is a fixture. It is retained only in the imported source cache for
auditability and is **not** an object in either the scattered scene or final assembly. The
environment therefore contains five parts: source body `1.obj` and wheels `0/3/4/5.obj`.
Beam contains all five authored members (`6/2/0/3/1.obj`) and has no fixture.
Its three mortised members use reviewed 7/10/10-box collision unions rather than 389 generic
tetrahedral cells; the inserted uprights retain three semantic children each. Exact meshes are
still used for rendering and grasp sampling, and the box openings pass the full RAI/MuJoCo
insertion audit with a +/-1 mm seated-pose envelope.

```bash
# Five-member Beam, standard search and force model.
python3 main.py --solve --search-profile standard \
    --assembly fabrica-beam --parts 5 --num-robots 2 --robot ur5_limited \
    --grippers two_finger,two_finger --seed 11023 \
    --save experiments/fabrica-beam-ur5-2r-standard-plan.json

# Five-part Car. Fabrica provides no insertion-force metadata, so this command states the
# benchmark condition explicitly; the generic 10 N default is a different bracing stress test.
python3 main.py --solve --search-profile standard \
    --assembly fabrica-car --parts 5 --num-robots 4 --robot ur5_limited \
    --grippers vacuum,two_finger,two_finger,vacuum \
    --insert-force 0.1 --seed 11023 \
    --save experiments/fabrica-car-ur5-4r-standard-small-force-plan.json

# Reproduce all local geometry from a neighbouring Fabrica checkout.
python3 part_meshes/fabrica/import_assets.py \
    --source ../../Fabrica/assets/fabrica

# Inspect collision (group 2) versus exact visuals (group 3).
python3 tools/visualization/view_fabrica.py duct --pose assembled
python3 tools/visualization/view_fabrica.py car --pose assembled

# Check complete insertion sweeps in RAI and MuJoCo and containment in the original CAD.
python3 tools/diagnostics/audit_fabrica.py duct --containment
python3 tools/diagnostics/audit_fabrica.py car --containment
```

The importer is generic, but assembly membership and mate directions/dependencies remain
explicit annotations: those semantics are not present in Fabrica's OBJ files. Collision is
derived from the meshes rather than annotated per model. Car
and Beam have individually reviewed trees. The other five currently use connected spanning
trees inferred from final-pose proximity and are intentionally easy to inspect in
`FabricaAssembly._SOURCE_ORDER/_PARENT`. Tests cover collision-free scattered scenes, authored
mate-opening topology, full peg/pin extents, and RAI/MuJoCo rigid-body grouping. In the MuJoCo
viewer, group 2 is the decomposed collision model and group 3 is the exact visual mesh.

Very light loose fasteners can accumulate millimetres of finite-precision table-contact drift
over exceptionally long lockstep recordings. `sim.py run --object-damping 0.001` adds explicit
viscous damping to free part joints for that diagnostic; its default is zero, and it changes
neither the saved plan/path nor collision or wrench certification. Record the value alongside
the replay whenever it is used.

`peg-hole` is the minimal two-part concavity check. `peg-stool` is the useful follow-up: its
seat is a plain plate containing three blind square sockets, and three round legs are inserted
24.5 mm into those sockets. There is no raised vacuum pedestal/pad on top of the seat. The seat
is one 13-child compound rigid body (plate plus 12 socket walls), so use the sampled grasp
database. The stool needs a mixed fleet: the vacuum gripper holds the broad seat while a
two-finger gripper handles the cylindrical legs.

Milestones do **not** require placement to be disabled. `--milestones` only selects the search
algorithm; placement successors remain enabled by default unless `--no-place` is explicitly
passed. The successful stool plan below uses both milestones and placement.

`--vacuum-engage-force N` covers the load that exists while a cup approaches and seals. With
`--wrench`, every vacuum `pick`, `add_grasp`, and fused `handover` first asks whether the
supports which exist *before* the new grasp can balance an `N`-newton push along the cup's
approach axis at the contact point. The new cup is deliberately absent from that LP, so it
cannot balance its own engagement force. This is separate from the steady attached-grasp
capability: after sealing, the cup preload is an internal balanced load and must not be added
permanently to gravity/insertion wrenches. Default `0` preserves old plans; the stool run uses
5 N. Two-finger grasps are unaffected.

The reproducible wrench-enabled stool search is:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --assembly peg-stool --parts 4 --num-robots 2 --robot ur10_limited \
    --grippers vacuum,two_finger --weight inf --beam 16 --milestones \
    --scatter-spacing 0.2 --sig-collapsed-h --sig-components press --real-ik \
    --wrench --vacuum-engage-force 5 --insert-force 20 --grasp-model asymmetric \
    --grasp-sampler --grasp-sampler-keep 12 --allow-curved-grasps \
    --bracing --seed 11023 --optimize-witnesses floor \
    --witness-margin 0.005 --witness-branch-budget 5 \
    --fuse-cograsp --tight-fingers --strict-collisions 1e-4 \
    --save experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --quiet-rai --no-view
```

This produced 12 actions, nine persisted IK-bearing witnesses, two witness sets, and three
milestone repairs. It uses the wrench model, asymmetric grasps, the 5 N vacuum engagement load,
20 N insertion load, collapsed-h `press` signature, milestones, and placement. The geometry
therefore does not need the earlier no-wrench/free-floating fallback. `--witness-margin 0.005`
is the shared hard search/IK clearance gate; floor witness optimization maximizes additional
clearance above it. `--strict-collisions 1e-4` is the numerical penetration contract repeated
during motion planning.

For this assembly, keep the 5 cm funnel on insertion but disable the auxiliary pick/place/
add-grasp/release funnels. The all-default-funnel problem contained an unsampleable auxiliary
transition, while removing the insertion funnel made the simulated gripper hit the collars too
hard. This command builds the exact final export and found a complete optimized path:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --load-plan experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --motion-plan --pre-pick 0 --pre-place 0 --pre-add-grasp 0 \
    --pre-insertion 0.05 --post-release 0 \
    --mp-export-save experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-final-export.json \
    --mp-runtime 90 --mp-shortcut-iters 100 --mp-seed 11023 \
    --mp-collision-tolerance 1e-4 --mp-robot-margin 0.1 --mp-margin-ramp 0.1 \
    --mp-save experiments/mp-peg-stool-2r-mixed-wrench-flat-engage5-seed11023-limits.json \
    --quiet-rai --no-view
```

The successful MP file has 116 states and embeds the exact post-fusion export: nine skill and
three pre-skill keyframes. Replay performs zero IK solves. Motion-planning predicates now reject
joint-limit violations as well as collisions/margins; this fixed a sampled shoulder detour that
was outside the UR10 range even though both task endpoints were legal. To retry only motion
planning, replace `--mp-export-save` with
`--mp-export-load experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-final-export.json`;
do not refresh witnesses.

Run and replay the actuated simulation with:

```bash
python3 sim.py run \
    experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-plan.json \
    --mp-path experiments/mp-peg-stool-2r-mixed-wrench-flat-engage5-seed11023-limits.json \
    --actuated --corridor --corridor-rate-clamp --coll-scale 0.8 \
    --spline --limit-rate --no-force-limits --servo-torque-guard 0.7 \
    --record experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-actuated.npz

python3 sim.py replay \
    experiments/peg-stool-2r-mixed-wrench-flat-engage5-seed11023-actuated.npz
```

The recording completed all 12 actions in 4,136 frames; its worst final position residual was
2.0 mm. Strict preflight checked 3,392 samples and reported both collision tolerance and joint
limits OK. That dated recording used the former torque-only meaning of
`--servo-torque-guard 0.7`: it engaged for 167 ticks below the 6 N force trigger. The current
guard is contact-conditioned as documented above, because a total weld wrench also contains
the static gravity reaction and must not freeze transport indefinitely. `--no-force-limits`
disables only the predictive speed retimer: it
assigns a table-supported assembly's entire load to one gripper and can incorrectly reduce a
segment to zero speed. MuJoCo contact forces and physical grasp-break limits remain active, and
the search still uses `--wrench`. `--coll-scale 0.8` is the requested debug-only arm collision
shrink, so this rollout is not a clearance certificate for unscaled collision geometry.

## Car-mini with three robots

The saved recording
`experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz` completed all 21
physical actions with seed 11023. Replay it with step 4 below. The current motion export also
couples each `add_grasp -> pre-assemble -> assemble -> release(s)` chain; this changes task IDs,
so do not combine the older `...default-float.json` MP path with the current source. Export the
current `...coupled.json` path with step 2.

The run uses limited UR10 arms, wrench checks, asymmetric two-finger grasps, a 20 N insertion
force, and 0.01 m insertion uncertainty.

Current defaults already enable milestones, real IK, wrench checking, asymmetric grasps,
bracing, strict collisions, tight fingers, floor witness optimization, co-grasp fusion,
release hoisting, pre-insertion, sequential optimized motion planning, 1e-4 MP collision
tolerance, 0.1 m robot margin, 1,000 shortcut attempts, collapsed-h signature search, and the
`press` signature component. The commands below do not repeat most of those defaults; pass
`--sig-collapsed-h --sig-components press` explicitly when a saved experiment needs to make
the search signature unambiguous.

### 1. Search and save the discrete plan

```bash
python3 main.py --solve \
    --assembly car-mini --parts 5 --num-robots 3 --weight inf \
    --robot ur10_limited --seed 11023 --min-sep 0.3 \
    --insert-force 20 --insertion-noise 0.01 --tf-close-torque 0.5 \
    --ignore-gravity --random-init --quiet-rai --no-view \
    --save experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json
```

### 2. Export and save the motion path

Always load the config and plan from the same JSON. Keep the saved plan and MP path together.

```bash
python3 main.py \
    --config experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --load-plan experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --motion-plan --mp-runtime 60 --mp-seed 11023 --mp-shortcut-iters 100 \
    --quiet-rai --no-view \
    --mp-export-save experiments/car-mini-3r-asymmetric-seed11023-final-export.json \
    --mp-save experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json
```

`--mp-export-save` is written before planning starts, even if BiRRT finds no path. It contains
the final post-fusion skill witnesses, every pre-skill keyframe, exact task goals/effects, task
order, export settings and signature. A successful current `--mp-save` also embeds this same
block in the path file. Replayers treat missing data as an error; they never fill it by solving
IK unless an explicit refresh flag is given.

Once that final export exists, regenerate ONLY the motion path (zero witness/pre-skill/fusion
IK solves) with:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --load-plan experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --motion-plan \
    --mp-export-load experiments/car-mini-3r-asymmetric-seed11023-final-export.json \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-save experiments/mp-car-mini-3r-asymmetric-seed11023-final.json
```

That older artifact produced 152 states and embedded its final export. The current robust
Car-mini result is the 522-state artifact in the evaluation table above. Its endpoint
keyframes are coupled, while `--mp-cograsp` intentionally reports the ordinary planner's
unconstrained interior detours.

On `--mp-export-load`, the artifact supplies the default collision/clearance settings, but an
option explicitly present on the current command line wins: `--mp-collision-tolerance`,
`--mp-robot-margin`, `--mp-margin-ramp`, and `--mp-table-margin` can therefore be retuned without
resolving any IK. The `[mp-export-settings]` line prints every effective value and labels its
source as `export`, `CLI`, or `CLI/scene`. Exact task/keyframe and structural settings remain
owned by the final export. Merely receiving a value through `--config` is not an explicit
override; name the option on this command when you intend to change the export's setting.

Do not add `--hoist-releases` to this load command. A saved plan already contains that
post-pass. The loader now avoids applying it twice unless the flag is explicitly requested.

Structural co-grasp coupling is automatic and always enabled; `--fuse-cograsp` and
`--pre-insertion 0.05` are also current defaults. For a paired chain, all assembly participants
first meet at one shared funnel-mouth keyframe when there is no support switch. If the target is
placed between `add_grasp` and `assemble`, the pre-switch holders instead share the true place
keyframe; released holders may then retract. These are endpoint equalities only: the regular
motion planner may detour between them. `--fuse-cograsp` is the additional same-kinematic-mode
witness re-solve. The current robust Car-mini evaluation fused 4/4 pairs and reached all 44
modes; its closed-chain drift diagnostic is reported in the table above.

### 2a. Replay the saved motion path without planning or physics

```bash
python3 tools/visualization/mp_viser_saved.py \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json \
    8080 \
    --mp-export experiments/car-mini-3r-asymmetric-seed11023-final-export.json
```

Open `http://localhost:8080`. This read-only viewer checks the complete ordered mode chain and
uses each saved transition configuration to reconstruct attachments. The extra `--mp-export` is
needed only because this particular path predates embedded final exports; newly written MP paths
need only the original three arguments. The saved coupled path can
print a task-signature warning because it predates the deterministic holder-order fix; its old
signature was reproduced exactly and differs only in equivalent robot-list ordering. A new
step-2 export has the stable signature and does not warn. Do not waive a signature mismatch for
physics execution: rerun step 2 before `sim.py run`.

### 3. Run the requested MuJoCo validation

```bash
python3 sim.py run \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-plan.json \
    --mp-path experiments/mp-car-mini-3r-asymmetric-seed11023-coupled.json \
    --actuated --corridor --corridor-rate-clamp \
    --coll-scale 0.8 --spline --limit-rate \
    --record experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz
```

For a fresh physics validation, require the final line to report
`21/21 actions ok; plan SUCCEEDED`. `--coll-scale 0.8` is a debug override that shrinks arm
collision primitives. The resulting simulation clearance does not exactly match the planner
certificate.

### 4. Replay the recording

```bash
python3 sim.py replay \
    experiments/car-mini-3r-asymmetric-seed11023-default-float-actuated.npz
```

## Car-mini-specific pitfalls

- `car-mini` has five parts. Do not use the old `--parts 12` value. The CLI clamps it to five,
  but `--parts 5` records the actual intent.
- Use `--num-robots 3` for this replay. The old command requested four robots.
- Do not use the old `--part-float 0.02` workaround. The current 0.002 m default is part of the
  validated plan. The 20 mm value produced table contacts and broken grasps in MuJoCo.
- `--scatter-spacing` has no effect with `--random-init`.
- Remove duplicate flags. In particular, the last repeated `--seed` silently wins. The CLI now
  warns about duplicates.
- `--viser`, `--mp-viser`, `--show-init`, and `--mp-collision-view` open blocking viewers. They
  are not part of a headless production run.
- Each wheel is one native MuJoCo cylinder. Its collision and visual geometry are now the same,
  and a regression test checks that the settled wheel rim is on the table.

Run the focused regression checks with:

```bash
python3 -m pytest -q \
    tests/test_config.py \
    tests/test_cograsp.py \
    tests/test_pre_skill.py \
    tests/test_sim_outcomes.py::test_car_mini_wheel_rims_rest_on_sim_table
```

## Cube-8 with four vacuum grippers

The 2026-08-14 run used wrench checking, the asymmetric grasp model, collapsed-h search, and
the `press` signature component. These options are current defaults, but they are written
explicitly here so a future default change cannot silently alter the experiment:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py --solve \
    --assembly cube --parts 8 --num-robots 4 --vacuum --weight inf \
    --wrench --grasp-model asymmetric \
    --sig-collapsed-h --sig-components press \
    --seed 11023 --check-init --quiet-rai --no-view \
    --save experiments/cube8-4r-vacuum-seed11023-plan.json
```

Search completed with 40 actions. It found 89,312 co-reachable signature states and persisted
33 witnesses. Generate the final zero-IK export first; it is saved before BiRRT starts, even
when the strict planning attempt times out:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/cube8-4r-vacuum-seed11023-plan.json \
    --load-plan experiments/cube8-4r-vacuum-seed11023-plan.json \
    --motion-plan --mp-runtime 60 --mp-seed 11023 --mp-shortcut-iters 100 \
    --mp-cograsp --quiet-rai --no-view \
    --mp-export-save experiments/cube8-4r-vacuum-seed11023-final-export.json \
    --mp-save experiments/mp-cube8-4r-vacuum-seed11023-final.json
```

The strict 1e-4 collision-tolerance attempt stopped at 36/48 modes. The terminal path used for
the simulation was generated from the saved export with an explicit 0.01 m tolerance and no
shortcut expansion:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config experiments/cube8-4r-vacuum-seed11023-plan.json \
    --load-plan experiments/cube8-4r-vacuum-seed11023-plan.json \
    --motion-plan \
    --mp-export-load experiments/cube8-4r-vacuum-seed11023-final-export.json \
    --mp-runtime 120 --mp-seed 11023 --mp-optimize --no-mp-shortcut \
    --mp-collision-tolerance 0.01 --mp-cograsp --quiet-rai --no-view \
    --mp-save experiments/mp-cube8-4r-vacuum-seed11023.json
```

This produced a terminal 167-state path. The co-grasp audit reports one non-rigid span after
action 16 (`a2+a4`, 3089 mm / 178.9 degrees of relative drift). It is a handover/place-created
co-grasp rather than an `add_grasp` chain, so the current add-grasp/assemble coupling does not
cover it. The path is not a valid production artifact until both this rigidity defect and its
strict base-path collisions are removed.

The old relaxed rollout was recorded with:

```bash
PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 sim.py run \
    experiments/cube8-4r-vacuum-seed11023-plan.json \
    --mp-path experiments/mp-cube8-4r-vacuum-seed11023-raw.json \
    --actuated --corridor --corridor-rate-clamp --corridor-refine 0 \
    --coll-scale 0.8 --spline --limit-rate --mp-cograsp --no-path-preflight \
    --record experiments/cube8-4r-vacuum-seed11023-actuated.npz
```

This is retained as a historical/debug artifact, not a current validation result.
`--no-path-preflight` is an explicit unsafe bypass and `--coll-scale 0.8` changes the robot
geometry. Verification saw eight accepted base-path violations, including up to 2.04 mm of
`a2_ur_coll8`/table penetration. The rollout happened to complete all 40/40 actions with a
worst final part error of 0.6 mm / 0.1 degrees. Replay its 33,120 recorded frames with:

```bash
python3 sim.py replay experiments/cube8-4r-vacuum-seed11023-actuated.npz
```

Current simulation replay defaults to a strict base-path preflight before spline/corridor
construction. It samples every edge at 0.01 rad, enforces joint limits, and uses a collision
tolerance of 1e-4 m even when the saved path was planned with a looser value. The compacted
shortcut path preserves its 405.23-rad geometry while reducing 757 per-stretch serialized
samples to 246 geometric waypoints. It is therefore the path to use once it passes validation:

```bash
python3 sim.py run \
    experiments/cube8-4r-vacuum-seed11023-plan.json \
    --mp-path experiments/mp-cube8-4r-vacuum-seed11023-final.json \
    --actuated --corridor --corridor-rate-clamp --spline --limit-rate --mp-cograsp
```

The current Cube shortcut artifact is correctly rejected in about five seconds, before spline
work: stretch 19, task mode `[19, 21, 27]`, has 0.117 mm of
`a2_ur_coll8`/`a2_ur_coll5` penetration. The raw path is rejected even earlier at stretch 1
with 0.173 mm of `a3_ur_coll8`/`a3_ur_coll5` penetration. Regenerate a strict path rather than
disabling the audit or tuning the corridor. `--path-preflight-tolerance` and
`--path-preflight-resolution` exist for explicit diagnostics; `--no-path-preflight` is for
historical ablations only.

## Cross and Chair coupled-keyframe evaluation

These are keyframe + motion-planning runs only; no MuJoCo simulation was run. `srp-cross.json`
and `chair8-r9-plan.json` were selected because they are the only saved Cross/Chair plans in
this checkout containing `add_grasp` actions.

Cross (7 parts, 4 robots):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill --pre-insertion 0.05 \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-export-save experiments/cross7-final-export.json \
    --mp-save experiments/mp-cross7-coupled.json
```

This plan contains `add_grasp -> place(target) -> assemble -> release -> release`. The place
changes the kinematic parent from a gripper to the table, so the two-phase witness fuser reports
`0/1` (expected). The first attempted shared goal incorrectly took former holder `a1` from the
post-place assemble witness, where it is at home: mode `[27, 27, 32, 27]` consequently left
`obj_0` 922 mm from its table pose. That invalid 193-state diagnostic has been removed; it must
not be used as a motion-planning artifact.

The corrected exporter uses `a1`'s real place slice, keeps persistent brace `a4` on the assembly
funnel branch, and lets inserter `a2` approach only after `a1` releases. The target pose error is
0.004 mm and the exact exported-keyframe audit reports 0/33 colliding goals. A corrected
60-second BiRRT run reached 29 modes (deepest task IDs `[32, 28, 32, 28]`) but did not finish, so
there is currently no valid Cross MP path to replay. Inspect the corrected exported keyframes
without running the planner using:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --mp-export-load experiments/cross7-final-export.json \
    --mp-modes --mp-runtime 0 \
    --quiet-rai --no-view --viser-port 8081
```

Retry motion planning from those precomputed keyframes, without resolving witnesses, using:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config srp-cross.json --load-plan srp-cross.json \
    --motion-plan --mp-export-load experiments/cross7-final-export.json \
    --mp-sequence --mp-optimize --mp-runtime 300 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view --mp-save experiments/mp-cross7-final.json
```

Exported paths must finish the configured RRT* optimization pass; `--no-mp-optimize` is accepted
only as an unsaved first-solution timing diagnostic. A 2026-08-14 attempt was stopped after it
again remained at 29/33 modes with
deepest task IDs `[32, 28, 32, 28]`. That is not a proof of infeasibility, but it localizes the
failure to the final transit search rather than witness IK or the corrected coupled keyframes.

Planner termination now prints a structured collision-predicate audit. `[mp/margin]` separates
mr_bench's base collision rejections from our robot/table-margin wrapper and names every wrapper
gate that fired; `[mp/base]` samples concrete penetrating pairs; `[mp/edge]` reports the dominant
first collision on failed edges at the frontier. On Cross, the wrapper rejected only 1/50,556
base-valid frontier checks in a 10-second run. The dominant failed-edge pair was the just-released
`a1` finger against placed `obj_5` (291/1,403 failed frontier edges); the direct transition also
puts carried `obj_6` 39.4 mm into an `a4` link. The wall is therefore the coordinated `a1`
retraction plus `a2` approach around bracing `a4`, not `--mp-robot-margin`. Set
`MP_MARGIN_DEBUG=1` only when an individual line for every wrapper veto is needed; the default
aggregate is substantially more readable.

Chair (8 parts, 4 robots):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config chair8-r9-plan.json --load-plan chair8-r9-plan.json \
    --motion-plan --fuse-cograsp --fuse-pre-skill --pre-insertion 0.05 \
    --mp-sequence --mp-optimize --mp-runtime 60 --mp-seed 11023 \
    --mp-shortcut-iters 100 --mp-collision-tolerance 1e-4 \
    --mp-robot-margin 0.1 --mp-margin-ramp 0.1 --mp-cograsp \
    --quiet-rai --no-view \
    --mp-export-save experiments/chair8-r9-final-export.json \
    --mp-save experiments/mp-chair8-r9-coupled.json
```

Chair built its 49-task keyframe problem, but the 60-second BiRRT run did not find a complete
path (41 modes reached; deepest task IDs `[43, 40, 40, 48]`), so no MP file was written. Its two
`add_grasp` actions are after the final assembly and support the final placement; this saved plan
therefore has no eligible add-grasp/assemble pair and does not exercise the new chain coupling.

Inspect Chair's saved action keyframes in the scrubbable replay viewer:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 replay.py \
    chair8-r9-plan.json --viser --viser-port 8082
```

To inspect the actual per-action keyframe ladder compared by witness optimization, overlay the
selected opaque keyframe with up to three translucent candidates:

```bash
python3 replay.py experiments/ur10-chair8-mixed-current-standard-plan.json \
    --viser --viser-port 8080 --optimization-alternatives 4 --alternative-opacity 0.16 \
    --active-robots-only
```

The slider still indexes actions; ghosts are candidates considered for that exact action, not
neighboring times and not the later same-pose GoalSet used by motion planning. Member zero is the
candidate selected by the optimizer. The translucent subset is chosen by greedy farthest-point
sampling in joint space solely to avoid drawing nearly coincident restarts; this does not affect
optimization. Changed arms and payloads attached to them are ghosted in the action's reconstructed
state. With `--active-robots-only`, only the arms involved in that action are rendered (both arms
of a handover/cograsp; the releasing arm for a release). Redundant translucent collision proxies
are hidden in this view; opaque primitive-only parts and the table remain visible. Candidate
ladders are persisted under
`witnesses.optimization_candidates`; older plans need one explicit optimizer pass to acquire
them, starting from a compatible persisted witness baseline rather than recertifying the
underlying search IK. A missing pool fails explicitly—the viewer never regenerates optimization
candidates.

For a deliberately more dramatic diagnostic illustration, use the separately persisted broad
branch pool:

```bash
python3 replay.py experiments/ur10-chair8-mixed-current-standard-plan-diagnostic.json \
    --viser --viser-port 8080 --optimization-alternatives 4 \
    --alternative-source diagnostic --alternative-opacity 0.16 --active-robots-only
```

This is explicitly **not** the optimizer ladder: it also contains individually solved IK branches
rejected by later clearance, continuity, or interchangeability gates. It is therefore useful for
showing how different candidate keyframes can be, but must not be presented as four feasible
motion-planning goals. `--alternative-source optimizer` remains the default and exact provenance
view. Viser preserves the imported mesh's per-vertex material colors instead of flattening a
multi-material robot link to one color; ghost layers also do not cast shadows.

Or inspect the exact 49 exported mr_bench modes (including pre-insertion keyframes and task
validity checks) without planning:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache python3 main.py \
    --config chair8-r9-plan.json --load-plan chair8-r9-plan.json \
    --motion-plan --mp-export-load experiments/chair8-r9-final-export.json \
    --mp-modes --mp-runtime 0 \
    --quiet-rai --no-view --viser-port 8083
```

This is a true replay: it reports `[mp-export-reuse] ... ALL witness IK/fusion solves SKIPPED`,
loads the exact persisted 49-task manifest, and refuses a plan/joint/signature mismatch. A bare
`--motion-plan --mp-modes` is the one-time export/debug route and may solve missing final
keyframes; `--mp-export-load` is the zero-solve viewer route.

The Chair ending has three robots co-grasping `obj_0`; after `a2` releases, `a1` must remain
the kinematic holder for the saved `place a1` action. This re-root selection is now canonical
by part and robot-roster order. Older source revisions leaked `frozenset` iteration order here
and could fail at that place action under an unlucky ambient `PYTHONHASHSEED`. The explicit
environment prefix above remains recommended for reproducibility of the rest of the pipeline.

Flags used in both evaluations:

- `--config FILE` restores the saved scene and run configuration; `--load-plan FILE` reuses its
  discrete actions and witnesses without searching again.
- `--motion-plan` builds and runs the mr_bench problem. `--mp-sequence` fixes the saved action
  order; `--mp-optimize` keeps improving RRT* cost until `--mp-runtime 60` expires.
- `--fuse-cograsp` runs the extra same-mode two-keyframe co-grasp IK pass.
  `--fuse-pre-skill` pins idle arms while re-solving funnel keyframes, and
  `--pre-insertion 0.05` creates a 5 cm funnel-mouth keyframe before each mate.
- `--mp-seed 11023` seeds planner and shortcut draws. The wall-clock cutoff still makes this
  approximately, not bit-exactly, reproducible; use `--mp-fake-clock` when bit-exact planner
  draws matter.
- `--mp-shortcut-iters 100` caps the non-time-bounded shortcut pass.
  `--mp-collision-tolerance 1e-4` allows at most 0.1 mm total penetration at validated states.
- `--mp-robot-margin 0.1 --mp-margin-ramp 0.1` requests 10 cm inter-robot clearance in transit,
  relaxed over the final 10 cm for robots participating in the same task.
- `--mp-cograsp` reports path-level relative gripper drift (with `--mp-sequence`).
  `--mp-save FILE` writes a path only when planning succeeds.
- `--mp-export-save FILE` writes the final task/keyframe artifact whether planning succeeds or
  not. `--mp-export-load FILE` reconstructs from it with zero IK/fusion solves. Current
  `--mp-save` files embed this artifact automatically; `--mp-export` on
  `tools/visualization/mp_viser_saved.py` is
  only for older path files.
- `--quiet-rai --no-view` suppresses known RAI collision-library noise and prevents blocking
  viewers. The three environment assignments fix hash order and put caches in writable paths.

## Vacuum SS-IK and open witness search

An intuitive and paper-style description of the current search pipeline—including the
signature-collapsed backward pass, fact components, milestone fills and repair, open witness
validation, and `--milestones-anytime`—is in [SEARCH_METHOD.md](SEARCH_METHOD.md).
The proposed paper scope, evaluation matrix, simulation-realism work, and real-robot experiment
plan are collected in [PAPER_PLAN.md](PAPER_PLAN.md).
A manifest-driven, fresh-process search benchmark runner and the scalable synthetic fixture-rack
family are documented in [benchmarks/README.md](../benchmarks/README.md).

### Milestone-oracle ablation

`--milestone-oracle` isolates how much backward reasoning a milestone run uses without changing
its concrete action model, IK gates, module handling, backtracking, adaptive fill windows, lazy
validation, or final settling:

- `global` (default) builds the full signature-collapsed order-ideal database and uses its part
  ordering, insertion/brace hints, and shifted fill heuristic.
- `chain` keeps those grasp/force/rest-face signals but projects the database onto
  `assembly_order` prefixes and their one incoming loose part. It is intended for declared
  sequence assemblies and rejects detached connect modules explicitly.
- `structural` builds no *global* signature database. Dependencies/modules and canonical part
  order pick the next milestone. Without `--fill-local-h`, fills use the cheap counting
  heuristic; the current standard instead constructs one small goal-conditioned local PDB per
  fill. The standard also enables `--useful-grasps`: raw grasps that exist at the mate geometry
  and can transmit the insertion load are ordered before grasps that necessarily require a
  regrasp, and the latter receive the admissible one-action regrasp cost. This is guidance only;
  `--no-useful-grasps` is the matched ablation and no grasp is pruned in either setting. The
  structural solver still runs the unrestricted unbraced phase followed, when enabled, by the
  unrestricted braced phase. If both fail in the
  local action scope, `--fill-scope-widening` repeats them while allowing temporary relocation of
  any loose part before rejecting the milestone. The local PDB supplies scalar cost-to-go
  guidance for the joint target rest-face/holder configuration.

  The local scope is passed directly into successor generation: irrelevant pick, place,
  handover, in-hand-reorientation and brace loops are skipped before successor states are
  allocated, and assembly generation is restricted to the requested incoming part/module.
  Releases remain global. The old post-generation scope predicate is retained as a safety check;
  this is an implementation optimization of the same ordered transition set, not an additional
  pruning assumption. The all-part completeness-widening pass uses unrestricted manipulation
  generation.

To inspect search ordering against a known feasible plan, add
`--diagnose-plan-rank experiments/KNOWN-PLAN.json` to a milestone solve. For every reference
fill that is representable in the current widening phase, this runs a separate, bounded copy of
the actual optimistic local A* (same successors, local PDB, action costs, scope and cheap gate;
no IK calls). Each demonstrated state reports its pop rank, best search `g/h/f`, demonstrated
prefix `ref-g`, the number of lower- and equal-`f` states popped first, and the demonstrated
action's rank among local successors. A state may have `g < ref-g` when another action prefix
reaches the same symbolic state more cheaply. The
diagnostic never guides or prunes the ordinary search, which continues normally afterward.

Matched single-run real-IK measurements on 2026-08-24 used the saved robust scene configs,
disabled post-search witness optimization/fusion, and imposed a 6 GB address-space guard. The
saved configs omit solver-only knobs, so this table uses the current local-fill default
`--weight 1` on both sides:

| assembly | global | chain | structural | interpretation |
|---|---:|---:|---:|---|
| Car-mini, 3 UR10 | 20.95 s / 475 MB / 25 actions | **15.33 s / 441 MB / 25** | 74.22 s / 1.74 GB / 25 | projected future guidance is valuable; pure counting floods place/regrasp plateaus |
| Cross-7, 4 UR10 | 12.88 s / 737 MB / 28 | 13.05 s / 737 MB / 28 | 13.16 s / **620 MB** / 27 | effectively tied; its dependencies already make the global space chain-like |
| Chair-8, mixed 4 UR10 | 119.73 s / 1.86 GB / 35 | unsupported (detached module) | **93.32 s / 1.47 GB / 35** | module structure supplies the useful order; skipping the 637,747-state DB wins |
| Cube-8, 4 vacuum UR10 | >180 s | >180 s (6/7 additions reached) | failed at 96.44 s during width-2 rescue / 4.61 GB | the remaining bottleneck is the concrete local fill, not only global backward construction |

The Cube chain control was also repeated with the standard command's `--weight 100`; it reached
only five of seven additions by 180 s. Thus the failure is not simply caused by an overly optimal
local A*. Hundreds of infeasible pick/place/handover alternatives on already-built parts share
the same local score. The next performance change should be a short manipulation-skeleton pass
(direct carry-and-insert, target hold, handover, brace) before the existing complete local graph
search. Keeping the latter as fallback preserves coverage; replacing it with a greedy cutoff
would not.

#### Goal-conditioned greedy fills and exact ablations

`--fill-local-h` builds a simpler alternative to full-assembly backward guidance.
For every one-part fill it builds a backward database whose goal is exactly “join this incoming
component and settle it.” Every grasp label uses the same fact-signature projection as the
standard heuristic; there is no special interface/end-grasp label. This avoids the raw-label
Cartesian product on detached modules. It is not tied to a chain: the current target and mover
components are read from the concrete checkpoint state.

The standard `--fill-greedy` uses `f=h` for each concrete fill and lets lazy IK repair continue
past the ordinary 200-candidate cutoff. Node/deadline budgets still apply. It changes ordering,
not the graph or feasibility gates, but it does not promise the cheapest local fill. This is
**not** end-to-end optimal search: locally cheapest settled checkpoints can make later fills more expensive, and
the milestone order, fill-width limit, and phased action-set widening remain in effect. No
reachability/action-order tie-breaker is part of this experiment.

`--fill-exact --no-fill-greedy` is the weight-one local A* ablation. It has the same exhaustive
repair-to-frontier/budget behavior, but optimizes within the currently exposed phase.

`--fill-incumbent-first` is an opt-in anytime ordering of that same local search. It first runs
lazy greedy best-first search (`f=h`) until it has a fully IK-validated fill, then starts the
ordinary weight-one lazy search with `g+h` bounded by the incumbent cost. The incumbent itself
is admitted: popping any validated goal at that cost lets A* consider every cheaper queue key
without turning finite nonlinear-solver failures into an infeasibility proof. Exact witnesses
and open-witness effort are solve-scoped and reused across the two passes. The flag requires
`--fill-local-h --fill-exact --lazy-fills`, is saved with the plan, and is deliberately not in
the standard profile while the benchmark result is mixed.

The local queue policy is deliberately independent of global `--weight`. This matters for
`--milestones-anytime T`: its milestone phase uses the standard greedy fills to obtain an
incumbent, then re-fills bounded consecutive milestone neighborhoods under their current cost.
By default it grows the neighborhood through the complete milestone horizon; `--lns-window K`
is an explicit cap. Every window first runs greedy best-first search (`f=h`) below the old span
cost, then weight-one A* below the greedy replacement cost (or the old cost if greedy found
nothing). It accepts only strict improvements and restarts after every accepted replacement.
Each refill uses the ordinary memory-light heap engine and may
manipulate exactly the parts touched by the removed incumbent actions; all other parts
remain frozen. Larger windows therefore widen the neighborhood through the incumbent itself,
without exposing unrelated manipulation. The final `k=n` window is a bounded whole-horizon
search over that incumbent-induced action scope. Exhausting it certifies no cheaper plan in that
finite scoped problem; this is not automatically a global certificate for action families that
the milestone/widening schedule never exposed. A plain standard run only returns the first plan.

Every LNS window uses the same greedy-then-weight-one schedule. Under a finite anytime deadline,
the remaining wall time is divided fairly among the remaining windows; greedy receives at most
half of a window's remaining slice, and unused time flows to the exact pass. Every attempted
pass prints an `[lns-window]` record containing its order, incumbent bound, touched-part
footprint, allocated slice, outcome, wall time, A* calls/expansions/peak OPEN, scoped versus raw
successors, cheap checks, and IK failures. These are operational bounded-search diagnostics: the
current LNS binder still treats a failed nonlinear attempt as an edge failure within that refill.

The refill has no portfolio phases. Heuristic values and exact cheap-collision verdicts are
memoized by their value-state/edge query. Deterministic successor lists use second-access
admission, so one-pass windows retain no edge graph while lazy retries reuse genuinely repeated
states. The first IK-refuted candidate triggers a one-time handoff to the same incremental
LazySP repair engine used by ordinary local fills; later refutations delete their edge and repair
the affected subtree instead of restarting A*. Before that handoff, no alternate-route graph is
allocated. `CollisionChecker` uses the same second-access policy for its much larger
physical-scene keys (first occurrence retains only a hash; full-key lookup remains exact), and
completed window arenas are returned before constructing the next window. These are work/storage
changes only: the graph, priority, cost bound, and feasibility predicates are unchanged.

For each non-final splice, LNS retargets the same signature-collapsed reverse Dijkstra used by
local fills to the incumbent's exact projected endpoint and restricts its ideal chain to the
destroyed interval. Reverse expansion is capped strictly below the incumbent window cost. Forward
A* uses the maximum of this direct endpoint distance and the older global-landmark difference; if
the bounded reverse table exhausts without reaching the window start, `[lns-window-h]` reports an
`endpoint-bound` certificate and the forward search is skipped. This is an admissible abstraction
bound, not an IK infeasibility claim. The final window keeps the ordinary set-valued assembly goal
rather than being forced to reproduce the incumbent's final concrete state. If the endpoint PDB
hits its 200,000-state guard, LNS reports the fallback and still runs with the landmark bound.

The reverse PDB is now query-driven. It seeds all abstract fill goals once, then resumes
multi-source Dijkstra only until the state requested by forward A* has been settled and expanded;
the frontier and lazily computed pick/wrench facts survive the pause. It returns exactly the same
finite distance as an eager table, and prints settled/discovered counts per fill. This is a
work-avoidance mechanism, not a promise that every PDB stays small: Car-mini's tested exact fills
eventually requested their entire 378--3,541-state graphs, while several Chair fills stopped with
unexpanded frontier states. The hard state guard still fails closed.

`--fill-skeleton-first` is the deliberately simpler reviewer baseline. Before an ordinary fill it
probes direct pick/insert, target-hold, one-handover, and one-brace paths using the same successor,
wrench, collision, and IK code. It validates at most `3 * num_robots` candidate assignments and
then prints an explicit transition to the unchanged complete local fallback; failure of the probe
is never reported as assembly infeasibility. It is skipped for incumbent-bounded and open-witness
fills so it cannot weaken those semantics.

Single guarded runs found 25-action Car-mini in 25.49 s / 470 MB and 27-action Cross-7 in
8.06 s / 386 MB. Mixed Chair-8 required complete fallback for its module and took 99.44 s /
1.57 GB, so this is not a universal replacement for reverse guidance. Use the repeated matched
manifest in `benchmarks/manifests/local_fill_reviewer.json` before making a paper claim.

The fresh three-seed manifest confirms that split: direct skeleton averages 28.26 s versus
76.69 s counting on Car-mini and 12.92 s versus 17.64 s on Cross, with equal cost on both. On
Chair it is slower (111.00 s versus 93.90 s) and falls back on all three module fills. Fixture-rack
is the current negative generalization case: all wrench-enabled variants reach the second lateral
bracket and then fail under the resource guard, while the matched no-wrench direct run completes
in 1.14 s. The four-part prefix commits its one retained milestone boundary to a settled rest pose
that loses the useful target hold. This is evidence for retaining a small, diverse set of boundary
states—not for adding another part-order heuristic. Full numbers and the guarantee boundary are in
`SEARCH_METHOD.md`.

Guarded real-IK runs on 2026-08-24 after the resumable-oracle and unified mixed-wrench change
(6 GB address-space limit, `PYTHONHASHSEED=0`, no witness post-optimization) gave:

| assembly | structural + local exact | result |
|---|---:|---|
| Car-mini, 3 UR10 | 47.31 s / 754 MB | 26 actions; every local PDB eventually exhausted (378--3,541 states) |
| Cross-7, 4 UR10 | 7.42 s / 411 MB | 31 actions; five of six fills left reverse frontier unexpanded |
| Chair-8, mixed 4 UR10 | 96.41 s / 1.65 GB | 35 actions; detached module supported; early PDBs partial, hard module fills exhausted (largest 88,136 states) |
| Cube-8, 4 vacuum UR10 | not rerun | earlier exact-local run found no plan within 180 s; sixth fill was dominated by failed concrete binds |

Cross demonstrates useful avoided reverse work. Car demonstrates the limiting case where the
query-driven oracle degenerates to the eager table. Chair's wall time is still dominated by 471
continuous binds (269 failures), not its partial PDB construction; automatic mixed-tool wrench
facts also make this a stricter and more expensive comparison than the former topological
mixed-tool fallback. The method remains substantially better than counting-only structural
search on Car, but it is not uniformly faster on Chair. Cube remains a continuous-realization
bottleneck rather than merely a backward-table bottleneck.

Example (replace the saved config/output as needed):

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache \
python3 main.py --config experiments/car-mini-3r-robust-seed11023-plan.json --solve \
  --weight 1 --milestone-oracle structural --fill-local-h --fill-exact \
  --no-open-witness --optimize-witnesses off --witness-sets 0 --no-fuse-cograsp \
  --quiet-rai --no-view --save experiments/car-mini-fill-local-exact.json
```

Example matched ablation command:

```bash
PYTHONHASHSEED=0 NUMBA_CACHE_DIR=/tmp/assembly-numba-cache \
MPLCONFIGDIR=/tmp/assembly-mpl-cache \
python3 main.py --config experiments/car-mini-3r-robust-seed11023-plan.json --solve \
  --milestone-oracle chain --no-open-witness \
  --optimize-witnesses off --witness-sets 0 --no-fuse-cograsp \
  --quiet-rai --no-view --save experiments/ablation-car-chain.json
```

Analytic IK seeding is on by default with `--real-ik`. It is calibrated separately for the
UR5/UR10 two-finger and vacuum tool frames; vacuum uses the grasp target's canonical aligned EE
roll. `--no-ik-seeding` disables all SS-IK seeding and restores the nonlinear reach precheck.
An empty vacuum solution set is not treated as a hard impossibility: the relaxed/full KOMO path
still decides borderline targets.

`--open-witness` is an experimental, isolated replacement for the permanent-failure behavior of
lazy width-1 milestone fills. The first unresolved edge immediately invalidates the incumbent:
later edges of that plan are not speculatively probed. That symbolic edge is suspended for the
rest of the fairness epoch and reopened in the next epoch. A physically equivalent alias shares
its attempt cursor but may invest a fresh stochastic batch because failure is not an
infeasibility certificate. Successes are cached; stochastic failures are not. Ordinary skills
escalate through 2, 4, then `--open-witness-batch N` attempts. Assemble
keeps one atomic N=7 structural batch: all six canonical orientation biases plus one unbiased
initialization in a shared scene. Splitting that small bundle across epochs measured worse on
peg-stool because the promising assembly did not reopen before greedy search found many other
plans. `--open-witness-epochs N`
(default 2) limits reopening effort per fill and is not an infeasibility certificate. The flag is
off by default and does not alter the legacy `bind()`/lazy-fill path. A batch shares scene and
reach-precheck setup.

Every fresh `--real-ik --save` now writes the **exact successful search witness** for every IK
action, even with `--optimize-witnesses off`. Open search retains the returned joint vector, not
just a successful boolean, and aligns those vectors to the final action list before writing JSON.
Release hoisting runs first; if it changes an IK predecessor scene, that changed edge is
certified once as part of the fresh solve. An IK action with a null witness makes export fail
immediately.

Witness optimization is layered on top of this immutable feasibility baseline. Changing, for
example, `off` to `floor` or changing the branch budget requests extra candidates but does not
discard the saved search solution. Failed candidate harvesting falls back to the exact baseline.
A normal `--load-plan`, `replay.py`, or sim replay never silently resolves missing IK.
`--refresh-witnesses` is the explicit one-time escape hatch for a legacy plan that predates this
persistence format; new files do not need it. It also permits a saved sampled-grasp hash to
differ, but only in this explicit mode: the old actions become a symbolic skeleton, persisted
witnesses are ignored, and successor replay plus every IK gate are run again under today's
database. If a grasp id now denotes an inadmissible action, refresh fails before writing a new
certificate. Without refresh, a hash mismatch always fails closed.

Example search only (no motion planning):

```bash
python3 main.py --solve \
  --assembly peg-stool-cuboid-4-tight-chamfer --parts 5 --num-robots 4 \
  --robot ur5_limited --grippers vacuum,two_finger,two_finger,vacuum \
  --weight 100 --milestones --sig-collapsed-h --sig-components press \
  --real-ik --open-witness --open-witness-batch 7 --open-witness-epochs 1 \
  --grasp-sampler --grasp-sampler-keep 20 --grasp-sampler-scenes \
  --grasp-sampler-wrap 0.1 --grasp-sampler-pad-support 0.02 \
  --wrench --grasp-model asymmetric --vacuum-no-assemble \
  --vacuum-engage-force 5 --two-finger-scale 10000 --grasp-compliance \
  --insert-force 0.1 --insertion-noise 0 --allow-curved-grasps \
  --scatter-spacing 0.2 --seed 11023 --witness-margin 1e-4 \
  --strict-collisions 1e-4 --optimize-witnesses off --witness-sets 0 \
  --no-fuse-cograsp --quiet-rai --no-view \
  --save experiments/bench-peg-stool-open-skill-tiered-plan.json
```

Historical V1 measurement on this command (before immediate replanning): true batched OPEN
reproduced the legacy
19-action plan in 38.3 s at 0.838 GB peak RSS. The otherwise identical legacy path took
54.0 s at 0.840 GB: search is 29% faster, bind wall falls from 25.2 to 9.1 s, and NLP solves
from 1,123 to 563. The two-part vacuum cube also improves from 9.7 s legacy to 6.6 s open.
As a second control, legacy with its ordinary restart budget reduced to six and reach-precheck
restarts disabled still took 66.0 s (31.9 s bind wall, 1,321 NLP solves) and selected a longer
20-action plan. Thus the measured gain is not merely a larger hidden per-edge effort budget.
The earlier one-bind-per-attempt prototype was much worse because it rebuilt the scene and
reran reach-precheck for every attempt; batching is the performance-critical distinction. V1
also kept validating the remainder of an incumbent after its first failure. That speculative
failure harvesting helped peg-stool but violated ordinary lazy-validation semantics, so the
current version stops immediately and is reported separately below.

To add optimized, replayable witness sets without rerunning search or baseline IK:

```bash
python3 main.py \
  --config experiments/bench-peg-stool-open-persisted-plan.json \
  --load-plan experiments/bench-peg-stool-open-persisted-plan.json \
  --optimize-witnesses floor --witness-sets 4 --witness-branch-budget 5 \
  --save experiments/bench-peg-stool-open-persisted-floor-plan.json \
  --quiet-rai --no-view

python3 replay.py experiments/bench-peg-stool-open-persisted-floor-plan.json --no-view
```

For an older plan without a `witnesses` block, repair it deliberately once with
`--refresh-witnesses --save NEW.json`. Without that flag, post-processing fails closed instead
of gambling on a fresh IK draw.

The corrected vacuum SS-IK itself does help on the hard case: the legacy search above took
54.0 s and 0.84 GB with analytic seeding versus 143.6 s and 1.70 GB under
`--no-ik-seeding`. The trivial two-part vacuum cube went the other way (9.7 s seeded versus
6.6 s unseeded), because collision-limited failures can spend more on exact branches than the
small nonlinear problem costs. The hard-case reduction is large enough to keep the calibrated
vacuum path enabled by default.

### Cross-assembly open-witness comparison

The original full-plan-prevalidation comparison was repeated on the robust Car-mini, Cross, and Chair
configurations on 2026-08-22. Both sides loaded the same saved configuration and disabled only
post-search work (`--optimize-witnesses off --witness-sets 0 --no-fuse-cograsp`); motion
planning was not run. The OPEN side additionally used
`--open-witness --open-witness-batch 7 --open-witness-epochs 2`.

| assembly | legacy search | V1 batched OPEN | change | NLP solves | bind wall | peak RSS | actions |
|---|---:|---:|---:|---:|---:|---:|---:|
| Car-mini robust, 3 UR10 limited | 34.33 s | 26.42 s | **-23.0%** | 1,384 -> 893 | 23.1 -> 15.6 s | 504 -> 488 MB | 26 -> 26 |
| Cross-7 robust, 4 UR10 | 10.38 s | 10.09 s | -2.8% | 320 -> 264 | 2.0 -> 1.4 s | 737 -> 737 MB | 27 -> 28 |
| Chair-8 robust, mixed UR10 | 70.86 s | 61.57 s | **-13.1%** | 2,751 -> 1,340 | 24.7 -> 15.6 s | 1,716 -> 1,700 MB | 33 -> 33 |

These are single paired measurements, not statistical confidence intervals. Cross is already
cheap and dominated by discrete search, so reducing its IK work does not materially improve the
end-to-end time; its greedy (`--weight inf`) OPEN run also chose one extra action. Every real
fill in all three OPEN runs completed in epoch one. These runs therefore validate the batching
speedup but do not exercise real-instance reopening; the fairness/reopening behavior has a unit
regression test.

Car-mini and Cross were subsequently floor-optimized outside the timed region and replayed from
18 and 22 persisted witnesses respectively, with per-action IK solving skipped. Chair's OPEN
and legacy searches both found the same 33-action structure, but independently regenerated
floor witnesses expose the same existing final co-hold problem: at the keyframe before step 32,
`a1_ur_coll7` penetrates `obj_5/obj_6` by 17.65 mm total. Chair's search timing is useful as a
matched performance comparison, but neither regenerated Chair artifact is claimed as a clean
replay. The identical failure on both paths localizes it outside the OPEN policy.

Current immediate-replan results use the same maximum batch but ordinary-skill tiers 2 -> 4 ->
7. Assemble retains its atomic seven-slot bundle. These are again individual runs:

| assembly | legacy | current OPEN | change | NLP solves | actions | replay |
|---|---:|---:|---:|---:|---:|---|
| Car-mini robust | 34.33 s | **23.25 s** | **-32.3%** | 1,384 -> 254 | 26 -> 25 | clean, 17 persisted witnesses |
| Cross-7 robust | 10.38 s | 15.76 s | +51.8% | 320 -> 261 | 27 -> 27 | clean, 21 persisted witnesses |
| Peg-stool chamfer | 54.0 s | **49.12 s** | -9.0% | 1,123 -> 538 | 19 -> 17 | old artifact predates exact persistence |

Car-mini confirms that speculative tail validation can be very expensive. Peg-stool shows its
opposite accidental benefit: one invalid incumbent often contained several independently doomed
edges, and V1 harvested all of those refutations before replanning. Immediate replanning explores
more sibling plans instead. Recovering that pruning without touching an already-invalid plan
requires a better first-edge selector or the deferred persistent pending-edge scheduler. Cross is
small and discrete-search dominated; its current single timing is not evidence of an IK
regression because NLP work still falls, but it is not an end-to-end performance win.

The old 17-action peg-stool artifact demonstrates the persistence bug: search found every edge,
but its plan-only JSON discarded those vectors and a separate floor pass failed to rediscover
step 2. It remains intentionally incomplete and should not be used for motion planning. A fresh
end-to-end run after the fix produced a 20-action plan with all 16 required search witnesses;
floor optimization retained all 16, and a second-process load of the floor artifact performed
zero bind/NLP calls:

- `experiments/bench-peg-stool-open-persisted-plan.json`
- `experiments/bench-peg-stool-open-persisted-floor-plan.json`

Car-mini and Cross earlier optimized artifacts are:

- `experiments/bench-car-mini-open-skill-tiered-witness-plan.json`
- `experiments/bench-cross7-open-skill-tiered-witness-plan.json`

The benchmark form is:

```bash
# Replace CONFIG and TAG with one of:
#   experiments/car-mini-3r-robust-seed11023-plan.json  / car-mini
#   experiments/cross7-4r-robust-seed0-plan.json        / cross7
#   experiments/chair8-4r-mixed-robust-seed1-plan.json  / chair8
python3 main.py --config CONFIG --solve \
  --no-open-witness --optimize-witnesses off --witness-sets 0 \
  --no-fuse-cograsp --quiet-rai --no-view \
  --save experiments/bench-TAG-legacy-plan.json

python3 main.py --config CONFIG --solve \
  --open-witness --open-witness-batch 7 --open-witness-epochs 2 \
  --optimize-witnesses off --witness-sets 0 --no-fuse-cograsp \
  --quiet-rai --no-view --save experiments/bench-TAG-open-b7e2-plan.json
```
