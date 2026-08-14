# mr_bench integration notes — issues & recommended changes

Collected while wiring the assembly planner's exported plans into `mr_bench` (used as a
module, unmodified). Everything below is a change **on the mr_bench side** that would make it
easier to consume as a library. Grouped by severity. File/line refs are from the current tree.

---

## A. Correctness bugs (block real use)

### A1. numpy ≥ 2 breaks the tree planners (BiRRT / RRT*)  — FIXED (2026-07-21)
`env.batch_config_cost` returns a **size-1 array** for single-pair queries. BiRRT/RRT*
`extend`/`connect` wrote that straight into the scalar per-node cost array via
`Node.cost.setter`. numpy < 2 broadcast a size-1 array into a scalar slot silently; **numpy ≥ 2
raised** `ValueError: setting an array element with a sequence` -- every tree planner unusable
(CompositePRM avoids the path, which hid it). NB the numpy-2 floor is **scipy>=2.0.0** (numba
only caps at `<2.5`), so downgrading numpy is not an option -- had to fix the code.
- **Fixed mr_bench-side** (`rrtstar_base.py` Node.cost.setter): `if isinstance(value, np.ndarray):
  value = value.item() if value.size == 1 else value`. Single choke point -> robust to array
  leaks from any path; `isinstance` fast-paths the scalar common case (~free vs collision-check
  runtime). Chosen over scalarizing at call sites (zero overhead but fragile -- must find every
  leak site; A4 showed they're spread across extend/connect/tree_extension).
- OUR SHIM REMOVED: `_patch_mrbench_numpy2` deleted from mp_export; `run_birrt` no longer calls
  it. Verified BiRRT optimize=False AND optimize=True (the crash path) reach terminal. NOTE the
  SEPARATE `_patch_mrbench_optimize` shim STAYS -- that's A4 (near/tree_extension index bug),
  still unfixed upstream.

### A2. `delete_visual_only_frames` orphans non-visual children  — FIXED (2026-07-21, made optional)
`rai_env.__init__`'s `delete_visual_only_frames` deletes every mesh-shaped frame. The robotiq's
**collision** shapes (finger capsules, inner-finger pad boxes) are *children of mesh frames*, so
deleting the mesh orphaned them (`parent=None`) and they floated, causing phantom collisions.
- **Fixed mr_bench-side (two parts):** (1) `delete_visual_only_frames` is now COLLISION-AWARE
  -- `visual_only(f) = shape=="mesh" and not contact`; it keeps every non-visual-only frame AND
  all its ancestors, deletes the rest of the pure-visual mesh subtrees. So collision capsules
  (and collision MESHES) survive with their mesh parents -- nothing orphans -- while genuinely
  decorative meshes are still pruned (measured: car-mini 223 frames -> 172 collision, -51).
  (2) `rai_env.__init__(self, delete_visual_only=True)` keyword (default True) as a coarse
  override to keep the whole visual scene in the collision config if ever wanted.
- OUR FINGER RE-PARENT HACK REMOVED: mp_export.make_problem calls plain `rai_env.__init__(self)`
  (default True now safe). Verified: fingers kept + attached, no orphans, start collision-free
  (finger-vs-palm penetration does NOT resurface), cross+car-mini reach terminal.
  (A5 handled the shapeless-part crash separately; ssBox baking also removed -- see A5.)

### A3. Eager import of the VAMP extension breaks the whole package  — FIXED (2026-07-21)
`problems/__init__.py` did `from . import mr_vamp_env` gated by `find_spec("mr_planner_core")`.
The compiled `mr_planner_core` (built for py3.10) **is found on disk** in a 3.12 venv but fails
to load (and `mr_vamp_env` also imports `meshcat`); `find_spec` present ≠ importable, so
importing *anything* from `problems` exploded.
- **Fixed mr_bench-side:** `problems/__init__.py` now guards `mr_vamp_env` / `pinocchio_env` /
  `mujoco_env` with `try/except ImportError` (no `find_spec`); the `mr_vamp_env` block pops the
  py3.10 `dist-packages` path in a `finally` so a failed import doesn't leak it onto `sys.path`.
- OUR STUB REMOVED: `mp_export._ensure_mrbench` no longer pre-seeds `sys.modules` — just adds
  the path and imports. Verified: clean import with no stub (`mr_vamp_env` correctly skipped),
  full cross motion-plan runs to terminal.

---

### A4. RRT* optimization path (`optimize=True`) crashes in `tree_extension`/`near`  — FIXED (2026-07-21)
**FIX (upstream, 1 guard in `near()`):** `if np.size(n_nearest_idx) and n_nearest_idx not in
indices:` (was `if n_nearest_idx not in indices:`). Skips the insert when the `np.where` lookup
is empty (a mode-switch transition node whose parent is in another mode's tree); the scalar and
found-tuple cases fall through unchanged. Restores the pre-numpy-1.25 behavior explicitly (old
numpy returned scalar False for the mismatched `==`, which happened to skip the insert = correct).
OUR SHIM `_patch_mrbench_optimize` REMOVED from mp_export. Verified optimize=True reaches terminal
on SEQUENCE cross (the crash path) + car-mini, no broadcast error. Details below.

`tree_extension` (`rrtstar_base.py:1766,1792`) computes `index = np.where(get_node_ids_subtree()
== parent.id)` -- which returns a **tuple of arrays** -- and passes it straight into
`near(mode, node, index)` whose 3rd arg is typed `n_nearest_idx: int` (everywhere else it's the
scalar from `self.nearest`). At `near` L1331 `if n_nearest_idx not in indices:` then broadcasts a
multi-element array (e.g. `(11,)`, `parent.id` matching several padded/zero id slots) against an
empty near-set `(1,0)` -> `ValueError: operands could not be broadcast together`. Only reached
with `optimize=True` (the rewiring/path-improvement loop), so `optimize=False` avoids it.
numpy ≥ 2's stricter broadcasting turns it into a hard crash. NOT related to A1 (that patch only
touches `Node.cost`).
- CONFIRMED EXACTLY (instrumented, all 88 near-calls in a cross-5 SEQUENCE optimize run):
  **miss ⟺ node.state.mode != parent.state.mode** (the config is AT a mode switch). 16/16 misses
  are transitions; 72/72 non-misses are same-mode; zero exceptions either way. WHY: nodes are
  add_node'd to their OWN mode's tree, so a node's parent (prev path element) is in the node's
  tree iff they share a mode; at a mode boundary the transition node's parent is in the PREVIOUS
  mode's tree -> empty np.where. SEMANTICS: in the else branch node.parent is ALREADY the fixed
  cross-mode transition edge (node.parent=parent + convert_node_to_transition_node); the near()
  there is for REWIRE over node's SAME-MODE neighbours, and the cross-mode parent is not one --
  so returning just the radius ball for a transition node is CORRECT, not a band-aid. The empty
  np.where IS the "this is a mode switch" signal.
- ROOT CAUSE (RE-DIAGNOSED 2026-07-21, instrumented -- my first A/B-subtree guess was WRONG):
  it is a CROSS-MODE lookup, not a bidirectional A-vs-B mix-up (ruled out: at the misses subtree
  B was EMPTY and nodes go to A). `tree_extension` fixes `mode = discretized_path[0].mode` and
  accumulates `parent = node` down the path. Every miss is in the ELSE branch (`mode !=
  node.state.mode`) i.e. where the discretized path CROSSES A MODE BOUNDARY: it then does
  `np.where(self.trees[node.state.mode].get_node_ids_subtree() == parent.id)` -- searching the
  NEW mode's tree for a `parent` that lives in the PREVIOUS mode's tree -> parent absent
  (inDict=False in both A and B of the new mode) -> empty `np.where` -> near() broadcast crash.
  Instrumentation: reliable near-only run (reached terminal) shows misses only on SEQUENCE (0 on
  DEPGRAPH here), B empty; a fuller tree_extension repro shows same_mode=False + parent not in
  the current-mode tree at every miss (that repro diverged once near stopped crashing, so trust
  the mechanism/pattern, not its exact counts).
- **Fix (final, in `near()` only):** `near(node.state.mode, ...)` is CORRECT -- it rewires
  `node` among its OWN-mode neighbours, and `node.state.mode` is the node's own tree. For a
  transition node `parent` is a cross-mode node (edge set via `node.parent = parent` +
  `convert_node_to_transition_node`), so it legitimately isn't in this tree -- it is NOT a valid
  same-mode rewire candidate. `near()` already does "near-set = radius ball ∪ {n_nearest_idx}"
  (`if n_nearest_idx not in indices: np.insert`), but n_nearest_idx arrives as a `np.where`
  TUPLE (not the scalar `self.nearest` returns), and when parent is absent it's EMPTY. So the
  whole fix is: coerce `n_nearest_idx = np.where(...)[0]` to a scalar and skip the insert when
  empty. NO `tree_extension` change (my earlier "look up in `self.trees[parent.state.mode]`"
  was WRONG -- near returns indices INTO node.state.mode's tree, a parent-mode index is
  meaningless there).
- Our SHIM (`mp_export._patch_mrbench_optimize`) already IS this fix (scalarize + skip empty),
  cause-agnostic. Verified `optimize=True` reaches terminal on SEQUENCE + DEPGRAPH. Once the
  `near()` fix lands upstream, drop the shim (A1/A2/A3/A5 pattern).

### A5. `set_to_mode` assumes the attached frame has geometry -> FCL crash on shapeless parts  — HIGH
On every attach, `set_to_mode` does `tmp.getFrame(task.frames[1]).setContact(-1)`
(`rai_base_env.py:805`). If `frames[1]` is a **shapeless** frame (`ST.none`), rai fabricates a
DEGENERATE `ST.mesh` with **0 vertices**, and the next `computeCollisions()` aborts the process:
`kin.cpp:fcl CHECK failed: 'f->shape->mesh().V.N' -- collision object with no vertices`. This
bites any part whose collision geometry lives on a CHILD frame (e.g. our car-mini wheels: the
part `obj_X` is `ST.none`, the rotated cylinder is on `obj_X__shape`). It compounds with A2:
`delete_visual_only_frames` also deletes every `"mesh"`-shaped frame, so you can't just bake a
mesh onto the part either -- the part gets deleted.
- **Fixed mr_bench-side (2026-07-21):** `set_to_mode` now guards the setContact --
  `frame = tmp.getFrame(frames[1]); if frame.getShapeType() != ry.ST.none and
  frame.info().get("contact"): frame.setContact(-1)` -- so a shapeless attached frame is left
  alone (no 0-vertex mesh). (First attempt had `f.info()` with `f` undefined + missing `ry.`
  on `ST` -> NameError; corrected.)
- OUR ssBox-BAKING WORKAROUND REMOVED. Key realisation (user caught my error): the wheel's
  collision was NEVER on `obj_X` -- it's on the `obj_X__shape` CHILD (real cylinder, contact=1).
  With the guard, `obj_X` stays shapeless (skipped, no crash) and the child keeps its cylinder
  collision -> the wheel still collides (with the world, not its carrier -- `setContact(-1)` is
  "collide except with parent", NOT collision-free; verified empirically). So the guard-only
  path is not just simpler but MORE ACCURATE (real cylinder vs our AABB ssBox). Verified: NO
  ssBox, car-mini reaches terminal (seq), cross unaffected; `obj_1__shape` cylinder collidable.

## B. Off-the-shelf usability (the main ask)

### B1. No importable "just plan this env" function  — HIGH (requested)
Every planner is constructed differently, each with its own config class and its own
`plan(ptc, optimize)` quirks:
- `CompositePRM(env, CompositePRMConfig()).plan(ptc, optimize=False)`
- `BidirectionalRRTstar(env, BaseRRTConfig()).plan(ptc, optimize=False)`
- `AITstar(env, BaseITConfig())`, `EITstar(env, BaseITConfig())`, `RRTstar(env, BaseRRTConfig())`
…and shortcutting is a *separate* call (`robot_mode_shortcut(env, path, max_iter=...)`).
`scripts/run_planner.py` exists but is a **CLI**, not a library entry point.
- **Fix:** a single public function, e.g.
  ```python
  def solve(env, planner="birrt", runtime=60.0, optimize=False,
            shortcut=True, shortcut_iters=1000, seed=None) -> (path, info)
  ```
  that maps a planner **name -> (class, default config)** via a registry, builds the PTC, runs
  it, and optionally shortcuts. This is exactly the ~30 lines we had to write (`run_birrt`);
  it belongs in the library so every consumer doesn't re-derive it.

### B2. Shortcutting has no time budget  — MEDIUM (looks "stuck")
`robot_mode_shortcut` runs a fixed loop (`max_iter=1000` successes, `max_attempts=2500`,
`shortcutting.py:117`) with **no PTC / time bound**. `runtime` only bounds the tree search, so
on a long path shortcutting silently dominates wall-clock and looks hung.
- **Fix:** accept a `PlannerTerminationCondition` (or `max_time`) in `robot_mode_shortcut`, and
  have the `solve()` of B1 pass a shared time budget. At minimum, document that `runtime` does
  not cover shortcutting.

### B2b. Shortcutting collision resolution is 10x finer than the planner's — MEDIUM (10-90x slower)
`robot_mode_shortcut(..., resolution=0.001)` hardcodes a **1mm/1mrad** default, while the env's
`collision_resolution` (what BiRRT itself uses for edge checks) is **0.01**. Each shortcut
candidate is collision-checked via `is_path_collision_free` -> `is_edge_collision_free`, whose
cost is `N = config_dist/resolution` checks **per edge** (`rai_base_env.py:618`); on an
already-densely-interpolated path this is the "horrible" case the code's own TODO flags
(`shortcutting.py:215-217`). Measured on cross-5/3R, 150 iters, IDENTICAL result cost:
`0.001 -> 46.0s`, `0.01 -> 4.6s (10x)`, `0.05 -> 1.0s (47x)`, `0.1 -> 0.5s (90x)`. So the fine
default buys **nothing** and costs 10x+.
- **Fix:** default `resolution` to `env.collision_resolution` (don't check finer than the
  planner did), and address the TODO (strip interpolated nodes before checking, so edges
  aren't length-2). We now pass `resolution=env.collision_resolution` from `run_birrt`.
- This is the "suspiciously slow shortcutting" — it was collision resolution, not the env.

### B2c. No per-mode "held" / grasp constraint — MEDIUM (forces joint-goal hacks)
There's no way to say "robot R keeps holding object O (relative pose fixed) during this mode"
without giving R a full **joint goal**. We hit this three times and faked each with a joint
goal pinning the holder at a specific config:
- **assemble** while another arm holds the base,
- **release** where a second arm keeps holding the object (else the object drifts to the idle
  holder's hand),
- **add_grasp** braces (currently a `goto`).
A joint goal OVER-constrains (forces an exact config instead of "any valid hold"), which can
make those keyframes needlessly hard/infeasible.
- **Fix:** a first-class per-mode **relative-pose grasp constraint** (gripper<->object rigid)
  that pins the *relative transform*, not the arm's joints. mr_bench appears to have a
  constraint branch already (referenced for the brace case) — expose it for these.

### B3. No documented env-construction contract  — MEDIUM
Building a custom env required reverse-engineering: `Task(name, robots, SingleGoal, type,
frames)`; `type` in {`pick`,`place`,`handover`,`goto`,...} with `goto` = no attach and others
= `C.attach(frames[0], frames[1])` (child <- parent); the `SequenceMixin` vs
`DependencyGraphMixin` init order (`self.tasks` -> `self.sequence`/`self.graph` ->
`BaseModeLogic.__init__`); the per-mode scene-graph reconstruction. A short "authoring a
problem" doc + a minimal example subclass would remove most of the guesswork.

---

## C. Robustness footguns

### C1. `set_to_mode` freezes attach transforms per mode-chain — order-dependent (relinking)  — MEDIUM
`set_to_mode` (`rai_base_env.py:704`) rebuilds a mode by replaying **one `attach` per mode
transition** along the `prev_mode` chain, freezing each child's relative pose **at the config
that chain reached**. In a `DependencyGraphMixin` problem, different linearizations reach a
given attach at different configs, so the reconstructed (frozen) transforms can drift between
equivalent modes. Consumers must guarantee the *held/idle* robots are posed identically across
every linearization for this to be consistent — there's no invariant enforcing it.
- **Suggestion:** make the attach relative-pose deterministic w.r.t. the *task goal* rather
  than the arrived-at chain config, or document the invariant a well-posed problem must satisfy.

### C2. `_make_sequence_from_names` assumes unique task names, silently  — FIXED (2026-07-21)
`planning_env.py:1049` / `1222` append an index for **every** task whose name matches — if two
tasks share a name you get duplicate/aliased sequence entries with no error.
- **Fixed:** `if len({t.name for t in self.tasks}) != len(self.tasks): raise ValueError(...)` at the
  top of BOTH copies (SequenceMixin :1050 = called by env setup like our make_problem;
  DependencyGraphMixin :1231 = its only INTERNAL user, guards graph/tasks name consistency).

### C3. `_make_start_mode_from_sequence` KeyErrors on a robot with no task  — FIXED (2026-07-21, 1 site left)
It builds `mode_dict[r]` from tasks and then indexes `mode_dict[r]` for every robot; a robot
that appears in **no** task raises `KeyError`. (Extra idle arms are a normal case for us.)
- **Fixed (strict):** guard `if r not in mode_dict: raise ValueError("Robot ... appears in no
  task ...")` at get_start_task_ids :1081, _make_start_mode :1262, _make_terminal :1286. STILL
  UNGUARDED: SequenceMixin.make_symbolic_end :1108 (same one-line guard needed). We never hit
  this (our terminal task lists ALL robots -> mode_dict always populated).

### C4. `DependencyGraphMixin._verify_graph` is a stub (`return True`)  — LOW
`planning_env.py:1272`. It never validates the "single leaf / one root per robot" invariants it
documents, so a malformed graph fails later with an opaque error instead of at construction.

---

## D. Visualization / debugging

### D1. `visualize_modes` homes idle robots  — MEDIUM (misleading for manip)
`scripts/show_problems.py:92-106` poses switching robots at their goal and **every idle robot
at HOME**. For a manipulation env an idle arm is often *holding* a part — homing it drags the
held part to the arm's home pose, so the view looks broken when the plan is fine. This cost us
real debugging time.
- **Fix:** an option to keep idle robots at their mode `entry_configuration` instead of home.

### D2. `visualize_modes` picks the next task with unseeded `random`  — LOW
`show_problems.py:76` `random.randint(...)`; for a DAG with multiple valid next combos the
walk is nondeterministic and unseeded.

### D3. `is_collision_free` error message on a raw ndarray  — LOW
Passing a bare `np.ndarray` (instead of a `Configuration`) fails deep inside with
`'numpy.ndarray' object has no attribute 'state'`. A type check + clear message would help.

---

## E. Packaging / dependencies

### E1. Undeclared / conflicting deps  — MEDIUM
`numba`, `sortedcontainers`, `simple-parsing` were needed but not installed by the package
metadata; installing `numba` also force-**downgraded numpy 2.5 -> 2.4**, which then interacts
with A1. Pin a compatible `numpy`/`numba` pair (or make `numba` optional) and declare the
`sortedcontainers` / `simple-parsing` runtime deps.

---

## Quick-win order
A1 (numpy setter, 1 line) → A3 (import guard) → A2 (re-parent on delete) → B1 (`solve()` fn) →
B2 (shortcut time budget) → D1 (idle-arm viz). A1/A2/A3 are the ones we had to work around from
our side to use the module at all.
