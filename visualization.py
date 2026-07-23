"""Viewer helpers: schematic state/plan views and the real-arm IK replay, plus the
per-skill IK popups. Depends on both the problem definition and the solver."""

import robotic as ry
import numpy as np

from env import *
from problem import *
from solver import *
import problem as _problem
import solver as _solver
globals().update({k: v for k, v in vars(_problem).items() if not k.startswith("__")})
globals().update({k: v for k, v in vars(_solver).items() if not k.startswith("__")})


def visualize_dependency_graph(assembly, path=None, view=False):
  """Render the assembly's PRECEDENCE / mate structure as a layered DAG (graphviz `dot`),
  so the build order the model actually enforces is visible at a glance. Edge kinds:
    * solid black  : mate-parent (the geometric weld a part seats onto -- also a precedence),
    * dashed grey  : an EXTRA dependency (a precedence that is not this part's mate parent,
                     e.g. a plate's diamond corner needing its second in-plane neighbour),
    * bold red     : a CONNECT mate (a module join; label shows connect_requires).
  Arrows point prerequisite -> dependent (build flows along the arrows); the base is pinned
  to the bottom rank. Returns the output PNG path. Needs pydot + the graphviz `dot` binary.

  Also prints the adjacency as text (works with no display / no graphviz)."""
  import os
  base = assembly.base
  connect = getattr(assembly, "connect_parts", set())
  requires = getattr(assembly, "connect_requires", {})

  # --- text form: parts and their precedences (always available) ---
  print(f"dependency graph: {type(assembly).__name__} ({len(assembly.objects)} parts, "
        f"base={base})")
  for p in assembly.objects:
    rel = assembly.relative_pose(p)
    mate = rel[0] if rel else None
    deps = [d for d in assembly.dependencies_of(p) if d != mate]
    tag = "  [CONNECT" + (f" needs {sorted(requires.get(p, set()), key=_part_index)}]"
                          if requires.get(p) else "]") if p in connect else ""
    parts = []
    if mate is not None:
      parts.append(f"mate<-{mate}" + (" (connect)" if p in connect else ""))
    if deps:
      parts.append(f"dep<-{deps}")
    print(f"  {p}: {', '.join(parts) if parts else '(base / free)'}{tag}")

  # --- graphviz render ---
  try:
    import pydot
  except ImportError:
    print("[dep-graph] pydot not available -- text form only")
    return None
  # MODULES = connected components once the connect (join) edges are removed. Each is a
  # self-contained order ideal -- a sub-assembly buildable on its own. Drawing them as
  # clusters makes "how many independent pieces, joined how" obvious.
  comp = {p: p for p in assembly.objects}

  def find(x):
    while comp[x] != x:
      comp[x] = comp[comp[x]]
      x = comp[x]
    return x

  def union(a, b):
    comp[find(a)] = find(b)
  for p in assembly.objects:
    rel = assembly.relative_pose(p)
    if rel is not None and p not in connect:
      union(p, rel[0])
    for d in assembly.dependencies_of(p):
      union(p, d)
  modules = {}
  for p in assembly.objects:
    modules.setdefault(find(p), []).append(p)

  g = pydot.Dot(graph_type="digraph", rankdir="BT", splines="true", nodesep="0.4",
                compound="true")
  for i, (root, members) in enumerate(sorted(modules.items(),
                                             key=lambda kv: min(_part_index(m) for m in kv[1]))):
    sub = pydot.Cluster(f"m{i}", label=(f"module {i}" if len(modules) > 1 else ""),
                        color="#8899bb", style="rounded")
    for p in sorted(members, key=_part_index):
      fill = "#cde6ff" if p == base else ("#ffe0cc" if p in connect else "#eeeeee")
      sub.add_node(pydot.Node(p, style="filled", fillcolor=fill, shape="box",
                              fontname="Helvetica"))
    g.add_subgraph(sub)
  # pin the base to the bottom rank
  for p in assembly.objects:
    rel = assembly.relative_pose(p)
    mate = rel[0] if rel else None
    if mate is not None:
      if p in connect:
        req = sorted(requires.get(p, set()), key=_part_index)
        g.add_edge(pydot.Edge(mate, p, color="red", penwidth="2.2",
                              label="connect" + (f"\\nneeds {req}" if req else ""),
                              fontcolor="red", fontsize="9"))
      else:
        g.add_edge(pydot.Edge(mate, p, color="black"))
    for d in assembly.dependencies_of(p):
      if d != mate:
        g.add_edge(pydot.Edge(d, p, color="grey40", style="dashed"))

  if path is None:
    path = os.path.join(os.path.dirname(__file__),
                        f"depgraph_{type(assembly).__name__}.png")
  g.write_png(path)
  print(f"[dep-graph] wrote {path}")
  if view:
    try:
      import matplotlib.pyplot as plt
      import matplotlib.image as mpimg
      plt.figure(figsize=(6, 8)); plt.imshow(mpimg.imread(path)); plt.axis("off")
      plt.tight_layout(); plt.show()
    except Exception as e:
      print(f"[dep-graph] view failed ({e}); open {path} manually")
  return path


def display_final_assembly(C, assembly, base_pose):
  C_final = ry.Config()
  C_final.addConfigurationCopy(C)

  for obj, parent in assembly.assembly_relative_poses.items():
    C_final.attach(
        parent[0],
        obj,
    )

    C_final.getFrame(obj).setRelativePosition(parent[1])
    # ASSEMBLED part frames are axis-aligned (the shape's own rotation lives on the __shape
    # child); reset the relative orientation so a SCATTER orientation set for the table layout
    # (e.g. legs laid flat) does not leak into the assembled view.
    C_final.getFrame(obj).setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])

    # keep the object's color, just lower its opacity (shape may live on a __shape child).
    # THIN shapes (cylinders/spheres) get a higher alpha: at 0.3 a thin leg has so little
    # material along any view ray that it nearly vanishes next to a 0.3 box.
    sf = C_final.getFrame(part_shape_frame(C_final, obj))
    mesh_colors = np.asarray(sf.getMeshColors(), dtype=float)
    rgb = (mesh_colors.reshape(-1, mesh_colors.shape[-1])[0][:3]) / 255.0
    a = 0.7 if assembly.part_shape(obj)[0] in ("cylinder", "sphere") else 0.3
    sf.setColor([rgb[0], rgb[1], rgb[2], a])

  C_final.getFrame(assembly.base).setPosition(base_pose)
  C_final.getFrame(assembly.base).setQuaternion([1.0, 0.0, 0.0, 0.0])

  C_final.view(True)

def show_subassemblies(C, assembly, subassemblies, base_pose):
  for subassembly in subassemblies:
    if len(subassembly) == 1:
      continue 

    C_final = ry.Config()
    C_final.addConfigurationCopy(C)

    for obj, parent in assembly.assembly_relative_poses.items():
      if obj in subassembly:
        C_final.attach(
            parent[0],
            obj,
        )

        C_final.getFrame(obj).setRelativePosition(parent[1])
        C_final.getFrame(obj).setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])  # see display_final_assembly

      # keep the object's color, just lower its opacity (shape may live on a __shape child;
      # thin cylinders/spheres get a higher alpha -- see display_final_assembly)
      sf = C_final.getFrame(part_shape_frame(C_final, obj))
      mesh_colors = np.asarray(sf.getMeshColors(), dtype=float)
      rgb = (mesh_colors.reshape(-1, mesh_colors.shape[-1])[0][:3]) / 255.0
      a = 0.7 if assembly.part_shape(obj)[0] in ("cylinder", "sphere") else 0.3
      sf.setColor([rgb[0], rgb[1], rgb[2], a])

    C_final.getFrame(assembly.base).setPosition(base_pose)

    C_final.view(True)

def visualize_mate_sequence(C, assembly, base_pose=(0.0, 0.0, 0.6), recorder=None,
                            view=True):
  """SCHEMATIC mate-sequence view (no planner): walk the DAG's assembly_order, attaching
  each part at its assembled pose, and at every step draw the MATE being formed --
    * green crosses  : the mate's contact interface point(s) (mate_interface_points),
    * orange arrow   : the insertion PRESS the seating must exert (insert_force along the
                       mate axis, at the interface),
    * blue arrow     : the MOMENT that press makes about the joined part's centre (nonzero
                       exactly when the interface is OFF the press axis -- the out-of-hand
                       torque case), lengths proportional (see _WF_SCALE etc).
  Structure only -- reads the DAG + geometry + wrench.insert_force, runs no search. With a
  ViserRecorder it records the steps (returns it, serve() for the web view); else it opens
  the blocking rai viewer per step. A quick way to sanity-check interface points and joint
  loads for a new assembly before ever solving it."""
  import wrench as _w
  base = assembly.base
  order = [p for p in assembly_order(assembly) if p != base]

  C_seq = ry.Config()
  C_seq.addConfigurationCopy(C)
  # start from the disassembled scene but PARK every part far away; reveal them as mated.
  for p in assembly.objects:
    C_seq.getFrame(p).setPosition([0.0, -900.0 - 3.0 * _part_index(p), 1.0])
    C_seq.getFrame(p).setQuaternion([1.0, 0.0, 0.0, 0.0])
  C_seq.getFrame(base).setPosition(list(base_pose))
  C_seq.getFrame(base).setQuaternion([1.0, 0.0, 0.0, 0.0])

  def _mate_vectors(part):
    # the press + its moment for the just-formed mate, at the drawn geometry.
    d = assembly.mate_axis(part)
    if d is None:
      return []
    d = np.asarray(d, dtype=float)
    R = _quat_to_R(np.asarray(C_seq.getFrame(part).getQuaternion(), dtype=float))
    f = _w.insert_force(part) * (R @ d)                    # world press ON the target
    c = np.asarray(C_seq.getFrame(part).getPosition(), dtype=float)
    out = []
    for off in assembly.mate_interface_points(part, d):
      pt = c + R @ off
      fn = np.linalg.norm(f)
      if fn > 1e-9:
        L = min(_WMAX, _WF_SCALE * fn)
        out.append((pt, pt + f / fn * L, _WP_RGB))         # press arrow (orange)
      m = np.cross(pt - c, f)                              # moment about the part centre
      mn = np.linalg.norm(m)
      if mn > 1e-9:
        L = min(_WMAX, _WT_SCALE * mn)
        out.append((c, c + m / mn * L, _WT_RGB))           # moment arrow (blue)
    return out

  def show(label, mated, new=None):
    vecs = _mate_vectors(new) if new is not None else []
    ifs = []
    for p in mated:
      if p == base:
        continue
      d = assembly.mate_axis(p)
      if d is None:
        continue
      R = _quat_to_R(np.asarray(C_seq.getFrame(p).getQuaternion(), dtype=float))
      cp = np.asarray(C_seq.getFrame(p).getPosition(), dtype=float)
      for off in assembly.mate_interface_points(p, d):
        pt = cp + R @ off
        for ax in np.eye(3):
          ifs.append((pt - ax * _IF_R, pt + ax * _IF_R, _IF_RGB))
    if recorder is not None:
      recorder.record(C_seq, label, vectors=vecs, interfaces=ifs)
    elif view:
      C_seq.view(True, label)

  show(f"0: base {base}", [base])
  mated = [base]
  for i, p in enumerate(order, 1):
    parent = assembly.relative_pose(p)[0]                  # attach at the assembled pose
    C_seq.getFrame(p).setPosition(
        (np.asarray(C_seq.getFrame(parent).getPosition(), dtype=float)
         + assembly.position_in_base(p) - assembly.position_in_base(parent)))
    C_seq.getFrame(p).setQuaternion([1.0, 0.0, 0.0, 0.0])
    mated.append(p)
    d = assembly.mate_axis(p)
    fmag = _w.insert_force(p)
    m = 0.0
    if d is not None:
      dd = np.asarray(d, dtype=float)
      for off in assembly.mate_interface_points(p, dd):
        m = max(m, float(np.linalg.norm(np.cross(off, dd))) * fmag)
    print(f"[mate {i}/{len(order)}] {p} <- {parent}  press={fmag:.1f}N  peak-moment={m:.3f}Nm")
    show(f"{i}: mate {p} <- {parent}", mated, new=p)
  return recorder


def visualize_grasps_viser(C, assembly, recorder=None, view=True, useful=None):
  """GRASP CATALOG (scrubbable): one step per part, its shape rendered mid-air with a
  free-floating EE at every candidate grasp of the part's DB. Coloring: if `useful` (a
  {part: set of mate-viable grasp ids}, e.g. solver.compute_useful_grasps) is given, green =
  mate-viable / grey = not; else green = FLAT surface / orange = CURVED (part_grasp_surfaces).
  A quick way to eyeball what grasps exist per part and which can seat the mate. With a
  ViserRecorder it records + returns it (serve() for the web view); else blocking rai view."""
  import problem as _p
  gpath = _scene_floating_ee_path(C)
  base = np.array([0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])     # part parked mid-air, identity
  GREEN, ORANGE, GREY = [0.2, 0.8, 0.3, 1], [0.95, 0.55, 0.1, 1], [0.6, 0.6, 0.6, 1]

  def show(part):
    Cv = make_world()
    add_part_frame(Cv, part, assembly.part_shape(part))
    Cv.getFrame(part).setPosition(base[:3]); Cv.getFrame(part).setQuaternion(base[3:])
    spec = assembly.part_shape(part)
    surf = _p.part_grasp_surfaces(spec)
    ids = list(range(len(_p.part_grasp_poses(spec))))
    U = (useful or {}).get(part)
    for gid in ids:
      gw = _pose_compose(base, grasp_pose(assembly, part, gid))
      pref = f"g{gid}_"
      place_floating_ee(Cv, gw, gpath, prefix=pref)
      col = (GREEN if gid in U else GREY) if U is not None else \
          (GREEN if surf[gid] == "flat" else ORANGE)
      color_gripper(Cv, pref, col)
    nflat = sum(1 for s in surf if s == "flat")
    tag = f" useful={sorted(U)}" if U is not None else f" ({nflat} flat, {len(ids)-nflat} curved)"
    label = f"{part} [{spec[0]}]: {len(ids)} grasps{tag}"
    if recorder is not None:
      recorder.record(Cv, label)
    elif view:
      Cv.view(True, label)
    print(f"[grasp-viz] {label}")

  for part in assembly.objects:
    show(part)
  return recorder


def visualize_rest_poses_viser(C, assembly, recorder=None, view=True, groups=None):
  """REST-POSE CATALOG (scrubbable): for each group, one step per STABLE rest face, the group
  laid out resting on the table in that pose (rest_faces + _rest_layout_poses -- exactly what
  the search/checker use). Default groups: every singleton, each connect-module, the stool,
  and the full assembly -- so an EMPTY rest set (an unparkable subassembly, e.g. a bad
  backrest module) is immediately visible as a group with zero steps. Records + returns the
  recorder (serve() for viser); else blocking rai view."""
  base = make_world()
  ttop = base.getFrame("table").getPosition()[2] + base.getFrame("table").getSize()[2] / 2.0
  objs = list(assembly.objects)
  if groups is None:
    groups = [(f"{p}", {p}) for p in objs]                 # singletons
    for r in getattr(assembly, "connect_parts", set()):
      groups.append((f"module<{r}>", {p for p in objs if r in _mate_closure(assembly, p)} | {r}))
    stool = {p for p in objs if assembly.relative_pose(p) and
             assembly.relative_pose(p)[0] == assembly.base} | {assembly.base}
    groups.append(("base+direct", stool))
    groups.append(("FULL", set(objs)))

  n = 0
  for name, grp in groups:
    faces = sorted(rest_faces(assembly, grp))
    if not faces:
      print(f"[rest-viz] {name} ({len(grp)} parts): NO REST POSE (unparkable)")
      continue
    ref = reference_part(grp)
    anchor_q = assembly.scatter_orientations.get(ref, np.array([1.0, 0.0, 0.0, 0.0]))
    for f in faces:
      Cv = make_world()
      poses = _rest_layout_poses(assembly, grp, ref, f, anchor_q)
      for p, pose in poses.items():
        add_part_frame(Cv, p, assembly.part_shape(p))
        Cv.getFrame(p).setPosition(np.asarray(pose[:3]) + [0, 0, ttop])
        Cv.getFrame(p).setQuaternion(pose[3:])
      label = f"{name} ({len(grp)}p): rest face {f} of {faces}"
      if recorder is not None:
        recorder.record(Cv, label)
      elif view:
        Cv.view(True, label)
      n += 1
    print(f"[rest-viz] {name} ({len(grp)} parts): rest faces {faces}")
  print(f"[rest-viz] {n} rest poses total")
  return recorder


def visualize_grasps(C, grasps, assembly, subassembly_base_pose, gripper_path=None):
  # for each subassembly, show its candidate grasps by placing a free-floating
  # gripper (the EE model matching the scene's gripper) at every grasp pose.
  # grasps: {subassembly (tuple): [pose, ...]}, each pose in the first obj's frame.
  if gripper_path is None:
    gripper_path = _scene_floating_ee_path(C)

  for subassembly, subassembly_grasps in grasps.items():
    C_viz = ry.Config()
    C_viz.addConfigurationCopy(C)

    # build the subassembly at base_pose (inspired by show_subassemblies)
    for obj, parent in assembly.assembly_relative_poses.items():
      if obj in subassembly and parent[0] in subassembly:
        C_viz.attach(parent[0], obj)
        C_viz.getFrame(obj).setRelativePosition(parent[1])

    # grasps are expressed in the first obj's frame; move it to base_pose
    ref = subassembly[0]
    C_viz.getFrame(ref).setPosition(subassembly_base_pose)
    ref_pose = np.asarray(C_viz.getFrame(ref).getPose())

    # drop a free-floating gripper at each grasp pose (grasps are in the ref frame)
    for i, grasp in enumerate(subassembly_grasps):
      grasp_world = _pose_compose(ref_pose, np.asarray(grasp))
      place_floating_ee(C_viz, grasp_world, gripper_path, prefix=f"grasp{i}_")

    C_viz.view(True)


def _scene_floating_ee_path(C):
  # the floating-EE model MATCHING the scene's real gripper (two-finger vs vacuum, detected
  # from the frame names). Replays used to fall back to DEFAULT_FLOATING_EE (the vacuum
  # file) and drew vacuum pens on two-finger runs.
  import os
  vacuum = any(n.endswith("_ur_vacuum") for n in C.getFrameNames())
  return os.path.join(os.path.dirname(__file__), UR10_PATHS[(True, vacuum)])


def visualize_state(C, state, assembly, gripper_path=None, view=True):
  """Render a search State: parts placed per the scene graph, plus a free-floating EE
  at the grasp of every held subassembly (the EE model matching the scene's gripper,
  unless an explicit gripper_path overrides). Returns the built config (views it if view)."""
  if gripper_path is None:
    gripper_path = _scene_floating_ee_path(C)
  C_viz = ry.Config()
  C_viz.addConfigurationCopy(C)

  part_poses = _state_part_poses(state, assembly)
  for p, pose in part_poses.items():
    f = C_viz.getFrame(p)
    f.setPosition(pose[:3])
    f.setQuaternion(pose[3:])

  # a floating gripper at EVERY grasp contact (kinematic hold AND braces), each placed
  # on the part it actually grasps, in that part's real world pose.
  for p in state.bodies():
    for c in state.supports.get(p, ()):
      if c[0] != "grasp":
        continue
      robot, gpart, grasp_id = c[1], c[2], c[3]
      grasp_world = _pose_compose(np.asarray(part_poses[gpart], dtype=float),
                                  grasp_pose(assembly, gpart, grasp_id))
      place_floating_ee(C_viz, grasp_world, gripper_path, prefix=f"ee_{robot}_{gpart}_")

  if view:
    C_viz.view(True)
  return C_viz


def visualize_successor_states(C, assembly, robots=None, per_depth=3):
  """Debug helper: walk from the initial state and open the viewer on the initial
  state, a few varied 1-step successors, and one representative 2-step (pick->assemble).
  Each `view()` is blocking -- close the window to advance to the next state."""
  if robots is None:
    robots = get_robots(C)

  s0 = get_initial_state(C, assembly)
  print(f"[0] initial: {s0}")
  visualize_state(C, s0, assembly)

  succ1 = list(successors(s0, assembly, robots))
  print(f"    {len(succ1)} one-step successors")

  # pick a few varied ones (distinct base part) so they don't all look alike
  shown_bases, shown = set(), 0
  for action, ns in succ1:
    base = action[2]
    if base in shown_bases:
      continue
    shown_bases.add(base)
    shown += 1
    print(f"[1] {action} -> {ns}")
    visualize_state(C, ns, assembly)
    # follow this branch to the first assemble it enables
    for a2, n2 in successors(ns, assembly, robots):
      if a2[0] == "assemble":
        print(f"[2] {action} then {a2} -> {n2}")
        visualize_state(C, n2, assembly)
        break
    if shown >= per_depth:
      break



# --- wrench arrows (--show-wrenches) ------------------------------------------------------
_WF_SCALE, _WT_SCALE, _WMAX = 0.05, 1.0, 0.2               # m per N / m per Nm / length cap
_WF_RGB, _WT_RGB, _WP_RGB = (230, 60, 40), (60, 110, 240), (255, 160, 20)


def _wrench_vectors(state, assembly, gk, applied=(), poses=None):
  # (start, end, rgb) arrow segments for the recorder: contact FORCES red, COUPLES blue,
  # applied PRESS orange; lengths proportional (capped). One entry per contact from
  # state_contact_wrenches; infeasible groups get a short magenta marker.
  vecs = []
  for pt, f, t, label in state_contact_wrenches(state, assembly, gk,
                                                applied=applied, poses=poses):
    if label == "!infeasible":
      vecs.append((pt, pt + np.array([0.0, 0.0, 0.1]), (255, 0, 255)))
      continue
    fn = float(np.linalg.norm(f))
    if fn > 1e-6:
      L = min(_WMAX, _WF_SCALE * fn)
      vecs.append((pt, pt + f / fn * L, _WP_RGB if label == "press" else _WF_RGB))
    tn = float(np.linalg.norm(t))
    if tn > 1e-6:
      L = min(_WMAX, _WT_SCALE * tn)
      vecs.append((pt, pt + t / tn * L, _WT_RGB))
  return vecs


_IF_RGB, _IF_R = (40, 200, 90), 0.024                     # interface cross: green, 24mm


def _interface_vectors(Cv, assembly, state):
  # small green CROSSES at the mate interface of every currently-MATED part (world point =
  # part pose + R @ mate_interface), for debugging where the model applies seam loads.
  vecs = []
  for p in assembly.objects:
    if not any(c[0] == "mate" for c in state.supports.get(p, ())):
      continue
    d = assembly.mate_axis(p)
    if d is None:
      continue
    R = _quat_to_R(np.asarray(Cv.getFrame(p).getQuaternion(), dtype=float))
    base = np.asarray(Cv.getFrame(p).getPosition(), dtype=float)
    for off in assembly.mate_interface_points(p, d):       # every contact point of the mate
      pt = base + R @ off
      for ax in np.eye(3):
        vecs.append((pt - ax * _IF_R, pt + ax * _IF_R, _IF_RGB))
  return vecs


def _press_applied(Cv, assembly, action):
  # the insertion press of an `assemble` action as an applied load for the arrows, computed
  # from the RENDERED config (so it sits on the drawn geometry): insert_force along the
  # mate axis, at the MATE INTERFACE (the part's face along the press axis -- matching
  # _insertion_load's application point), acting on the target group (hint = onto).
  import wrench as _w
  if action[0] != "assemble":
    return ()
  part, onto = action[2], action[3]
  d = assembly.mate_axis(part)
  if d is None:
    return ()
  d = np.asarray(d, dtype=float)
  R = _quat_to_R(np.asarray(Cv.getFrame(onto).getQuaternion(), dtype=float))
  return ((_w.insert_force(part) * (R @ d),
           np.asarray(Cv.getFrame(part).getPosition(), dtype=float)
           + R @ assembly.mate_interface(part, d),
           onto),)


def _cv_poses(Cv, assembly):
  return {p: np.concatenate([Cv.getFrame(p).getPosition(), Cv.getFrame(p).getQuaternion()])
          for p in assembly.objects}


def visualize_plan(C, assembly, plan, robots=None, gripper_path=None, recorder=None,
                   bracing=False, wrench_gk=None):
  """Replay a plan (list of actions from solve_assembly_problem) step by step, opening
  the viewer on the initial state and on the state after each action. Each view() is
  blocking -- close the window to advance to the next step. With `recorder` (a
  viser_view.ViserRecorder) the steps are RECORDED instead of viewed; call
  recorder.serve() afterwards for the scrubbable web view."""
  if robots is None:
    robots = get_robots(C)

  def show(state, label, action=None):
    C_viz = visualize_state(C, state, assembly, gripper_path, view=recorder is None)
    if recorder is not None:
      vecs = None
      if wrench_gk is not None:
        ap = _press_applied(C_viz, assembly, action) if action is not None else ()
        vecs = _wrench_vectors(state, assembly, wrench_gk, applied=ap,
                               poses=_cv_poses(C_viz, assembly))
      recorder.record(C_viz, label, vectors=vecs,
                      interfaces=_interface_vectors(C_viz, assembly, state))

  # PREFER the state trajectory the search RECORDED (plan.states): walking it can't disagree
  # with the search about which actions exist. Only when it is absent (e.g. a plan loaded from
  # disk) do we RE-DERIVE via _declutter_edges -- which must then match the search's successor
  # flags exactly (grasp table, bracing), the historical source of "could not apply" bugs.
  recorded = getattr(plan, "states", None)
  s = recorded[0] if recorded else get_initial_state(C, assembly)
  print(f"[0/{len(plan)}] initial: {s}")
  show(s, "0: initial")

  allobj = set(assembly.objects)                           # for resolving stow edges (declutter)
  grasp_table = None if recorded else _search_context(C, assembly, robots)[1]
  for i, action in enumerate(plan, 1):
    if recorded is not None:
      nxt = recorded[i]
    else:                                                  # re-derive (must match search flags)
      nxt = next((ns for a, ns in _declutter_edges(s, assembly, robots, grasp_table, allobj,
                                                   bracing=bracing)
                  if a == action), None)
      if nxt is None:
        print(f"[{i}/{len(plan)}] !! could not apply {action} in {s}")
        return
    s = nxt
    print(f"[{i}/{len(plan)}] {action} -> {s}")
    show(s, f"{i}: {action[0]} {' '.join(map(str, action[1:]))}", action=action)


def _active_robots(action, s, rik):
  # the robots whose WITNESS DOFs get applied to the live replay config: the actor, plus
  # the holders (kinematic + braces) of the group the action MOVES. Braces of UNRELATED
  # groups must NOT be witness-posed: their DOFs are solved relative to the stateless IK
  # scene (which parks held groups at the holder's HOME pose), so transplanting them into
  # the replay -- where that group sits elsewhere -- tears the grasp. Unrelated braces
  # simply keep their pose (their group does not move during this action).
  kind = action[0]
  if kind in ("pick", "inhand_reorient"):
    return [action[1]]                                    # picks/reorients move no held group
  if kind in ("place", "stow"):                            # stow = a place aside (declutter)
    r, root = action[1], action[2]
    group = next((g for g in s.subassemblies() if root in g), frozenset())
    return list(dict.fromkeys([r] + rik._holders_of_group(s, group)))
  if kind == "assemble":
    _, r, _part, onto = action
    target = next((g for g in s.subassemblies() if onto in g), frozenset())
    return list(dict.fromkeys([r] + rik._holders_of_group(s, target)))
  if kind in ("add_grasp", "handover"):                   # new arm reaches; holders present it
    r, gpart = action[1], action[2]
    group = next((g for g in s.subassemblies() if gpart in g), frozenset())
    return list(dict.fromkeys([r] + rik._holders_of_group(s, group)))
  return []                                               # release: nobody moves


def _carrier_frame_name(carrier, rik):
  # the frame a part should be attached to for a given scene-graph carrier.
  if carrier[0] == "gripper":
    return rik.gripper_frame[carrier[1]]
  if carrier[0] == "part":
    return carrier[1]
  return "table"


def _write_robot_dofs(C, robot, q_full):
  # overwrite ONLY `robot`'s joint DOFs in C from the full witness vector q_full (leaving
  # every other robot where it already is -- that is what gives the replay continuity).
  names = C.getJointNames()
  q = C.getJointState()
  for i, nm in enumerate(names):
    if nm.startswith(robot + "_"):
      q[i] = q_full[i]
  C.setJointState(q)


def visualize_plan_ik(C, assembly, plan, robots=None, robot_ik=None, view=True,
                      recorder=None, bracing=False, witnesses=None, wrench_gk=None):
  """Replay a plan on a SINGLE persistent config: for each action we solve the skill IK,
  write the active arms' DOFs into the live config (so the arms keep their poses between
  steps), and mirror the scene-graph re-parenting with C.attach (so grabbed parts stay
  attached and follow the arm). Effectively a set_to_state(); view() loop.

  Because non-active arms are left where they are (not reset to home), this reads as a
  continuous choreography. It is still not a collision-checked trajectory -- only the
  per-action key configs are IK-solved. Each view() is blocking -- close to advance.
  With `recorder` (a viser_view.ViserRecorder) the steps are RECORDED instead of viewed;
  call recorder.serve() afterwards for the scrubbable web view."""
  if robots is None:
    robots = get_robots(C)
  if robot_ik is None:
    robot_ik = RobotIK(C, robots)

  # the live config: real arms at home + parts at their initial table poses (attached to
  # the table). We mutate THIS as the plan unfolds instead of rebuilding per step.
  Cv = ry.Config()
  Cv.addConfigurationCopy(robot_ik.C)

  # prefer the search's RECORDED state trajectory (plan.states); re-derive only as a fallback
  # (loaded-from-disk plans), where the successor flags must match the search -- see visualize_plan.
  recorded = getattr(plan, "states", None)
  s = recorded[0] if recorded else get_initial_state(C, assembly)
  print(f"[0/{len(plan)}] initial: {s}")
  if recorder is not None:
    vecs0 = _wrench_vectors(s, assembly, wrench_gk, poses=_cv_poses(Cv, assembly)) \
        if wrench_gk is not None else None
    recorder.record(Cv, "0: initial", vectors=vecs0,
                    interfaces=_interface_vectors(Cv, assembly, s))
  elif view:
    Cv.view(True, "0: initial")

  # CONSISTENCY MODEL of the replay: every frame shows either
  #   (a) held groups ALIGNED TO THEIR HOLDER'S GRIPPER (what the IK actually solved), or
  #   (b) unheld on-table groups seated at their canonical rest pose.
  # A group settles (b) at the start of the frame AFTER its last grasp released; while any
  # grasp/brace holds it, it stays with the arm (a). No snap ever separates a gripper from
  # its grasp -- that is what previously made braced assemblies clip through their EEs.

  def _held_and_table(state, group):
    held = any(c[0] == "grasp" for p in group for c in state.supports.get(p, ()))
    table = any(("table",) in state.supports.get(p, ()) for p in group)
    return held, table

  def _seat_rest(state, group):
    # canonical rest for the group's PLACEMENT: ref at its initial x/y, rotated so the
    # placement face points down, lifted so the LOWEST part touches the table. Written
    # via the group's kinematic root.
    root = _group_root(state, group)
    ref = min(group, key=_part_index)
    anchor = state.poses.get(ref)                               # ref's recorded table pose
    anchor = np.asarray(anchor) if anchor is not None \
        else np.asarray(robot_ik.C.getFrame(ref).getPose())    # (its initial pose by default)
    ref_pose = _group_rest_pose(assembly, group, ref,
                                anchor, _placement_face(state, group))
    rel = assembly.position_in_base(root) - assembly.position_in_base(ref)
    root_pose = _pose_compose(ref_pose, np.concatenate([rel, [1.0, 0.0, 0.0, 0.0]]))
    f = Cv.getFrame(root)
    f.setPosition(root_pose[:3])
    f.setQuaternion(root_pose[3:])

  def _align_group_to_holder(state, group, robot, gpart, grasp_id):
    # rigidly move the whole group so `gpart` sits at `robot`'s gripper exactly at its
    # grasp transform -- the configuration the IK solved in (scene_for attaches held
    # groups to their holder, even table-braced ones; the replay must match it).
    grasp = grasp_pose(assembly, gpart, grasp_id)
    ee_t = np.concatenate([grasp[:3],
                           _quat_mul(grasp[3:], robot_ik.q_correction[robot])])
    grip = np.asarray(Cv.getFrame(robot_ik.gripper_frame[robot]).getPose())
    p_target = _pose_compose(grip, _pose_invert(ee_t))
    p_cur = np.asarray(Cv.getFrame(gpart).getPose())
    delta = _pose_compose(p_target, _pose_invert(p_cur))
    root = _group_root(state, group)
    cur = np.asarray(Cv.getFrame(root).getPose())
    new = _pose_compose(delta, cur)
    f = Cv.getFrame(root)
    f.setPosition(new[:3])
    f.setQuaternion(new[3:])

  allobj = set(assembly.objects)                           # for resolving stow edges (declutter)
  grasp_table = None if recorded else _search_context(C, assembly, robots)[1]
  for i, action in enumerate(plan, 1):
    if recorded is not None:
      nxt = recorded[i]
    else:                                                  # re-derive (must match search flags)
      nxt = next((ns for a, ns in _declutter_edges(s, assembly, robots, grasp_table, allobj,
                                                   bracing=bracing)
                  if a == action), None)
      if nxt is None:
        print(f"[{i}/{len(plan)}] !! could not apply {action} in {s}")
        return
    # a stow is a place aside: run the place IK (which now targets the recorded stow pose).
    bind_action = ("place",) + tuple(action[1:]) if action[0] == "stow" else action
    if witnesses is not None and i - 1 < len(witnesses) and witnesses[i - 1] is not None:
      witness = witnesses[i - 1]                          # precomputed (e.g. optimize_witnesses)
      _ok = True                                          # by construction: kept witnesses are feasible
    else:
      _ok, witness = robot_ik.bind(bind_action, s, nxt, assembly)
    # 1) active arms -> the solved config; every other arm -> home, EXCEPT arms that must
    #    stay with their grasp. A KINEMATIC holder of a SOLO-held group may ride home (the
    #    group travels with it, out of the way). But if the group has OTHER holders
    #    (braces), homing the kinematic holder would carry the group away from the brace
    #    arms (which do not move) and tear those grasps -- so all holders of a multiply-
    #    held group keep their pose, as do braces on table-parented groups.
    active = set(_active_robots(action, s, robot_ik)) if witness is not None else set()
    # homing is decided on the POST-state: a robot that just RELEASED is free in nxt and
    # tucks home in this very frame -- visually announcing the release -- while everyone
    # still holding keeps its pose.
    holders_map = robot_ik.holders(nxt)
    grp_of = {p: g for g in nxt.subassemblies() for p in g}

    def _may_home(r):
      if r not in holders_map:
        return True                                       # free arm (incl. just-released)
      pt, _gid = holders_map[r]
      g = grp_of[pt]
      root = _group_root(nxt, g)
      kinematic = (pt == root and nxt.parent.get(root) == ("gripper", r))
      solo = len(robot_ik._holders_of_group(nxt, g)) == 1
      return kinematic and solo                           # ride home only if unbraced

    # mirror the kinematic re-parenting on the SAME config. attach preserves each part's
    # current world pose. Re-rooting a held subassembly REVERSES part->part edges, so a part
    # must be re-attached only AFTER its NEW parent has been re-attached -- otherwise we'd link
    # obj_0->obj_1 while obj_1 is still a child of obj_0, making a kinematic loop (rai warns,
    # then segfaults). Process new-parent-first (topological over changes).
    changed = {p: c for p, c in nxt.parent.items() if s.parent.get(p) != c}
    relinked = set()

    def _relink(p):
      if p in relinked:
        return
      relinked.add(p)
      carrier = changed[p]
      if carrier[0] == "part" and carrier[1] in changed:
        _relink(carrier[1])                    # settle (re-parent) the new parent first
      Cv.attach(_carrier_frame_name(carrier, robot_ik), p)

    def _apply_relink():
      for p in changed:
        _relink(p)

    # ORDER: on a HANDOVER frame (a release that re-roots the group onto a remaining holder;
    # witness is None) re-attach BEFORE homing -- so the new carrier is bound to the group at
    # its actual holding pose and then rides home CARRYING it. If we homed first, the arms
    # (and the group, still attached to the RELEASING arm) would move, and the re-attach would
    # capture that stale pose -> the assembly floats off the new carrier's gripper. For the
    # witnessed skills (pick/place/assemble) keep home/witness-then-relink (+ the snap below),
    # which is the order those were tuned for.
    if witness is None:
      _apply_relink()
    for r in robots:
      if r in active:
        _write_robot_dofs(Cv, r, witness)
      elif _may_home(r):
        _write_robot_dofs(Cv, r, robot_ik.home)
    if witness is not None:
      _apply_relink()
    # 3) per-action consistency snap:
    if action[0] == "pick":
      # pin the freshly grasped group exactly onto the gripper (covers IK residual and a
      # failed re-solve -- otherwise a bad relative offset gets baked into the attach and
      # trails the arm forever).
      group = next(g for g in nxt.subassemblies() if action[2] in g)
      _align_group_to_holder(nxt, group, action[1], action[2], action[3])
    elif action[0] == "inhand_reorient":
      # the in-hand skill's net effect: the held part rotates in the gripper to its NEW
      # grasp transform (no arm motion is solved for the reorient itself).
      group = next(g for g in nxt.subassemblies() if action[2] in g)
      _align_group_to_holder(nxt, group, action[1], action[2], action[3])
    elif action[0] in ("place", "stow"):
      # the group left the gripper: seat it exactly at its table pose (the stow cell for a
      # stow -- _seat_rest reads the recorded pose).
      _seat_rest(nxt, next(g for g in nxt.subassemblies() if action[2] in g))
    elif action[0] == "assemble":
      # exact mate: the added part sits exactly at its offset under `onto` ...
      f = Cv.getFrame(action[2])                                # mated part (child of onto)
      f.setRelativePosition(assembly.relative_pose(action[2])[1])
      f.setRelativeQuaternion([1.0, 0.0, 0.0, 0.0])
      # ... and, ONLY when the merged group is kinematically HELD (root gripper-parented),
      # the whole group aligns to its KINEMATIC HOLDER's gripper -- that is the group's
      # actual bond; the actor's arm comes to the group, never the group to the actor
      # (aligning to the actor once yanked a held group to a stale actor pose when the
      # replay re-solve failed). A TABLE-parented (possibly braced) group stays exactly
      # where it rests. Skip when witness is None: arm poses are stale, only the exact
      # part-relative mate snap above is trustworthy then.
      group = next(g for g in nxt.subassemblies() if action[2] in g)
      root = _group_root(nxt, group)
      carrier = nxt.parent.get(root, (None,))
      hold_gid = nxt.grasp.get(root)
      if witness is not None and carrier[0] == "gripper" and hold_gid is not None:
        _align_group_to_holder(nxt, group, carrier[1], root, hold_gid)
    # 4) settle: any group that (now) rests on the table with NO grasp left sits at its
    #    canonical rest pose -- in particular a group whose last brace just released
    #    (physically: it drops onto the table this instant, not next frame).
    for g in nxt.subassemblies():
      held, table = _held_and_table(nxt, g)
      if table and not held:
        _seat_rest(nxt, g)
    print(f"[{i}/{len(plan)}] {action} (feasible={_ok}) -> {nxt}")
    if recorder is not None:
      vecs = None
      if wrench_gk is not None:
        vecs = _wrench_vectors(nxt, assembly, wrench_gk,
                               applied=_press_applied(Cv, assembly, action),
                               poses=_cv_poses(Cv, assembly))
      recorder.record(Cv, f"{i}: {action[0]} {' '.join(map(str, action[1:]))}", vectors=vecs,
                      interfaces=_interface_vectors(Cv, assembly, nxt))
    elif view:
      Cv.view(True, f"{i}: {action[0]} {' '.join(map(str, action[1:]))}")
    s = nxt
  return Cv



def _ik_test_scene(C, assembly, n_parts, num_robots, vacuum):
  # the IK test popups run on the ACTUAL configured scene when main passes (C, assembly) --
  # robot type, scatter/--random-init layout, part shapes all as configured. The legacy
  # self-built cube-cross env remains the fallback for direct calls.
  if C is None:
    C = make_four_robot_env(num_robots=num_robots, floating_ee=False, vacuum=vacuum)
    assembly = CubeCrossAssembly(n_parts=n_parts)
    add_disassembled_obj(C, assembly)
  return C, assembly


def visualize_pick_ik(part="obj_3", grasp_id=0, robot=None, n_parts=7, num_robots=2,
                      vacuum=False, C=None, assembly=None):
  """Set up the real-robot env and pop up the pick IK for one (robot, part, grasp) so you
  can eyeball whether the grasp is sensible. Needs a display. Pass (C, assembly) to run on
  an existing configured scene (e.g. --random-init); the self-built cube-cross env with a
  real arm is the fallback."""
  C, assembly = _ik_test_scene(C, assembly, n_parts, num_robots, vacuum)
  robots = get_robots(C)
  if robot is None:
    robot = robots[0]
  rik = RobotIK(C, robots, debug_view=True)
  s0 = get_initial_state(C, assembly)
  ok, _witness = rik.bind(("pick", robot, part, grasp_id), s0, None, assembly)
  print(f"pick IK ({robot} -> {part}, grasp {grasp_id}): feasible={ok}")
  return ok


def visualize_assemble_ik(part="obj_1", onto=None, robot=None, grasp_id=0,
                          target_robot=None, onto_grasp=1, n_parts=7,
                          num_robots=2, vacuum=False, C=None, assembly=None):
  """Pop up the assemble IK: `robot` holds `part` at `grasp_id` and mates it onto `onto`
  (its manual parent, by default). Needs a display. Pass (C, assembly) to run on an
  existing configured scene.

  By default `onto` sits on the table at its scanned pose (a fixed target). If
  `target_robot` is given, that robot instead HOLDS `onto` (at `onto_grasp`) -- so both
  cubes are in the air and the assemble IK co-optimizes both arms (adder + target holder)
  to bring the mate together."""
  C, assembly = _ik_test_scene(C, assembly, n_parts, num_robots, vacuum)
  robots = get_robots(C)
  if robot is None:
    robot = robots[0]
  if onto is None:
    onto = assembly.relative_pose(part)[0]                # the part's manual mate parent
  s0 = get_initial_state(C, assembly)
  # build the pre-assemble state: if a target_robot is given it first picks `onto` (so the
  # target is held in the air), then `robot` picks `part`. both end up held.
  state = s0
  if target_robot is not None:
    state = next(ns for act, ns in picks(state, assembly, robots)
                 if act == ("pick", target_robot, onto, onto_grasp))
  held = next(ns for act, ns in picks(state, assembly, robots)
              if act == ("pick", robot, part, grasp_id))
  rik = RobotIK(C, robots, debug_view=True)
  ok, _witness = rik.bind(("assemble", robot, part, onto), held, None, assembly)
  who = f"{robot}:{part} -> {target_robot or 'table'}:{onto}"
  print(f"assemble IK ({who}, grasp {grasp_id}/{onto_grasp}): feasible={ok}")
  return ok


def visualize_place_ik(part="obj_1", grasp_id=0, robot=None, n_assembled=1, n_parts=7,
                       num_robots=2, vacuum=False, face=_IDENTITY_FACE, C=None,
                       assembly=None):
  """Pop up the place IK: with `n_assembled` parts already mated onto the base (0 = none),
  `robot` picks the subassembly BY `part` at `grasp_id` (re-rooting; for n_assembled=0 the
  part is a loose singleton) and sets it down on the table resting on placement `face`
  (5 = upright/identity). Needs a display. Pass (C, assembly) to run on an existing
  configured scene."""
  C, assembly = _ik_test_scene(C, assembly, n_parts, num_robots, vacuum)
  robots = get_robots(C)
  if robot is None:
    robot = robots[0]

  # build the pre-place state DISCRETELY: assemble obj_1..obj_k (pick/assemble/release with
  # arbitrary grasps -- they leave no trace), then pick the group by `part` at `grasp_id`.
  state = get_initial_state(C, assembly)
  for i in range(1, n_assembled + 1):
    p = f"obj_{i}"
    act, state = next((a, ns) for a, ns in picks(state, assembly, robots)
                      if a[0] == "pick" and a[2] == p)
    act, state = next((a, ns) for a, ns in assembles(state, assembly, robots)
                      if a[2] == p)
    act, state = next(((a, ns) for a, ns in releases(state, assembly, robots)
                       if a[2] == p), (None, state))      # release may be support-gated away
  match = [(a, ns) for a, ns in picks(state, assembly, robots)
           if a[1] == robot and a[2] == part and a[3] == grasp_id]
  if not match:
    avail = sorted({(a[2], a[3]) for a, _ in picks(state, assembly, robots) if a[1] == robot})
    print(f"pick {part}@{grasp_id} not available here; available (part, grasp): {avail}")
    return False
  _, held = match[0]

  # the REAL place successor (its pose payload is the landing anchor _ik_place reads --
  # binding with s1=None crashes on s1.poses since the payload mechanism landed).
  pl = [(a, ns) for a, ns in places(held, assembly, robots)
        if a[1] == robot and a[3] == face and len(a) == 4]
  if not pl:
    print(f"no place successor for {robot} at face {face} from this state")
    return False
  act, s1 = pl[0]

  rik = RobotIK(C, robots, debug_view=True)
  ok, _witness = rik.bind(act, held, s1, assembly)
  print(f"place IK ({robot} sets down {part}, {n_assembled} parts assembled, "
        f"grasp {grasp_id}, face {face}): feasible={ok}")
  return ok
