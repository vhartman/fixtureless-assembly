"""Viser-based plan viewer: record per-step frame poses during a (sequential, stateful)
replay, then serve them in a web viewer with a step slider / playback controls.

The rai viewer shows one blocking window per step; viser instead gets the WHOLE replay as
a scrubbable timeline at http://localhost:<port>. Because our replays are sequential
(each frame builds on the previous config state), we do not re-run replay logic on slider
jumps -- we snapshot every frame's pose at record time, which makes random access trivial.
The frame SET may differ between steps (e.g. visualize_plan spawns floating EEs per
grasp), so each mesh handle is shown only in the steps whose snapshot contains it.

Usage:
    rec = ViserRecorder()
    rec.record(C_step0, "0: initial")     # once per replay step, on the step's config
    ...
    rec.serve(port=8080)                  # blocks until Stop / Ctrl-C

Adapted from mr_bench's RaiEnv.display_path_viser (rai_base_env.py).
"""

import time

import numpy as np
import robotic as ry


class ViserRecorder:
  def __init__(self, hide_collision_primitives=False):
    self.geometry = {}                    # name -> (verts, tris, color_rgb, opacity)
    self.markers = {}                     # name -> axes_length (rai ST.marker frames)
    self.steps = []                       # [{name: (pos3, wxyz4)}]
    self.labels = []                      # one per step
    self.step_vectors = []                # one list per step: (start3, end3, rgb) arrows
                                          # (forces red, torques blue, press orange)
    self.step_interfaces = []             # one list per step: mate-interface crosses (green)
    self.hide_collision_primitives = bool(hide_collision_primitives)
    self.hidden_collision_primitives = set()

  def _is_display_collision_primitive(self, frame):
    """Whether a frame is a redundant collision proxy rather than display geometry.

    We cannot hide every collidable frame: primitive-only assembly parts and the table are
    both collision geometry *and* their intended appearance. Robot/link proxies, gripper
    capsules, and imported-object hull overlays are instead conventionally translucent or
    named ``coll*`` and have separate opaque visual geometry. The combined test preserves
    primitive-only scene objects while removing the white proxy skeletons from paper/debug
    views.
    """
    if not self.hide_collision_primitives:
      return False
    info = frame.info()
    if bool(info.get("visual", False)):
      return False
    name = frame.name.lower()
    named_proxy = ("coll" in name or "collision" in name)
    color = info.get("color", ())
    translucent = (hasattr(color, "__len__") and len(color) >= 4
                   and float(color[3]) < 0.999)
    contact = float(info.get("contact", 0.0) or 0.0)
    return named_proxy or (contact < 0.0 and translucent)

  def record(self, C, label="", vectors=None, interfaces=None):
    """Snapshot every mesh-bearing frame of `C` as one step. Geometry is captured the
    first time a frame name is seen (assumed rigid thereafter). rai ST.marker frames
    have no mesh; they are tracked separately and served as viser axis frames."""
    snap = {}
    for frame in C.getFrames():
      name = frame.name
      if self._is_display_collision_primitive(frame):
        self.hidden_collision_primitives.add(name)
        continue
      if name not in self.geometry and name not in self.markers:
        verts = np.asarray(frame.getMeshPoints(), dtype=np.float32)
        tris = np.asarray(frame.getMeshTriangles(), dtype=np.uint32)
        if verts.ndim < 2 or tris.ndim < 2 or len(verts) == 0 or len(tris) == 0:
          info = frame.info()
          if info.get("shape") == "marker":
            size = info.get("size", [0.1])
            self.markers[name] = float(size[0] if hasattr(size, "__len__") else size)
          else:
            continue                      # no renderable geometry
        else:
          # per-vertex colors when present; frame-info color as fallback
          mesh_colors = np.asarray(frame.getMeshColors())
          if mesh_colors.ndim == 2 and len(mesh_colors) > 0:
            c = mesh_colors[0]            # [R,G,B,A] 0-255
            color_rgb = (int(c[0]), int(c[1]), int(c[2]))
            alpha = int(c[3]) if len(c) > 3 else 255
            opacity = alpha / 255.0 if alpha < 255 else None
          else:
            raw = frame.info().get("color", [0.7, 0.7, 0.7])
            color_rgb = tuple(int(x * 255) for x in raw[:3])
            opacity = float(raw[3]) if len(raw) > 3 and raw[3] < 1.0 else None
          self.geometry[name] = (verts, tris, color_rgb, opacity)
      if name in self.geometry or name in self.markers:
        snap[name] = (np.array(frame.getPosition(), dtype=np.float32),
                      np.array(frame.getQuaternion(), dtype=np.float32))  # ry = [w,x,y,z]
    self.steps.append(snap)
    self.labels.append(label)
    self.step_vectors.append(list(vectors) if vectors else [])
    self.step_interfaces.append(list(interfaces) if interfaces else [])

  def snapshot_poses(self, C, names=None):
    """Snapshot poses for geometry already known to the recorder.

    Used for diagnostic ghost layers after the opaque canonical step has been recorded.
    Requiring known geometry keeps an alternative from silently changing the scene model.
    """
    available = set(C.getFrameNames())
    requested = (list(self.geometry) + list(self.markers)
                 if names is None else list(names))
    return {
        name: (np.asarray(C.getFrame(name).getPosition(), dtype=np.float32),
               np.asarray(C.getFrame(name).getQuaternion(), dtype=np.float32))
        for name in requested
        if name in available}

  def snapshot_poses(self, C):
    """Snapshot poses for geometry already registered by :meth:`record`.

    Alternative optimizer keyframes reuse the selected step's meshes but may move both arms
    and attached payloads.  Capturing from a cloned, stateful replay configuration preserves
    those kinematics without adding the alternative as another timeline step.
    """
    available = set(C.getFrameNames())
    return {
        name: (np.asarray(C.getFrame(name).getPosition(), dtype=np.float32),
               np.asarray(C.getFrame(name).getQuaternion(), dtype=np.float32))
        for name in list(self.geometry) + list(self.markers) if name in available
    }

  def overlay_alternatives(self, alternatives, count=4, opacity=0.18,
                           description="witness alternatives"):
    """Add alternative poses for the *same* recorded step as translucent ghost layers.

    ``alternatives`` maps recorder step indices to lists of ``{frame_name: pose}`` snapshots;
    the ordinary snapshot remains the opaque selected solution. This is intentionally agnostic
    to how alternatives were produced. Saved-plan replay uses the persisted witness optimizer
    ladder, so visualization never reruns optimization or IK.
    """
    count = int(count)
    opacity = float(opacity)
    if count < 2:
      raise ValueError("alternative overlay count must be >= 2")
    if not 0.0 < opacity < 1.0:
      raise ValueError("alternative ghost opacity must lie strictly between 0 and 1")
    if getattr(self, "_alternative_overlay_applied", False):
      raise ValueError("alternative overlays have already been applied to this recording")

    def same_pose(a, b):
      pa, qa = a
      pb, qb = b
      return (np.allclose(pa, pb, atol=1e-7, rtol=0.0) and
              (np.allclose(qa, qb, atol=1e-7, rtol=0.0) or
               np.allclose(qa, -qb, atol=1e-7, rtol=0.0)))

    used = set()
    shown_steps = 0
    max_layers = 0
    for step_index, members in sorted(alternatives.items()):
      if not 0 <= int(step_index) < len(self.steps):
        raise ValueError(f"alternative overlay step {step_index} is out of range")
      current = self.steps[int(step_index)]
      shown = list(members)[:count - 1]
      if not shown:
        continue
      max_layers = max(max_layers, len(shown))
      for layer, member in enumerate(shown, 1):
        for name, pose in member.items():
          if name not in self.geometry:
            continue
          if name in current and same_pose(current[name], pose):
            continue
          ghost = f"keyframe_alt_{layer}/{name}"
          current[ghost] = pose
          used.add((layer, name))
      self.labels[int(step_index)] += (
          f"  |  selected candidate=0; translucent alternatives=1..{len(shown)}")
      shown_steps += 1

    for layer, name in sorted(used):
      verts, tris, color, original_opacity = self.geometry[name]
      alpha = 1.0 if original_opacity is None else original_opacity
      alpha_scale = max(0.35, 1.0 - 0.18 * (layer - 1))
      self.geometry[f"keyframe_alt_{layer}/{name}"] = (
          verts, tris, color, opacity * alpha * alpha_scale)
    self._alternative_overlay_applied = True
    print(f"[viser] {description}: {shown_steps} keyframe(s), 1 opaque selected witness + "
          f"up to {max_layers} translucent alternative(s) at opacity {opacity:g}; "
          f"{len(used)} distinct ghost layer/mesh pairs")
    return self

  def serve(self, port=8080, pause_time=0.5):
    """Open the viser server and block until Stop is pressed (or Ctrl-C)."""
    import viser

    if not self.steps:
      print("[viser] nothing recorded")
      return

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False

    handles = {}
    for name, (verts, tris, color_rgb, opacity) in self.geometry.items():
      handles[name] = server.scene.add_mesh_simple(
          name=f"frames/{name}", vertices=verts, faces=tris,
          color=color_rgb, flat_shading=False, opacity=opacity)
    for name, size in self.markers.items():
      handles[name] = server.scene.add_frame(
          name=f"frames/{name}", axes_length=0.5 * size, axes_radius=0.04 * size)

    # wrench arrows (--show-wrenches): one line-segment batch per step, toggled with the
    # step like the meshes. Older viser without add_line_segments -> skipped with a note.
    vec_handles = [None] * len(self.steps)
    if any(self.step_vectors):
      try:
        for i, vecs in enumerate(self.step_vectors):
          if not vecs:
            continue
          pts = np.array([[list(s), list(e)] for s, e, _c in vecs], dtype=np.float32)
          cols = np.array([[list(c), list(c)] for _s, _e, c in vecs], dtype=np.uint8)
          vec_handles[i] = server.scene.add_line_segments(
              name=f"wrench/step{i}", points=pts, colors=cols, line_width=4.0,
              visible=False)
      except Exception as e:                              # viser too old for line segments
        print(f"[viser] wrench arrows unavailable ({e}); update viser to see them")
        vec_handles = [None] * len(self.steps)
    if_handles = [None] * len(self.steps)
    if any(self.step_interfaces):
      try:
        for i, vecs in enumerate(self.step_interfaces):
          if not vecs:
            continue
          pts = np.array([[list(s), list(e)] for s, e, _c in vecs], dtype=np.float32)
          cols = np.array([[list(c), list(c)] for _s, _e, c in vecs], dtype=np.uint8)
          if_handles[i] = server.scene.add_line_segments(
              name=f"interfaces/step{i}", points=pts, colors=cols, line_width=3.0,
              visible=False)
      except Exception:
        if_handles = [None] * len(self.steps)

    step_slider = server.gui.add_slider("Step", min=0, max=len(self.steps) - 1,
                                        step=1, initial_value=0)
    wrench_checkbox = None
    if any(vh is not None for vh in vec_handles):
      wrench_checkbox = server.gui.add_checkbox("Show wrenches", initial_value=True)
    markers_checkbox = None
    if self.markers:
      markers_checkbox = server.gui.add_checkbox("Show markers", initial_value=True)
    if_checkbox = None
    if any(vh is not None for vh in if_handles):
      if_checkbox = server.gui.add_checkbox("Show mate interfaces", initial_value=True)
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)
    pause_field = server.gui.add_number("Pause time (s)", initial_value=pause_time,
                                        min=0.0, step=0.05)
    prev_btn = server.gui.add_button("< Prev")
    next_btn = server.gui.add_button("Next >")
    stop_btn = server.gui.add_button("Stop")
    label_md = server.gui.add_markdown("")

    def set_step(i):
      i = max(0, min(int(i), len(self.steps) - 1))
      snap = self.steps[i]
      show_w = wrench_checkbox is None or wrench_checkbox.value
      for j, vh in enumerate(vec_handles):
        if vh is not None:
          vh.visible = (j == i and show_w)
      show_if = if_checkbox is None or if_checkbox.value
      for j, vh in enumerate(if_handles):
        if vh is not None:
          vh.visible = (j == i and show_if)
      show_m = markers_checkbox is None or markers_checkbox.value
      for name, h in handles.items():
        pose = snap.get(name)
        if pose is None:
          h.visible = False              # frame absent in this step (e.g. schematic EE)
        else:
          h.visible = show_m if name in self.markers else True
          h.position, h.wxyz = pose
      label_md.content = f"**Step {i} / {len(self.steps) - 1}**  \n{self.labels[i]}"

    _stopped = False

    @stop_btn.on_click
    def _(_):
      nonlocal _stopped
      _stopped = True

    @prev_btn.on_click
    def _(_):
      play_checkbox.value = False
      step_slider.value = max(step_slider.value - 1, 0)
      set_step(step_slider.value)

    @next_btn.on_click
    def _(_):
      play_checkbox.value = False
      step_slider.value = min(step_slider.value + 1, len(self.steps) - 1)
      set_step(step_slider.value)

    @step_slider.on_update
    def _(event):
      if not play_checkbox.value:
        set_step(event.target.value)

    if wrench_checkbox is not None:
      @wrench_checkbox.on_update
      def _(_):
        set_step(step_slider.value)

    if markers_checkbox is not None:
      @markers_checkbox.on_update
      def _(_):
        set_step(step_slider.value)

    if if_checkbox is not None:
      @if_checkbox.on_update
      def _(_):
        set_step(step_slider.value)

    set_step(0)
    print(f"[viser] open http://localhost:{port} -- Stop in the GUI or Ctrl-C to exit.")
    try:
      while not _stopped:
        if play_checkbox.value:
          nxt = (step_slider.value + 1) % len(self.steps)
          step_slider.value = nxt
          set_step(nxt)
        time.sleep(max(pause_field.value, 0.02))
    except KeyboardInterrupt:
      pass
    finally:
      server.stop()
