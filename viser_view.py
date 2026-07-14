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
  def __init__(self):
    self.geometry = {}                    # name -> (verts, tris, color_rgb, opacity)
    self.markers = {}                     # name -> axes_length (rai ST.marker frames)
    self.steps = []                       # [{name: (pos3, wxyz4)}]
    self.labels = []                      # one per step

  def record(self, C, label=""):
    """Snapshot every mesh-bearing frame of `C` as one step. Geometry is captured the
    first time a frame name is seen (assumed rigid thereafter). rai ST.marker frames
    have no mesh; they are tracked separately and served as viser axis frames."""
    snap = {}
    for frame in C.getFrames():
      name = frame.name
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

    step_slider = server.gui.add_slider("Step", min=0, max=len(self.steps) - 1,
                                        step=1, initial_value=0)
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
      for name, h in handles.items():
        pose = snap.get(name)
        if pose is None:
          h.visible = False              # frame absent in this step (e.g. schematic EE)
        else:
          h.visible = True
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
