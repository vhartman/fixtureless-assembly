"""One-off audit helper: rebuild a scene from a saved plan whose config block has every
POST-2026-07-29 flag stripped, to exercise the RunConfig-default fallbacks in replay.py
(saved_cfg). Not part of the pipeline; safe to delete."""
import json

import replay

_STRIP = (
    "grasp_model", "inserter_check", "insert_force", "curved_torque_scale",
    "insertion_noise", "insertion_offset", "robust_mode", "tight_fingers",
    "part_float", "pre_insertion", "pre_pick", "pre_place", "pre_add_grasp",
    "post_release", "witness_margin", "witness_sets",
    "bracing", "witness_robot_collisions", "strict_collisions",
    "grasp_sampler_margin", "grasp_sampler_pad_support", "mp_robot_margin",
    "mp_margin_ramp", "mp_fake_clock", "mate_eq_weight", "place_eq_weight",
    "handovers", "allow_place", "dense_grasps",
)

with open("audit-smoke-plan.json") as f:
    d = json.load(f)
cfg = d["config"]
for k in _STRIP:
    cfg.pop(k, None)

C, asm = replay.build_scene(cfg)
print("legacy scene rebuilt OK:", len(asm.objects), "parts,", cfg.get("assembly"))
