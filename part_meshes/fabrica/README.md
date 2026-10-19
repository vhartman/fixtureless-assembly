# Fabrica mesh imports

These are metre-scale, part-local visual meshes imported from Yunsheng Tian et al.'s
[Fabrica](https://github.com/yunshengtian/Fabrica) benchmark (MIT license). The source OBJ
files use centimetres and place every part directly in the common final assembly frame.

Run `import_assets.py` to reproduce the checked-in geometry from the neighbouring Fabrica
checkout. Each model's `manifest.json` records the source hash, original AABB centre, scale
conversion, and collision-decomposition settings. The exact OBJ is the visual and grasp-
sampling surface. Collision is stored as one GLB per part; its named convex components and
part-local offsets are consumed identically by RAI and MuJoCo.

The `tetra-inner` importer constructs a simplified boundary proposal, tetrahedralizes it
with TetGen, and merges cells whose union is convex. It then checks each cell against the
**original** CAD: the centre must be inside and the cell must not intersect any
CAD boundary triangle (including triangles fully enclosed by the cell). Intersecting cells are contracted about their interior centre until they
fit, with the final check including float32 conversion. A second merge pass accepts larger
convex cells only when they pass the original-CAD boundary check. This is a floating-point
containment check, not an exact-arithmetic certificate. The importer rejects results whose
summed cell volume is below 80% of CAD volume and records that ratio in the manifest.
This is a fidelity screen, not a union-volume certificate: merged or VHACD cells can overlap.
This preserves empty holes, at the cost of small gaps between cells and substantially more
collision shapes. Exact visuals and grasp-sampling surfaces remain unchanged.

The offline boundary gate uses AABB-indexed triangle/convex half-space clipping, with
touching counted as intersection. This has bounded work, unlike libccd's MPR portal
discovery, which stalled on a nearly degenerate Circular Stool candidate. RAI and
MuJoCo insertion audits remain independent checks using their own collision engines.

`--method vhacd-inner` uses VHACD proposals with the same containment checks and is useful
when tetrahedralization of a simplified boundary is problematic. It fails if too much
material is lost; Duct failed this route and uses tetrahedra. No method is substituted silently.

`--method vhacd` is retained for comparison. Its approximate hulls are **not suitable by
default for tight mates**: even 128–256 Duct hulls bridge the bores. The older centre-point
hole test does not establish insertion feasibility. Files whose manifest still says
`vhacd` have not been converted to the contained representation.

The geometric conversion covers `car`, `beam`, `cooling_manifold`, `duct`, `gamepad`,
`plumbers_block`, and `stool_circular`. Mate parents, assembly directions, and collision
proxies remain small per-model annotations because they are not encoded in Fabrica's asset
files; Fabrica itself computes precedence using a separate disassembly planner. For the five
models beyond Car/Beam, the current first-pass dependency tree is inferred from final-pose
proximity and kept explicit in `assembly_planner/problem.py` so it can be reviewed rather than changing with
sampling noise.

Every authored stem, peg, pin and screw remains collision geometry; it is no longer trimmed
to an exposed cap in the final pose. Grasp sampling bypasses the convex-component union and
uses the exact OBJ, so artificial decomposition seams cannot become grasp surfaces. In
MuJoCo, all `obj_N__collisionK` geoms and the non-colliding `obj_N__visual` geom belong to the
single `obj_N` body. Viewer group 2 shows collision components and group 3 shows exact meshes.

Reproduce from the assembly repository with:

```bash
python3 part_meshes/fabrica/import_assets.py \
  --source ../../Fabrica/assets/fabrica --models duct car --method tetra-inner

# Check complete moving-part geometry against the assembled prefix in both engines.
# Default: 50 mm insertion, samples every 0.5 mm, penetration tolerance 1 micrometre.
python3 tools/diagnostics/audit_fabrica.py duct --containment
python3 tools/diagnostics/audit_fabrica.py car --containment

python3 tools/visualization/view_fabrica.py duct --pose assembled
python3 tools/visualization/view_fabrica.py car --pose assembled
```

The Duct cover and all six fasteners enter along the common flange normal
`(-0.5, 0, -sqrt(0.75))`. The former radial XY insertion guesses were invalid even for
the authored meshes. An audit is sampled, not continuous; a finer `--step` tightens it.
Use `--output /tmp/fabrica-candidate` when experimenting, then
`tools/diagnostics/audit_fabrica.py duct --asset-root /tmp/fabrica-candidate` before replacing
assets.
Sub-micrometre duplicate seam vertices are welded at 0.1 micrometre (needed by Circular
Stool). Sources that remain non-watertight require repair; the importer fails explicitly.

Two other authored insertion directions are nonvertical: Gamepad source `4` (rear
connector) enters along `-Y`; Plumbers Block source `0` (shaft) enters along `+Y`.
These directions were checked against the exact CAD and then against both collision
engines. Duct's loose cover and fasteners start rotated by -30 degrees about Y so their
head/flange faces rest horizontally, rather than starting at the tilted assembled pose.

For the more detailed remaining assets, explicit per-part proposal methods can avoid
unnecessarily large tetrahedralizations without changing the containment gate:

```bash
python3 part_meshes/fabrica/import_assets.py --models plumbers_block \
  --method tetra-inner --tetra-faces 2000
python3 part_meshes/fabrica/import_assets.py --models gamepad \
  --method vhacd-inner --part-method 3=tetra-inner
python3 part_meshes/fabrica/import_assets.py --models stool_circular \
  --method vhacd-inner --tetra-faces 2000 \
  --part-method 0=tetra-inner --part-method 2=tetra-inner \
  --part-method 4=tetra-inner --part-method 6=tetra-inner --part-method 8=tetra-inner
```

The importer checks against the serialized OBJ, including decimal quantization, and
checks the float32 collision representation before writing it. `--refit-existing`
rechecks/refits older contained assets against the exported CAD without recomputing or
merging their decomposition. Always run the independent audit after import/refitting.

The Car source's `2.obj` is a fixture, not an assembly part. The importer retains it for a
hash-complete source conversion, but `fabrica-car` excludes it from the initial scene and
the final assembly. None of the other currently available Fabrica models has such a fixture.

Beam uses reviewed axis-aligned box unions at runtime for its three mortised wooden members:
seven boxes for the four-hole base rail and ten boxes for each tapered one-hole crossbar. The
two inserted uprights retain their three-child semantic proxies. This reduces the complete
five-part collider from 395 children to 33 without changing the exact visual/grasp meshes.
The boxes stop on either side of every authored through-hole and are strictly contained in
the rounded CAD skin.
`tools/diagnostics/audit_fabrica.py beam --containment --robustness 0.001` is the required
regression: it checks every full insertion sweep and the seated pose under +/-1 mm translations
in both RAI and MuJoCo. Re-running the generic importer does not overwrite this reviewed runtime
proxy, which is defined in `FabricaAssembly._beam_primitive_collision_shape`.

Cooling Manifold deliberately approximates the fin stack with solid rectangular collision
envelopes. The connector-bore columns are excluded from these boxes; their detailed
collision cells remain. The visual mesh still shows individual fins. This explicit
exception is recorded as `approximation_boxes` in the manifest: containment checks allow
those boxes, but insertion collision checks do not exempt them.

The four Cooling Manifold corner fasteners use semantic collision proxies: a cylindrical
stem plus a hexagonal head.  The matching base geometry uses annular wedge components, not a
solid cylinder, so the socket encloses the stem without filling its insertion channel.  The
head remains visible and collidable; only the stem radius is reduced to provide the authored
fit clearance.

Circular Stool keeps the exact render meshes and mate transforms.  Its four fitted leg
proxies are retreated 2.5 mm axially/radially inside the exact CAD.  This is recorded as
`axial_fit_trim` in the manifest and is deliberately local to collision geometry: it gives
the insertion channel a finite numerical clearance without moving or shrinking the visible
part. Re-run `tools/diagnostics/audit_fabrica.py circular-stool --containment` after changing it.

For end-to-end CAD planning, use hard witness residual contracts in addition to collision
checks.  `benchmarks/run_fabrica_smoke.py` applies 40 micrometre grasp/mate translation
limits and a 0.05 degree mate-angle limit.  This prevents a locally clean but slightly
off-pose attachment from accumulating into a later part/part overlap.  The attachment-chain
gate then checks the exact accumulated configuration before BiRRT starts; increasing the
motion-planning tolerance is not a substitute for fixing a bad witness.
