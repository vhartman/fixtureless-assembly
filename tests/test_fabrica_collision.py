"""Geometric boundary tests, including cases a surface/surface test misses."""
import numpy as np
import trimesh
from part_meshes.fabrica.tetra_collision import BoundaryCheck, solid_contains


def test_solid_contains_is_deterministic_at_ray_degenerate_points():
    """Membership must not depend on trimesh's randomized ray direction."""
    shell = trimesh.creation.box([2, 2, 2])
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
    ])
    expected = np.array([True, True, False, False])
    for _ in range(5):
        assert np.array_equal(solid_contains(shell, points, chunk_size=1), expected)
        assert np.array_equal(solid_contains(shell, points, chunk_size=4), expected)


def test_solid_contains_accepts_nested_consistently_oriented_shells():
    outer = trimesh.creation.box([4, 4, 4])
    inner = trimesh.creation.box([2, 2, 2])
    nested = trimesh.util.concatenate([outer, inner])
    assert solid_contains(nested, [[0, 0, 0], [1.5, 0, 0], [3, 0, 0]]).tolist() == [
        True, True, False]


def test_boundary_clipping_inside_outside_crossing_and_enclosure():
    shell = trimesh.creation.box([2, 2, 2])
    check = BoundaryCheck(shell)
    assert not check.intersects(trimesh.creation.box([1, 1, 1]))
    far = trimesh.creation.box([1, 1, 1])
    far.apply_translation([3, 0, 0])
    assert not check.intersects(far)
    crossing = trimesh.creation.box([1, 1, 1])
    crossing.apply_translation([.75, 0, 0])
    assert check.intersects(crossing)
    assert check.intersects(shell)
    # All source triangles inside the candidate: surface/surface crossings are absent.
    assert check.intersects(trimesh.creation.box([3, 3, 3]))


def test_boundary_clipping_triangle_with_all_vertices_outside():
    triangle = trimesh.Trimesh(vertices=[[-2, -2, 0], [2, -2, 0], [0, 2, 0]],
                               faces=[[0, 1, 2]], process=False)
    assert BoundaryCheck(triangle).intersects(trimesh.creation.box([1, 1, 1]))


def test_boundary_clipping_small_positive_gap_and_thin_cells():
    check = BoundaryCheck(trimesh.creation.box([.1, .1, .1]))
    cell = trimesh.creation.box([.01, .01, 1e-8])
    cell.apply_translation([0, 0, .05 - 1e-7])
    assert not check.intersects(cell)
    cell.apply_translation([0, 0, 1e-7])
    assert check.intersects(cell)


def test_boundary_clipping_matches_analytic_box_intersection():
    rng = np.random.default_rng(7)
    check = BoundaryCheck(trimesh.creation.box([2, 2, 2]))
    for _ in range(100):
        center, half = rng.uniform(-2, 2, 3), rng.uniform(.05, 1.5, 3)
        box = trimesh.creation.box(2 * half)
        box.apply_translation(center)
        lo, hi = center - half, center + half
        volume_overlap = np.all(hi >= -1) and np.all(lo <= 1)
        strictly_inside = np.all(lo > -1) and np.all(hi < 1)
        assert check.intersects(box) == (volume_overlap and not strictly_inside)
