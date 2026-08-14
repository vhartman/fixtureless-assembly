"""Numerical check: is the exact composite gain monotone in the shortened robot set?

For the per-edge cost max_r d_r + 0.01 * sum_r d_r, with the candidate per-edge distance
d_r * m_k (m in {0,1,2} from the mode-switch doubling) and arbitrary old per-edge distances,
check whether removing a robot from the set can ever INCREASE the exact segment gain.
Random instances only; this is evidence, not a proof.
"""

from __future__ import annotations

import numpy as np


def edge_cost(dists_per_robot):
    # (n_robots, n_edges) -> (n_edges,)
    return dists_per_robot.max(axis=0) + 0.01 * dists_per_robot.sum(axis=0)


def gain(old_dists, candidate_dists):
    return float(edge_cost(old_dists).sum() - edge_cost(candidate_dists).sum())


def main():
    rng = np.random.default_rng(7)
    worst = 0.0  # most negative "removal benefit"; <= 0 means monotone
    n_edges = rng.integers(2, 40)
    # random mode-switch multipliers in {0,1,2} with 1 - c(k+1) + c(k) structure
    c = np.zeros(n_edges + 1, dtype=int)
    for k in range(1, n_edges):
        c[k] = int(rng.random() < 0.2)
    m = 1 - c[1:] + c[:-1]

    trials = 0
    for _ in range(4000):
        n_robots = int(rng.integers(2, 6))
        # old per-robot distances: arbitrary, but with the same zero pattern as m
        # (the old path also doubles mode-switch states).
        old = np.zeros((n_robots, n_edges))
        for r in range(n_robots):
            base = rng.random(n_edges)
            old[r] = base * m  # forced zero on m=0 edges, doubled on m=2 edges
            old[r] = old[r] * rng.uniform(0.5, 2.0)
        # straight per-edge distances: d_r * m_k with d_r = length/span
        d = old.sum(axis=1) / float(n_edges)
        cand = np.zeros_like(old)
        for r in range(n_robots):
            cand[r] = d[r] * m
        # total straight length must not exceed the old length (triangle inequality)
        for r in range(n_robots):
            if cand[r].sum() > old[r].sum() + 1e-12:
                old[r] *= 1.01  # keep the constraint; small slack
        for drop in range(1, n_robots):
            cand_sub = cand.copy()
            cand_sub[drop] = old[drop]  # removal keeps the old segment
            removal_benefit = gain(old, cand_sub) - gain(old, cand)
            worst = max(worst, removal_benefit)
            trials += 1
    print(f"trials={trials} worst removal benefit={worst:.6e}")
    print("monotone in the set" if worst <= 1e-9 else "NOT monotone: a counterexample exists")


if __name__ == "__main__":
    main()
