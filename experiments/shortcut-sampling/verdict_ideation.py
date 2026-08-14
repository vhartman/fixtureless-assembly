"""Analysis of the ideation-round sampler bench (bench3) vs the landed baseline.

Reads  <scene>-bench3.json  (+ <scene>-bench2.json for the Halton-edge reference of the R2
decision rule) and writes  ideation_bench.json.
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = ["car-mini-lim", "car-mini-opt", "cross8-nb"]
ARMS = ["baseline", "r2", "sweep_bisect", "sweep_bisect_loc", "gainprop"]
CHECKS = [100, 250, 500, 1000]
FRACS = [0.25, 0.5, 0.75, 1.0]

D = {s: json.load(open(os.path.join(HERE, f"{s}-bench3.json"))) for s in SCENES}
try:
    D2 = {s: json.load(open(os.path.join(HERE, f"{s}-bench2.json"))) for s in SCENES}
except FileNotFoundError:
    D2 = None


def runs(sc, arm):
    return sorted([r for r in D[sc]["runs"] if r["arm"] == arm], key=lambda r: r["seed"])


def cost_at_checks(rec, n):
    c = rec["curve"]["cost"]
    if not c:
        return rec["cost_initial"]
    return float(c[min(n, len(c)) - 1])


def cost_at_axis(rec, axis, budget):
    xs = rec["curve"][axis]
    cs = rec["curve"]["cost"]
    if not xs or xs[0] > budget:
        return float(rec["cost_initial"])
    k = int(np.searchsorted(np.asarray(xs), budget, side="right")) - 1
    return float(cs[k])


def mwu(a, b):
    """Exact two-sided Mann-Whitney p (permutation over combinations), as in verdict2."""
    a, b = list(a), list(b)
    na, nb = len(a), len(b)
    allv = sorted(a + b)

    def U(A, B):
        return sum(1 for x in A for y in B if x < y) + 0.5 * sum(
            1 for x in A for y in B if x == y)

    obs = U(a, b)
    cnt = tot = 0
    for combo in itertools.combinations(range(na + nb), na):
        A = [allv[i] for i in combo]
        Bv = [allv[i] for i in range(na + nb) if i not in combo]
        u = U(A, Bv)
        tot += 1
        if min(u, na * nb - u) <= min(obs, na * nb - obs) + 1e-12:
            cnt += 1
    return cnt / tot


def stat(v):
    v = np.asarray(v, dtype=float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                vals=[float(x) for x in v])


O = {"scenes": SCENES, "arms": ARMS,
     "budget": D[SCENES[0]]["budget"],
     "resolution": {s: D[s]["resolution"] for s in SCENES},
     "tolerance": {s: D[s]["tolerance"] for s in SCENES}}

# ---------------------------------------------------------------------------------------
print("=" * 110)
print("A. COST AT CANDIDATE CHECKS (mean +- sd over 5 seeds; exact MWU vs baseline)")
O["A_cost_at_checks"] = {}
for sc in SCENES:
    print(f"\n  {sc}  (prepared cost {D[sc]['prepared']['cost']:.2f})")
    O["A_cost_at_checks"][sc] = {}
    base = {n: [cost_at_checks(r, n) for r in runs(sc, "baseline")] for n in CHECKS}
    hdr = "".join(f"{'@'+str(n):>22}" for n in CHECKS)
    print(f"    {'arm':<18}{hdr}")
    for arm in ARMS:
        cells = []
        O["A_cost_at_checks"][sc][arm] = {}
        for n in CHECKS:
            v = [cost_at_checks(r, n) for r in runs(sc, arm)]
            s = stat(v)
            p = None if arm == "baseline" else mwu(v, base[n])
            O["A_cost_at_checks"][sc][arm][str(n)] = dict(**s, p_vs_baseline=p)
            cells.append(f"{s['mean']:8.2f}+-{s['sd']:5.2f}"
                         + (f" p={p:.3f}" if p is not None else "        "))
        print(f"    {arm:<18}" + "".join(f"{c:>22}" for c in cells))

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("B. COST AT MATCHED CONFIGURATION CHECKS")
O["B_cost_at_configs"] = {}
for sc in SCENES:
    lo = min(r["curve"]["configs"][-1] for a in ARMS for r in runs(sc, a))
    grid = [int(round(lo * f)) for f in FRACS]
    O["B_cost_at_configs"][sc] = {"grid": grid}
    print(f"\n  {sc}  (common config budget {lo})")
    base = {g: [cost_at_axis(r, "configs", g) for r in runs(sc, "baseline")] for g in grid}
    print(f"    {'arm':<18}" + "".join(f"{'@'+str(g):>22}" for g in grid))
    for arm in ARMS:
        cells = []
        O["B_cost_at_configs"][sc][arm] = {}
        for g in grid:
            v = [cost_at_axis(r, "configs", g) for r in runs(sc, arm)]
            s = stat(v)
            p = None if arm == "baseline" else mwu(v, base[g])
            O["B_cost_at_configs"][sc][arm][str(g)] = dict(**s, p_vs_baseline=p)
            cells.append(f"{s['mean']:8.2f}+-{s['sd']:5.2f}"
                         + (f" p={p:.3f}" if p is not None else "        "))
        print(f"    {arm:<18}" + "".join(f"{c:>22}" for c in cells))

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("C. COST AT MATCHED WALL TIME (arms interleaved across seeds)")
O["C_cost_at_time"] = {}
for sc in SCENES:
    lo = min(r["curve"]["time"][-1] for a in ARMS for r in runs(sc, a))
    grid = [round(lo * f, 3) for f in FRACS]
    O["C_cost_at_time"][sc] = {"grid": grid}
    print(f"\n  {sc}  (common wall budget {lo:.2f}s)")
    base = {g: [cost_at_axis(r, "time", g) for r in runs(sc, "baseline")] for g in grid}
    print(f"    {'arm':<18}" + "".join(f"{'@'+str(g)+'s':>22}" for g in grid))
    for arm in ARMS:
        cells = []
        O["C_cost_at_time"][sc][arm] = {}
        for g in grid:
            v = [cost_at_axis(r, "time", g) for r in runs(sc, arm)]
            s = stat(v)
            p = None if arm == "baseline" else mwu(v, base[g])
            O["C_cost_at_time"][sc][arm][str(g)] = dict(**s, p_vs_baseline=p)
            cells.append(f"{s['mean']:8.2f}+-{s['sd']:5.2f}"
                         + (f" p={p:.3f}" if p is not None else "        "))
        print(f"    {arm:<18}" + "".join(f"{c:>22}" for c in cells))

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("D. PER-SAMPLE OVERHEAD (the axis that killed Halton)")
O["D_overhead"] = {}
for sc in SCENES:
    O["D_overhead"][sc] = {}
    print(f"\n  {sc}")
    print(f"    {'arm':<18}{'us/config':>12}{'sampler us/call':>17}{'sampler share %':>17}"
          f"{'wall s':>9}{'configs':>10}")
    for arm in ARMS:
        rs = runs(sc, arm)
        upc = stat([r["us_per_config"] for r in rs])
        spc = stat([r["sampler_us_per_call"] for r in rs])
        share = stat([100.0 * r["sampler_time"] / r["wall"] for r in rs])
        wl = stat([r["wall"] for r in rs])
        cf = stat([r["configs"] for r in rs])
        O["D_overhead"][sc][arm] = dict(us_per_config=upc, sampler_us_per_call=spc,
                                        sampler_share_pct=share, wall=wl, configs=cf)
        print(f"    {arm:<18}{upc['mean']:>12.1f}{spc['mean']:>17.1f}{share['mean']:>17.2f}"
              f"{wl['mean']:>9.2f}{cf['mean']:>10.0f}")

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("E. SHIPPED-BUDGET ENDPOINT + BOOKKEEPING")
O["E_endpoint"] = {}
for sc in SCENES:
    O["E_endpoint"][sc] = {}
    base = [r["cost_final"] for r in runs(sc, "baseline")]
    print(f"\n  {sc}")
    print(f"    {'arm':<18}{'cost_final':>20}{'p':>8}{'chk':>7}{'acc':>6}{'iters':>7}"
          f"{'short':>7}{'costrej':>8}{'stop':>14}")
    for arm in ARMS:
        rs = runs(sc, arm)
        v = [r["cost_final"] for r in rs]
        s = stat(v)
        p = None if arm == "baseline" else mwu(v, base)
        O["E_endpoint"][sc][arm] = dict(
            cost_final=s, p_vs_baseline=p,
            pchecks=stat([r["pchecks"] for r in rs]),
            accepted=stat([r["accepted"] for r in rs]),
            iters=stat([r["iters"] for r in rs]),
            stop=[r["stop_reason"] for r in rs],
            extra=[r["extra"] for r in rs])
        print(f"    {arm:<18}{s['mean']:>12.2f}+-{s['sd']:>5.2f}"
              f"{('  --  ' if p is None else f'{p:>8.3f}')}"
              f"{np.mean([r['pchecks'] for r in rs]):>7.0f}"
              f"{np.mean([r['accepted'] for r in rs]):>6.0f}"
              f"{np.mean([r['iters'] for r in rs]):>7.0f}"
              f"{np.mean([r['skipped_short'] for r in rs]):>7.0f}"
              f"{np.mean([r['cost_rejects'] for r in rs]):>8.0f}"
              f"{rs[0]['stop_reason']:>14}")

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("F. INVARIANTS")
O["F_invariants"] = {}
bad_total = 0
for sc in SCENES:
    per = {}
    for arm in ARMS:
        bad = [r["seed"] for r in runs(sc, arm) if not r["invariants"]["all_ok"]]
        per[arm] = dict(n_ok=len(runs(sc, arm)) - len(bad), bad_seeds=bad,
                        detail=[{k: v for k, v in r["invariants"].items() if k != "colliding_states"}
                                for r in runs(sc, arm) if not r["invariants"]["all_ok"]])
        bad_total += len(bad)
    O["F_invariants"][sc] = per
    print(f"  {sc}: " + ", ".join(f"{a}:{per[a]['n_ok']}/5" for a in ARMS))
print(f"  TOTAL violations: {bad_total}")

# ---------------------------------------------------------------------------------------
# G. the R2 decision rule from the ideation output, using bench2's Halton edge as reference
print("\n" + "=" * 110)
print("G. R2 DECISION RULE (default an LDS only if...)")
O["G_r2_rule"] = {}
if D2 is not None:
    def v2(sc, arm, n):
        rs = sorted([r for r in D2[sc]["main"] if r["arm"] == arm], key=lambda r: r["seed"])
        out = []
        for r in rs:
            c = r["curve"]["cost"]
            out.append(float(c[min(n, len(c)) - 1]) if c else r["cost_initial"])
        return out

    for sc in SCENES:
        per = {}
        for n in [100, 250, 500]:
            halton_edge = float(np.mean(v2(sc, "uniform_perrun", n))
                                - np.mean(v2(sc, "halton_perrun", n)))
            r2_edge = float(
                np.mean([cost_at_checks(r, n) for r in runs(sc, "baseline")])
                - np.mean([cost_at_checks(r, n) for r in runs(sc, "r2")]))
            per[str(n)] = dict(halton_edge_bench2=halton_edge, r2_edge_bench3=r2_edge,
                               recovers_half=bool(r2_edge >= 0.5 * halton_edge))
        O["G_r2_rule"][sc] = per
        print(f"  {sc}: " + "  ".join(
            f"@{n}: halton {per[str(n)]['halton_edge_bench2']:+.2f} vs r2 "
            f"{per[str(n)]['r2_edge_bench3']:+.2f} ({'PASS' if per[str(n)]['recovers_half'] else 'fail'})"
            for n in ["100", "250", "500"]))
    # (b) no regression beyond noise anywhere; (c) wall within 2% of baseline
    reg = []
    for sc in SCENES:
        for n in CHECKS:
            e = O["A_cost_at_checks"][sc]["r2"][str(n)]
            b = O["A_cost_at_checks"][sc]["baseline"][str(n)]
            if e["mean"] > b["mean"] and (e["p_vs_baseline"] or 1) < 0.05:
                reg.append((sc, n))
    wall_ok = all(
        O["D_overhead"][sc]["r2"]["wall"]["mean"]
        <= 1.02 * O["D_overhead"][sc]["baseline"]["wall"]["mean"] for sc in SCENES)
    n_pass = sum(1 for sc in SCENES
                 if sum(O["G_r2_rule"][sc][str(n)]["recovers_half"] for n in [100, 250, 500]) >= 2)
    O["G_r2_rule"]["summary"] = dict(
        a_scenes_recovering=n_pass, b_significant_regressions=reg, c_wall_within_2pct=wall_ok,
        verdict="DEFAULT_R2" if (n_pass >= 2 and not reg and wall_ok) else "NO_LDS_DEFAULT")
    print(f"  (a) scenes recovering >=half of Halton edge on >=2 budgets: {n_pass}/3"
          f"   (b) significant regressions: {reg}   (c) wall within 2%: {wall_ok}")
    print(f"  -> {O['G_r2_rule']['summary']['verdict']}")

# ---------------------------------------------------------------------------------------
print("\n" + "=" * 110)
print("H. BLUNT SUMMARY: does ANY arm beat baseline beyond seed noise on ANY axis?")
O["H_summary"] = {}
sig = []
for sc in SCENES:
    for arm in ARMS[1:]:
        for sect, name in [("A_cost_at_checks", "checks"), ("B_cost_at_configs", "configs"),
                           ("C_cost_at_time", "time")]:
            for key, cell in O[sect][sc][arm].items():
                if not isinstance(cell, dict) or "p_vs_baseline" not in cell:
                    continue
                b = O[sect][sc]["baseline"][key]
                if cell["p_vs_baseline"] is not None and cell["p_vs_baseline"] < 0.05 \
                        and cell["mean"] < b["mean"]:
                    sig.append(dict(scene=sc, arm=arm, axis=name, budget=key,
                                    arm_mean=cell["mean"], base_mean=b["mean"],
                                    p=cell["p_vs_baseline"]))
O["H_summary"]["significant_wins"] = sig
if sig:
    for s in sig:
        print(f"  {s['scene']:14s} {s['arm']:17s} {s['axis']:8s}@{s['budget']:>7}: "
              f"{s['base_mean']:.2f} -> {s['arm_mean']:.2f}  p={s['p']:.3f}")
else:
    print("  NOTHING beats the landed uniform-per-run baseline beyond seed noise. Null result.")

json.dump(O, open(os.path.join(HERE, "ideation_bench.json"), "w"), indent=1, default=float)
print("\nwrote", os.path.join(HERE, "ideation_bench.json"))
