"""Coordinate-descent tuning of PARAMS on one season; checks the result on the others.

  python research/tune.py            -> prints each step and the best settings
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest  # noqa: E402
import model  # noqa: E402
from config import PARAMS  # noqa: E402

TUNE = [2026]
CHECK = [2025, 2027]

GRID = [
    ("min_half_life", [4.0, 6.0, 9.0, 13.0]),
    ("min_prior_games", [0.5, 2.0, 4.0]),
    ("starter_delta", [4.0, 7.0, 10.0]),
    ("max_minutes", [37.0, 38.5, 40.0]),
    ("fill_cap_add", [5.0, 8.0, 12.0]),
    ("fill_power", [0.5, 1.0, 1.5, 2.0]),
    ("pos_tau", [0.6, 1.25, 3.0]),
    ("excess_starter_w", [0.25, 0.5, 1.0]),
    ("high_min_shrink", [0.0, 0.2, 0.4]),
    ("return_factor", [0.75, 0.85, 0.95, 1.0]),
    ("blowout_w", [0.3, 0.5, 1.0]),
    ("min_other_team_w", [0.3, 0.6, 1.0]),
    ("rate_half_life", [12.0, 25.0, 45.0]),
    ("rate_last_season_w", [0.3, 0.6, 1.0]),
    ("pace_alpha", [0.0, 0.5, 0.8, 1.2]),
    ("team_norm_scale", [0.0, 0.5, 1.0, 1.5]),
]


GRID2 = [
    ("min_prior_games", [0.0, 0.25, 0.5, 1.0]),
    ("role_blend_k", [0.0, 1.5, 3.0, 6.0]),
    ("starter_delta", [4.0, 5.5, 7.0]),
    ("max_minutes", [36.0, 37.0, 38.0]),
    ("fill_power", [0.25, 0.5, 0.75]),
    ("excess_starter_w", [0.1, 0.25, 0.4]),
    ("blowout_w", [0.15, 0.3, 0.5]),
    ("min_half_life", [5.0, 6.0, 8.0]),
    ("high_min_shrink", [0.0, 0.1, 0.2]),
]


def apply(params, key, val):
    p = copy.deepcopy(params)
    if key == "team_norm_scale":
        p["team_norm_beta"] = {k: min(1.0, v * val) for k, v in PARAMS["team_norm_beta"].items()}
        p["_team_norm_scale"] = val
    else:
        p[key] = val
    return p


def score(res):
    e = (res["fp_DK"] - res["a_fp_DK"]).abs()
    big = res["proj_min"] >= 20
    return e.mean(), e[big].mean(), (res["proj_min"] - res["a_min"]).abs().mean()


def main():
    import json
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="json file of params to start from")
    ap.add_argument("--grid2", action="store_true")
    a = ap.parse_args()
    global GRID
    if a.grid2:
        GRID = GRID2
    seasons = TUNE + CHECK
    pg, tg = backtest.load_all(seasons)
    hist = model.History(pg, tg)

    def ev(params, ss):
        r = backtest.run(ss, params, pg=pg, tg=tg, hist=hist, quiet=True)
        return {s: score(r[r["season"] == s]) for s in ss}

    best = copy.deepcopy(PARAMS)
    if a.start:
        for k, v in json.load(open(a.start)).items():
            best[k] = v
    start_params = copy.deepcopy(best)
    base = ev(best, TUNE)[TUNE[0]]
    print(f"start: DK MAE {base[0]:.4f}  (proj 20+ {base[1]:.4f})  min MAE {base[2]:.4f}", flush=True)
    best_score = base[0]
    for rnd in range(1):
        for key, vals in GRID:
            cur = best.get("_team_norm_scale", 1.0) if key == "team_norm_scale" else best[key]
            for v in vals:
                if v == cur:
                    continue
                t0 = time.time()
                cand = apply(best, key, v)
                sc = ev(cand, TUNE)[TUNE[0]]
                flag = ""
                if sc[0] < best_score - 0.002:
                    best, best_score, flag = cand, sc[0], "  <-- better"
                print(f"r{rnd} {key}={v}: {sc[0]:.4f} (20+ {sc[1]:.4f}, min {sc[2]:.4f}) "
                      f"{time.time() - t0:.0f}s{flag}", flush=True)
    print("\nBEST:", {k: best[k] for k, _ in GRID if k in best},
          "team_norm_scale", best.get("_team_norm_scale", 1.0), flush=True)
    out = Path(__file__).with_name("best_params.json")
    out.write_text(json.dumps({k: v for k, v in best.items() if not k.startswith("_")}, indent=1))
    print("saved", out, flush=True)
    print("\nCheck seasons (start -> tuned):")
    d = ev(start_params, seasons)
    t = ev(best, seasons)
    for s in seasons:
        print(f"  {s}: DK MAE {d[s][0]:.4f} -> {t[s][0]:.4f} | proj20+ {d[s][1]:.4f} -> {t[s][1]:.4f}"
              f" | min {d[s][2]:.4f} -> {t[s][2]:.4f}", flush=True)


if __name__ == "__main__":
    main()
