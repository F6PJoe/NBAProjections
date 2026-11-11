"""Minutes when a player's role changes tonight (starter <-> bench), 3 seasons, known starters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model

seasons = [2025, 2026, 2027]
pg, tg = backtest.load_all(seasons)
hist = model.History(pg, tg)

def report(params, label):
    r = backtest.run(seasons, params, pg=pg, tg=tg, hist=hist, quiet=True)
    r["me"] = r["proj_min"] - r["a_min"]; r["e"] = (r["fp_DK"] - r["a_fp_DK"]).abs()
    down = (r["start_rate"] >= 0.7) & ~r["starter_tonight"]
    up = (r["start_rate"] <= 0.3) & r["starter_tonight"]
    parts = []
    for s in seasons:
        x = r[r["season"] == s]
        parts.append(f"{s} {x['e'].mean():.3f}")
    d, u = r[down], r[up]
    print(f"{label:28s} all: {' | '.join(parts)} || demoted n={len(d)} min bias {d['me'].mean():+.1f} "
          f"DK MAE {d['e'].mean():.2f} || promoted n={len(u)} min bias {u['me'].mean():+.1f} DK MAE {u['e'].mean():.2f}", flush=True)

report({}, "current (4 both ways)")
for dn in (7.0, 10.0, 14.0):
    report({"starter_delta_down": dn}, f"demotion -{dn:.0f}")
for up in (2.0, 7.0):
    report({"starter_delta_up": up}, f"promotion +{up:.0f}")
for k in (1.5, 6.0):
    report({"role_blend_k": k}, f"role history weight k={k}")
