import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model
seasons = [2025, 2026, 2027]
pg, tg = backtest.load_all(seasons)
hist = model.History(pg, tg)
for sc in (1.0, 0.5, 0.25, 0.1, 0.0):
    r = backtest.run(seasons, {"rate_prior_scale": sc}, pg=pg, tg=tg, hist=hist, quiet=True)
    e = (r["fp_DK"] - r["a_fp_DK"])
    out = []
    for s in seasons:
        x = r[r["season"] == s]; ee = x["fp_DK"] - x["a_fp_DK"]; big = x["proj_min"] >= 28
        out.append(f"{s}: MAE {ee.abs().mean():.3f} (28+ min {ee[big].abs().mean():.3f}, bias {ee[big].mean():+.2f})")
    print(f"scale {sc}: " + " | ".join(out), flush=True)
