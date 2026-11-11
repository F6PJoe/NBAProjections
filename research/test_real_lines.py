"""2026-27 backtest with the real Vegas lines DFF lists for its main-slate games."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model
from compare_dff import load_dff
from names import norm

pg, tg = backtest.load_all([2027])
hist = model.History(pg, tg)
base = backtest.run([2027], pg=pg, tg=tg, hist=hist, quiet=True)
dates = sorted(pd.to_datetime(base["game_date"]).dt.date.unique())
dff = load_dff(dates)
dff["key"] = dff["name"].map(norm)
lines = dff.dropna(subset=["spread", "ou"]).groupby(["game_date", "team"])[["spread", "ou"]].median()
lookup = {k: {"total": v.ou, "spread": v.spread} for k, v in lines.iterrows()}
print(f"real lines for {len(lookup)} team-games")

def score(r, label):
    r = r.copy(); r["game_date"] = pd.to_datetime(r["game_date"]); r["key"] = r["name"].map(norm)
    m = r.merge(dff[["game_date", "team", "key", "ppg_proj"]], on=["game_date", "team", "key"])
    m = m[m["ppg_proj"] > 0]
    e = m["fp_DK"] - m["a_fp_DK"]; d = m["ppg_proj"] - m["a_fp_DK"]
    print(f"{label:38s} ours {e.abs().mean():.3f}  DFF {d.abs().mean():.3f}  (n={len(m)}, ours bias {e.mean():+.2f})")

score(base, "estimated lines (backtest default)")
for extra in ({}, {"pace_alpha": 0.0}, {"pace_alpha": 1.0}, {"spread_min_k": 0.15}, {"spread_min_k": 0.3},
              {"team_norm_beta": {"pts": 0.9}}):
    r = backtest.run([2027], extra, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lookup)
    score(r, f"real lines {extra}")
