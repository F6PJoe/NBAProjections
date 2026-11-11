"""Does leaning harder on Vegas (team totals / spreads) help? 2026-27 with real lines.
Lines: DFF pages (main-slate games) + the scoreboards our live runs saved."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import backtest, model, live
from compare_dff import load_dff

pg, tg = backtest.load_all([2027])
hist = model.History(pg, tg)
dates = sorted(pg[pg["season"] == 2027]["game_date"].dt.date.unique())
dff = load_dff(dates)
lines = dff.dropna(subset=["spread", "ou"]).groupby(["game_date", "team"])[["spread", "ou"]].median()
lookup = {k: {"total": v.ou, "spread": v.spread} for k, v in lines.iterrows()}
for d in dates:
    for g in live.saved_games(d).itertuples():
        if g.total and g.spread_home is not None:
            lookup[(pd.Timestamp(d), g.home)] = {"total": g.total, "spread": g.spread_home}
            lookup[(pd.Timestamp(d), g.away)] = {"total": g.total, "spread": -g.spread_home}
print(f"real lines for {len(lookup)} team-games")

def score(r, label):
    r = r.copy()
    r["spread"] = [lookup.get((pd.Timestamp(d), t), {}).get("spread") for d, t in zip(r["game_date"], r["team"])]
    w = r.dropna(subset=["spread"])
    e = w["fp_DK"] - w["a_fp_DK"]
    dog = w["spread"] >= 6; fav = w["spread"] <= -6
    print(f"{label:34s} MAE {e.abs().mean():.3f} (n={len(w)}) | bias: big dogs {e[dog].mean():+.2f}  "
          f"big favs {e[fav].mean():+.2f}  close {e[~dog & ~fav].mean():+.2f}", flush=True)
    return e.abs().mean()

variants = [
    ("current", {}),
    ("team pts follow Vegas fully", {"team_norm_beta": {"pts": 1.0}}),
    ("+ other stats follow Vegas ^0.5", {"team_norm_beta": {"pts": 1.0, "ast": 0.8, "reb": 0.6}, "implied_other": 0.5}),
    ("+ other stats follow Vegas ^1.0", {"team_norm_beta": {"pts": 1.0, "ast": 0.8, "reb": 0.6}, "implied_other": 1.0}),
    ("spread cuts starter min 0.15", {"spread_min_k": 0.15}),
    ("spread cuts starter min 0.30", {"spread_min_k": 0.30}),
    ("Vegas ^0.5 + spread 0.15", {"team_norm_beta": {"pts": 1.0, "ast": 0.8, "reb": 0.6}, "implied_other": 0.5, "spread_min_k": 0.15}),
]
for label, params in variants:
    r = backtest.run([2027], params, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lookup)
    score(r, label)
