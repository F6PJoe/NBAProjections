"""Missing-star production: spread evenly (0) vs toward high-usage players (eta > 0).
Seasons 2025-2026 with estimated lines; 2026-27 with real lines (DFF pages + saved scoreboards)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model, live
from compare_dff import load_dff

pg, tg = backtest.load_all([2025, 2026, 2027])
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
for eta in (0.0, 0.5, 1.0, 1.5, 2.0):
    out = []
    for seasons, lk in (([2025, 2026], None), ([2027], lookup)):
        r = backtest.run(seasons, {"usage_concentration": eta}, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lk)
        for s in seasons:
            x = r[r["season"] == s]; e = (x["fp_DK"] - x["a_fp_DK"]).abs(); big = x["a_min"] >= 24
            out.append(f"{s}: {e.mean():.3f} (24+ min {e[big].mean():.3f})")
    print(f"eta {eta}: " + " | ".join(out), flush=True)
