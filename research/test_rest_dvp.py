"""Rest (back-to-backs) and opponent defense by position. 3 seasons, known starters;
2026-27 with real lines. Also scored on the games each idea is about."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model, live, benchmark

seasons = [2025, 2026, 2027]
pg, tg = backtest.load_all(seasons)
hist = model.History(pg, tg)
dates = sorted(pg[pg["season"] == 2027]["game_date"].dt.date.unique())
lookup = {}
for d in dates:
    df = benchmark.dff(d, "DK")
    if not df.empty:
        for t, sp, ou in df.dropna(subset=["spread", "ou"])[["team", "spread", "ou"]].drop_duplicates("team").values:
            lookup[(pd.Timestamp(d), t)] = {"total": ou, "spread": sp}
# which team-games are night 2 of a back-to-back
tgs = tg.sort_values(["team", "game_date"])
tgs["b2b"] = tgs.groupby("team")["game_date"].diff().dt.days == 1
b2b = set(map(tuple, tgs[tgs["b2b"]][["game_id", "team"]].values))

def run(params, label):
    parts = []
    for ss, lk in (([2025, 2026], None), ([2027], lookup)):
        r = backtest.run(ss, params, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lk)
        r["e"] = (r["fp_DK"] - r["a_fp_DK"]).abs()
        r["b2b"] = [(g, t) in b2b for g, t in zip(r["game_id"], r["team"])]
        for s in ss:
            x = r[r["season"] == s]; big = x["a_min"] >= 24
            parts.append(f"{s}: {x['e'].mean():.3f} 24+ {x.loc[big, 'e'].mean():.3f} b2b-starters "
                         f"{x.loc[x['b2b'] & x['starter_tonight'], 'e'].mean():.3f}")
    print(f"{label:26s} " + " || ".join(parts), flush=True)

run({}, "current")
for k in (1.0, 2.0):
    run({"b2b_starter_min": k}, f"b2b starters -{k:.0f} min")
for f in (1.02, 1.04):
    run({"opp_b2b_factor": f}, f"opponent on b2b x{f}")
for g in (0.5, 1.0):
    run({"dvp_strength": g}, f"defense by position ^{g}")
