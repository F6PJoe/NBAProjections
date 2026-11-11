"""The three spots DailyFantasyFuel beats us: bench, players back from 5+ missed games,
120+ team totals. 3 seasons (2026-27 with real lines). Reports overall + each group's MAE/bias."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model, benchmark

seasons = [2025, 2026, 2027]
pg, tg = backtest.load_all(seasons)
hist = model.History(pg, tg)
lookup = {}
for d in sorted(pg[pg["season"] == 2027]["game_date"].dt.date.unique()):
    df = benchmark.dff(d, "DK")
    if not df.empty:
        for t, sp, ou in df.dropna(subset=["spread", "ou"])[["team", "spread", "ou"]].drop_duplicates("team").values:
            lookup[(pd.Timestamp(d), t)] = {"total": ou, "spread": sp}

def run(params, label):
    parts = []
    for ss, lk in (([2025, 2026], None), ([2027], lookup)):
        r = backtest.run(ss, params, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lk)
        r["err"] = r["fp_DK"] - r["a_fp_DK"]; r["e"] = r["err"].abs()
        for s in ss:
            x = r[r["season"] == s]
            g = {"bench": x["proj_min"] < 15, "back5+": x["missed_in_row"] >= 5, "ttl120": x["team_total"] >= 120}
            parts.append(f"{s}: {x['e'].mean():.3f} | " + " ".join(
                f"{k} {x.loc[m, 'e'].mean():.2f}/{x.loc[m, 'err'].mean():+.1f}" for k, m in g.items()))
    print(f"{label:22s} " + " || ".join(parts), flush=True)

run({}, "current")
for w in (0.5, 0.25):
    run({"dnp_cd_w": w}, f"DNP games weight {w}")
for f in (0.95, 1.0):
    run({"return_factor": f}, f"return factor {f}")
for a in (0.8, 1.0):
    run({"pace_alpha": a}, f"pace alpha {a}")
run({"implied_other": 1.0}, "implied other 1.0")
