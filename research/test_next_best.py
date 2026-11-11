"""When a player is out, does each of his stats go mostly to tonight's next-best player in
that stat? Weight extra production by projected stat total ^ eta. 3 seasons, known starters;
2026-27 with real lines. Also scored on 'key player out' team-games only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest, model, live
from compare_dff import load_dff

seasons = [2025, 2026, 2027]
pg, tg = backtest.load_all(seasons)
hist = model.History(pg, tg)
dates = sorted(pg[pg["season"] == 2027]["game_date"].dt.date.unique())
dff = load_dff(dates)
lines = dff.dropna(subset=["spread", "ou"]).groupby(["game_date", "team"])[["spread", "ou"]].median()
lookup = {k: {"total": v.ou, "spread": v.spread} for k, v in lines.iterrows()}

# team-games where a player averaging 30+ min in his last games did NOT dress
pg = pg.sort_values("game_date")
played = pg[pg["status"] == "played"]
avg = played.groupby("player_id")["min"].transform(lambda m: m.shift(1).rolling(5, min_periods=3).mean())
pg["recent_min"] = avg.reindex(pg.index)
pg["recent_min"] = pg.groupby("player_id")["recent_min"].ffill()
key_out = set(map(tuple, pg[(pg["status"] == "out") & (pg["recent_min"] >= 30)][["game_id", "team"]].values))

def run(params, label):
    parts = []
    for ss, lk in (([2025, 2026], None), ([2027], lookup)):
        r = backtest.run(ss, params, pg=pg, tg=tg, hist=hist, quiet=True, vegas_lookup=lk)
        for s in ss:
            x = r[r["season"] == s]
            e = (x["fp_DK"] - x["a_fp_DK"]).abs()
            ko = [(g, t) in key_out for g, t in zip(x["game_id"], x["team"])]
            ko = pd.Series(ko, index=x.index)
            big = x["a_min"] >= 24
            parts.append(f"{s}: {e.mean():.3f} | star-out {e[ko].mean():.3f} (24+ {e[ko & big].mean():.3f})")
    print(f"{label:24s} " + " || ".join(parts), flush=True)

run({}, "current (even spread)")
for eta in (0.5, 1.0, 1.5):
    run({"usage_concentration": eta, "usage_weight_mode": "total"}, f"next-best by stat ^{eta}")
