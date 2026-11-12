"""Where does DailyFantasyFuel beat us this season? Today's model, walk-forward with known
starters/inactives, same players DFF projected (all classic slates).
  python research/diagnose_dff.py     -> research/diagnose_dff.csv + printed breakdowns"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import backtest, benchmark
from config import CURRENT_SEASON, SCORING
from names import Matcher

pg, tg = backtest.load_all([CURRENT_SEASON])
r = backtest.run([CURRENT_SEASON], pg=pg, tg=tg, quiet=True, use_starters=True)
r = r[r["a_status"].isin(["played", "dnp_cd"])].copy()
frames = []
for d in sorted(r["game_date"].unique()):
    day = pd.Timestamp(d).date()
    x = r[r["game_date"] == d]
    df = benchmark.dff(day, "DK")
    if df.empty:
        continue
    mt = Matcher(zip(x["player_id"], x["name"], x["team"]))
    df["player_id"] = [mt.match(n, t) for n, t in zip(df["name"], df["team"])]
    df = df.dropna(subset=["player_id"]).drop_duplicates("player_id")
    cols = ["player_id", "ppg_proj", "salary", "depth_rank", "l5_avg", "l10_avg", "szn_avg", "inj"]
    frames.append(x.merge(df[cols], on="player_id", how="inner"))
m = pd.concat(frames)
m = m[m["ppg_proj"] > 0]
from config import DATA_HOME
(DATA_HOME / "research").mkdir(exist_ok=True)
m.to_csv(DATA_HOME / "research" / "diagnose_dff.csv", index=False)   # has DFF numbers: private repo only
print(len(m), "player-games,", m["game_date"].nunique(), "nights")
