"""Official injury report fix for box-score labels (history.official_out), 3-season backtest.
  python research/test_official_out.py run 0|1|2 OUTDIR   # one level (set via env, history-wide)
  python research/test_official_out.py compare OUTDIR      # graded on the same player-games"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

SEASONS = [2025, 2026, 2027]
if sys.argv[1] == "run":
    os.environ["OFFICIAL_OUT"] = sys.argv[2].replace("gl", "").replace("old", "")
    os.environ["REG_PLAYED_ONLY"] = "1" if sys.argv[2] in ("1", "1gl") else "0"
    os.environ["OFFICIAL_OUT_GL"] = "1" if "gl" in sys.argv[2] else "0"
    import backtest
    r = backtest.run(SEASONS, use_starters=True, quiet=True)
    r[["season", "game_id", "player_id", "fp_DK", "a_fp_DK", "a_min", "a_status"]].to_parquet(
        Path(sys.argv[3]) / f"lvl{sys.argv[2]}.parquet")
else:
    d = Path(sys.argv[2])
    R = {l: pd.read_parquet(d / f"lvl{l}.parquet").drop_duplicates(["game_id", "player_id"]).set_index(["game_id", "player_id"]) for l in (0, 1, 2, "1gl", "2gl", "1old") if (d / f"lvl{l}.parquet").exists()}
    truth = R[2][R[2]["a_status"].isin(["played", "dnp_cd"])]
    idx = truth.index
    for x in R.values():
        idx = idx.intersection(x.index)
    rows = []
    for l, lab in ((0, "current"), ("1gl", "relabel DNPs officially Out (incl. G League)"), ("2gl", "relabel + add missing Out (incl. G League)"),
                   (1, "relabel DNPs officially Out (no G League)"), ("1old", "relabel only, old regular-DNP averaging"), (2, "relabel + add missing Out (no G League)")):
        if l not in R:
            continue
        x = R[l].loc[idx]; e = (x["fp_DK"] - truth.loc[idx, "a_fp_DK"]).abs(); s = truth.loc[idx, "season"]
        a = truth.loc[idx, "a_min"]
        out = {"label": lab}
        for yr in SEASONS:
            out[str(yr)] = e[s == yr].mean(); out[f"{yr}_24+"] = e[(s == yr) & (a >= 24)].mean()
        rows.append(out)
    df = pd.DataFrame(rows).set_index("label")
    for c in list(df.columns):
        df["d" + c] = df[c] - df[c].iloc[0]
    print(f"{len(idx)} common player-games")
    print(df.round(3).to_string())
    df.round(4).to_csv(Path(__file__).with_name("test_official_out.csv"))
