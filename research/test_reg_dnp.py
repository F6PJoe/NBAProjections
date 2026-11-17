"""Box scores from 2026-27 label injured regulars "DNP - Coach's Decision". Test treating a
regular's DNP-CD as 'out' (history.reg_dnp_min). Scored on the same played player-games.
  python research/test_reg_dnp.py"""
import os, sys, subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

SEASONS = [2025, 2026, 2027]


def one(thr):
    os.environ["REG_DNP_MIN"] = str(thr)
    import backtest
    pg, tg = backtest.load_all(SEASONS)
    r = backtest.run(SEASONS, pg=pg, tg=tg, quiet=True, use_starters=True)
    r = r[r["a_status"] == "played"][["player_id", "game_date", "season", "a_min", "fp_DK", "a_fp_DK", "proj_min"]]
    return thr, r


if __name__ == "__main__":
    ths = [int(x) for x in sys.argv[1:]] or [0, 24, 20, 15]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(one, ths))
    keys = None
    for r in res.values():
        k = set(zip(r["player_id"], r["game_date"]))
        keys = k if keys is None else keys & k
    rows = []
    for thr, r in res.items():
        r = r[[k in keys for k in zip(r["player_id"], r["game_date"])]]
        e = r["fp_DK"] - r["a_fp_DK"]
        row = {"reg_dnp_min": thr, "n": len(r)}
        for s in SEASONS:
            m = r["season"] == s
            row[f"{s}"] = e[m].abs().mean()
            row[f"{s}_24+"] = e[m & (r["a_min"] >= 24)].abs().mean()
            row[f"{s}_minMAE"] = (r["proj_min"] - r["a_min"])[m].abs().mean()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("reg_dnp_min")
    pd.set_option("display.width", 250)
    print((df - df.loc[0]).round(3).to_string())
    print(df.round(3).to_string())
    df.round(4).to_csv(Path(__file__).with_name("test_reg_dnp.csv"))
