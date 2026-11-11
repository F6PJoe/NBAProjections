"""Our walk-forward projections vs DailyFantasyFuel's, both graded on real box scores.

Same players only: DFF main-slate players who dressed (played or coach's-decision DNP)
and whom DFF projected above 0.

  python research/compare_dff.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest  # noqa: E402
import benchmark  # noqa: E402
from config import OUTPUT  # noqa: E402
from names import norm  # noqa: E402


def load_dff(dates):
    frames = []
    for d in dates:
        df = benchmark.dff(d, "DK")
        if not df.empty:
            frames.append(df.assign(game_date=pd.Timestamp(d)))
    return pd.concat(frames, ignore_index=True)


def main():
    pg, tg = backtest.load_all([2027])
    res = {}
    for tag, starters in (("lineups known", True), ("no lineups", False)):
        r = backtest.run([2027], use_starters=starters, pg=pg, tg=tg, quiet=True)
        res[tag] = r
    dates = sorted(pd.to_datetime(res["lineups known"]["game_date"]).dt.date.unique())
    dff = load_dff(dates)
    from names import Matcher
    people = pg.drop_duplicates(["player_id", "team"])[["player_id", "name", "team"]]
    mt = Matcher(zip(people["player_id"], people["name"], people["team"]))
    dff["player_id"] = [mt.match(n, t) for n, t in zip(dff["name"], dff["team"])]
    print(f"DFF rows: {len(dff)}  (projected 0: {(dff['ppg_proj'] <= 0).sum()})")

    for tag, r in res.items():
        r = r.copy()
        r["game_date"] = pd.to_datetime(r["game_date"])
        m = r.merge(dff.dropna(subset=["player_id"])[["game_date", "player_id", "ppg_proj", "salary", "inj"]],
                    on=["game_date", "player_id"], how="inner")
        m = m[m["ppg_proj"] > 0]
        m["e_ours"] = m["fp_DK"] - m["a_fp_DK"]
        m["e_dff"] = m["ppg_proj"] - m["a_fp_DK"]
        print(f"\n===== ours ({tag}) vs DFF, same {len(m)} player-games "
              f"(matched {len(m)} of {int((dff['ppg_proj'] > 0).sum())} DFF rows) =====")

        def agg(d):
            return pd.Series({"n": len(d), "ours_MAE": d["e_ours"].abs().mean(),
                              "DFF_MAE": d["e_dff"].abs().mean(),
                              "ours_bias": d["e_ours"].mean(), "DFF_bias": d["e_dff"].mean(),
                              "ours_closer_%": 100 * (d["e_ours"].abs() < d["e_dff"].abs()).mean()})
        print(agg(m).round(2).to_string())
        print("\nby salary:")
        m["sal"] = pd.cut(m["salary"], [0, 4500, 6500, 8500, 20000],
                          labels=["<4.5k", "4.5-6.5k", "6.5-8.5k", "8.5k+"])
        print(m.groupby("sal", observed=True).apply(agg, include_groups=False).round(2).to_string())
        print("\nby date:")
        print(m.groupby(m["game_date"].dt.strftime("%m-%d")).apply(agg, include_groups=False)
              .round(2).to_string())
        print(f"\ncorrelation ours vs DFF: {np.corrcoef(m['fp_DK'], m['ppg_proj'])[0, 1]:.3f}")
        m.to_csv(OUTPUT / f"compare_dff_{tag.replace(' ', '_')}.csv", index=False)


if __name__ == "__main__":
    main()
