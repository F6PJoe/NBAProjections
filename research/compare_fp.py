"""Our walk-forward projections vs FantasyPros' daily projections, graded on box scores.

Same players only (FantasyPros gave them minutes and they dressed). FantasyPros stats are
whole numbers, so part of their error is rounding.

  python research/compare_fp.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import backtest  # noqa: E402
import benchmark  # noqa: E402


def main():
    pg, tg = backtest.load_all([2027])
    r = backtest.run([2027], pg=pg, tg=tg, quiet=True)
    r["game_date"] = pd.to_datetime(r["game_date"])
    frames = []
    for d in sorted(r["game_date"].dt.date.unique()):
        fp = benchmark.fantasypros(d, 2027)
        if not fp.empty:
            frames.append(fp.assign(game_date=pd.Timestamp(d)))
    fp = pd.concat(frames, ignore_index=True)
    m = r.merge(fp[["game_date", "player_id", "fp_DK", "fp_FD", "min"]].rename(
        columns={"fp_DK": "fpros_DK", "fp_FD": "fpros_FD", "min": "fpros_min"}),
        on=["game_date", "player_id"], how="inner")
    m = m[m["fpros_min"] > 0]
    m["e_ours"] = m["fp_DK"] - m["a_fp_DK"]
    m["e_fp"] = m["fpros_DK"] - m["a_fp_DK"]

    def agg(d):
        return pd.Series({"n": len(d), "ours_MAE": d["e_ours"].abs().mean(), "FPros_MAE": d["e_fp"].abs().mean(),
                          "ours_bias": d["e_ours"].mean(), "FPros_bias": d["e_fp"].mean(),
                          "ours_min_MAE": (d["proj_min"] - d["a_min"]).abs().mean(),
                          "FPros_min_MAE": (d["fpros_min"] - d["a_min"]).abs().mean()})
    print(f"===== ours (lineups known) vs FantasyPros, same {len(m)} player-games, "
          f"{m['game_date'].nunique()} nights =====")
    print(agg(m).round(2).to_string())
    print("\nby date:")
    print(m.groupby(m["game_date"].dt.strftime("%m-%d")).apply(agg, include_groups=False).round(2).to_string())
    print(f"\ncorrelation ours vs FantasyPros: {np.corrcoef(m['fp_DK'], m['fpros_DK'])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
