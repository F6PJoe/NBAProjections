"""Are our minutes too spread out? Calibration of projected vs actual minutes (3 seasons,
known starters/inactives) and fixes, run in parallel.
  python research/test_min_calibration.py           # calibration table for current settings
  python research/test_min_calibration.py --grid    # all variants, in parallel"""
import copy, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import backtest
from config import PARAMS

SEASONS = [2025, 2026, 2027]
_DATA = {}


def _init():
    _DATA["pg"], _DATA["tg"] = backtest.load_all(SEASONS)


def score(args):
    params, label = args
    r = backtest.run(SEASONS, params=params, use_starters=True, pg=_DATA["pg"], tg=_DATA["tg"], quiet=True)
    r = r[r["a_status"].isin(["played", "dnp_cd"])]
    e = r["fp_DK"] - r["a_fp_DK"]
    out = {"label": label}
    for s in SEASONS:
        m = r["season"] == s
        out[f"{s}"] = e[m].abs().mean()
        out[f"{s}_24+"] = e[m & (r["a_min"] >= 24)].abs().mean()
    st = r["starter_tonight"] == True
    lo = (r["proj_min"] > 10) & (r["proj_min"] <= 20)
    hi = r["proj_min"] > 33
    out["bias10-20"] = (r.loc[lo, "proj_min"] - r.loc[lo, "a_min"]).mean()
    out["bias33+"] = (r.loc[hi, "proj_min"] - r.loc[hi, "a_min"]).mean()
    out["starters_MAE"] = e[st].abs().mean()
    out["bench_MAE"] = e[~st].abs().mean()
    return out


def variants():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("trim >32 by 0.25", high_min_start=32, high_min_shrink=0.25),
        v("trim >32 by 0.5", high_min_start=32, high_min_shrink=0.5),
        v("trim >30 by 0.25", high_min_start=30, high_min_shrink=0.25),
        v("trim >30 by 0.5", high_min_start=30, high_min_shrink=0.5),
        v("starters share cuts 0.5", excess_starter_w=0.5),
        v("starters share cuts 1.0", excess_starter_w=1.0),
        v("max minutes 35", max_minutes=35.0),
        v("max minutes 34", max_minutes=34.0),
        v("starter floor 20", starter_min_floor=20.0),
        v("starter floor 22", starter_min_floor=22.0),
        v("starter floor 24", starter_min_floor=24.0),
        v("jump last2 +6 half", role_jump_k=2, role_jump_min=6.0, role_jump_w=0.5),
        v("jump last2 +6 full", role_jump_k=2, role_jump_min=6.0, role_jump_w=1.0),
        v("jump last3 +5 half", role_jump_k=3, role_jump_min=5.0, role_jump_w=0.5),
        v("jump last3 +5 full", role_jump_k=3, role_jump_min=5.0, role_jump_w=1.0),
    ]


def combos():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    j = dict(role_jump_k=2, role_jump_min=6.0, role_jump_w=0.5)
    return [
        v("current"),
        v("max 33", max_minutes=33.0),
        v("max34 + floor20", max_minutes=34.0, starter_min_floor=20.0),
        v("max34 + floor20 + jump", max_minutes=34.0, starter_min_floor=20.0, **j),
        v("max34 + floor22 + jump", max_minutes=34.0, starter_min_floor=22.0, **j),
        v("max33 + floor20 + jump", max_minutes=33.0, starter_min_floor=20.0, **j),
        v("max35 + floor20 + jump", max_minutes=35.0, starter_min_floor=20.0, **j),
    ]


if __name__ == "__main__":
    if "--grid" in sys.argv or "--combos" in sys.argv:
        todo = combos() if "--combos" in sys.argv else variants()
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as ex:
            rows = list(ex.map(score, todo))
        df = pd.DataFrame(rows).set_index("label")
        base = df.loc["current"]
        for s in SEASONS:
            df[f"d{s}"] = df[f"{s}"] - base[f"{s}"]
            df[f"d{s}_24+"] = df[f"{s}_24+"] - base[f"{s}_24+"]
        pd.set_option("display.width", 250)
        print(df.round(3).to_string())
        name = "test_min_calibration_combos.csv" if "--combos" in sys.argv else "test_min_calibration_results.csv"
        df.round(4).to_csv(Path(__file__).with_name(name))
    else:
        _init()
        p, _ = variants()[0]
        r = backtest.run(SEASONS, params=p, use_starters=True, pg=_DATA["pg"], tg=_DATA["tg"], quiet=True)
        r = r[r["a_status"].isin(["played", "dnp_cd"])]
        r["b"] = pd.cut(r["proj_min"], [-1, 5, 10, 15, 20, 26, 30, 33, 36, 50])
        g = r.groupby("b", observed=True).agg(n=("a_min", "size"), proj=("proj_min", "mean"), actual=("a_min", "mean"))
        print(g.assign(min_bias=g["proj"] - g["actual"]).round(2).to_string())
