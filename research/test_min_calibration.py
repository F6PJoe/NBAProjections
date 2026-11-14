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


def drops():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("drop last2 -6 x0.25", role_drop_min=6.0, role_drop_w=0.25),
        v("drop last2 -6 x0.5", role_drop_min=6.0, role_drop_w=0.5),
        v("drop last2 -8 x0.5", role_drop_min=8.0, role_drop_w=0.5),
        v("drop last2 -10 x0.5", role_drop_min=10.0, role_drop_w=0.5),
        v("drop last2 -6 x0.75", role_drop_min=6.0, role_drop_w=0.75),
    ]


def older():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("older seasons (w .4)", older_seasons=1),
        v("older seasons (w .2)", older_seasons=1, rate_older_w=0.2),
        v("older seasons (w .6)", older_seasons=1, rate_older_w=0.6),
        v("older seasons min x0.9", older_seasons=1, older_min_factor=0.9),
        v("older 2 seasons back", older_seasons=2),
    ]


def rotation():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("cut below rotation x0.5", rot_cut_factor=0.5),
        v("cut below rotation x0.25", rot_cut_factor=0.25),
        v("cut below rotation+1 x0.5", rot_cut_factor=0.5, rot_slack=1),
        v("cut below rotation+1 x0.25", rot_cut_factor=0.25, rot_slack=1),
        v("cut below rotation-1 x0.5", rot_cut_factor=0.5, rot_slack=-1),
        v("cut x0.5, last 5 games", rot_cut_factor=0.5, rot_games=5),
        v("cut x0.75", rot_cut_factor=0.75),
    ] if "--round2" not in sys.argv else [
        v("current"),
        v("+1 x0.25", rot_cut_factor=0.25, rot_slack=1),
        v("+1 x0.1", rot_cut_factor=0.1, rot_slack=1),
        v("+2 x0.25", rot_cut_factor=0.25, rot_slack=2),
        v("+2 x0.1", rot_cut_factor=0.1, rot_slack=2),
        v("+1 x0.25 max33", rot_cut_factor=0.25, rot_slack=1, max_minutes=33.0),
        v("+1 x0.25 trim>32 .25", rot_cut_factor=0.25, rot_slack=1, high_min_start=32, high_min_shrink=0.25),
        v("+1 x0.25 last5", rot_cut_factor=0.25, rot_slack=1, rot_games=5),
    ]


def retest():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("position tau 0.75 (stricter)", pos_tau=0.75),
        v("position tau 2.0 (looser)", pos_tau=2.0),
        v("DNP-CD weight 0.5", dnp_cd_w=0.5),
        v("new-team prior 0.5", new_team_prior_w=0.5),
        v("role drop -6 x0.5", role_drop_min=6.0, role_drop_w=0.5),
        v("trim >32 by 0.25", high_min_start=32, high_min_shrink=0.25),
        v("max minutes 33", max_minutes=33.0),
    ]


def dnp():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [v("current"), v("DNP-CD weight 0.5", dnp_cd_w=0.5), v("ignore DNP-CDs (weight 0)", dnp_cd_w=1e-9)]


def fill():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    t = dict(high_min_start=32, high_min_shrink=0.25)
    return [
        v("current"),
        v("trim >32", **t),
        v("deep bench gets none", fill_cut_w=0.0),
        v("fill power 0.5", fill_power=0.5),
        v("fill power 1.0", fill_power=1.0),
        v("trim + deep bench none", fill_cut_w=0.0, **t),
        v("trim + power 0.5", fill_power=0.5, **t),
        v("trim + power 1.0", fill_power=1.0, **t),
        v("trim + power 0.5 + deep none", fill_power=0.5, fill_cut_w=0.0, **t),
    ]


def fixes():
    def v(label, **kw):
        p = copy.deepcopy(PARAMS); p.update(kw); return (p, label)
    return [
        v("current"),
        v("jump skips role blend", jump_skips_role_blend=True),
        v("DNP-CD weight 0.5", dnp_cd_w=0.5),
        v("jump skip + DNP-CD 0.5", jump_skips_role_blend=True, dnp_cd_w=0.5),
        v("new-team prior 0.5", new_team_prior_w=0.5),
        v("new-team prior 0.25", new_team_prior_w=0.25),
        v("jump skip + new-team 0.5", jump_skips_role_blend=True, new_team_prior_w=0.5),
        v("all three", jump_skips_role_blend=True, dnp_cd_w=0.5, new_team_prior_w=0.5),
    ]


if __name__ == "__main__":
    if any(f in sys.argv for f in ("--grid", "--combos", "--older", "--drops", "--fixes", "--rotation", "--retest", "--dnp", "--fill")):
        todo = (fill() if "--fill" in sys.argv else dnp() if "--dnp" in sys.argv else retest() if "--retest" in sys.argv else rotation() if "--rotation" in sys.argv else fixes() if "--fixes" in sys.argv else combos() if "--combos" in sys.argv else older() if "--older" in sys.argv
                else drops() if "--drops" in sys.argv else variants())
        with ProcessPoolExecutor(max_workers=8, initializer=_init) as ex:
            rows = list(ex.map(score, todo))
        df = pd.DataFrame(rows).set_index("label")
        base = df.loc["current"]
        for s in SEASONS:
            df[f"d{s}"] = df[f"{s}"] - base[f"{s}"]
            df[f"d{s}_24+"] = df[f"{s}_24+"] - base[f"{s}_24+"]
        pd.set_option("display.width", 250)
        print(df.round(3).to_string())
        name = ("test_fill.csv" if "--fill" in sys.argv else "test_dnp_weight.csv" if "--dnp" in sys.argv else "test_retest_with_rotation.csv" if "--retest" in sys.argv else "test_rotation_size.csv" if "--rotation" in sys.argv else "test_min_fixes.csv" if "--fixes" in sys.argv else "test_min_calibration_combos.csv" if "--combos" in sys.argv else
                "test_older_seasons.csv" if "--older" in sys.argv else
                "test_role_drop.csv" if "--drops" in sys.argv else "test_min_calibration_results.csv")
        df.round(4).to_csv(Path(__file__).with_name(name))
    else:
        _init()
        p, _ = variants()[0]
        r = backtest.run(SEASONS, params=p, use_starters=True, pg=_DATA["pg"], tg=_DATA["tg"], quiet=True)
        r = r[r["a_status"].isin(["played", "dnp_cd"])]
        r["b"] = pd.cut(r["proj_min"], [-1, 5, 10, 15, 20, 26, 30, 33, 36, 50])
        g = r.groupby("b", observed=True).agg(n=("a_min", "size"), proj=("proj_min", "mean"), actual=("a_min", "mean"))
        print(g.assign(min_bias=g["proj"] - g["actual"]).round(2).to_string())
