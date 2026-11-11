"""Walk-forward backtest: project every past game using only what was known before it.

Availability (who dressed) and, by default, who started come from the box score -
the same facts a run right before tip-off has. Vegas lines are not archived for
past games, so the backtest uses the model's own team-points estimate; live runs
use the real lines and should do a bit better.

  python backtest.py                      # seasons 2026 (2025-26) and 2027 (so far)
  python backtest.py --no-starters        # as if run before lineups are known
  python backtest.py --set starter_delta=6 --set min_half_life=8
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd

import model
from config import PARAMS, STATS, OUTPUT, CURRENT_SEASON
from history import load_player_games, load_rosters, position_index, team_games


def load_all(seasons):
    need = sorted(set(seasons) | {s - 1 for s in seasons} | {s - 2 for s in seasons})
    pg = load_player_games(need)
    pos = {}
    for yr in need:  # later seasons overwrite earlier ones
        r = load_rosters(yr)
        pos.update(dict(zip(r["player_id"].astype(int), r["position"].map(position_index))))
    pg["pos"] = pg["player_id"].map(pos).fillna(3.0)
    model.set_positions(dict(zip(pg["player_id"], pg["pos"])))
    tg = team_games(pg)
    return pg, tg


def actual_fp(df):
    tens = sum((df[s] >= 10).astype(int) for s in ("pts", "reb", "ast", "stl", "blk"))
    dk = (df["pts"] + 0.5 * df["fg3m"] + 1.25 * df["reb"] + 1.5 * df["ast"] + 2 * df["stl"]
          + 2 * df["blk"] - 0.5 * df["tov"] + 1.5 * (tens >= 2) + 3 * (tens >= 3))
    fd = df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"] + 3 * df["stl"] + 3 * df["blk"] - df["tov"]
    return dk, fd


def run(seasons, params=None, use_starters=True, pg=None, tg=None, hist=None, quiet=False,
        vegas_lookup=None, only_dates=None):
    """vegas_lookup: optional {(game_date Timestamp, team): {"total", "spread"}} of real lines.
    only_dates: optional list of dates to project (default: every game in the seasons)."""
    P = copy.deepcopy(PARAMS)
    if params:
        for k, v in params.items():
            if isinstance(P.get(k), dict) and isinstance(v, dict):
                P[k].update(v)
            else:
                P[k] = v
    if pg is None:
        pg, tg = load_all(seasons)
    if hist is None:
        hist = model.History(pg, tg)
    out = []
    played_by_tg = (pg[pg["status"] == "played"].groupby(["game_id", "team"])["player_id"]
                    .apply(set).to_dict())
    for season in seasons:
        prev = pg[(pg["season"] == season - 1)]
        pos_means = model.position_means(prev if len(prev) else pg[pg["season"] < season])
        cur = pg[pg["season"] == season]
        if only_dates is not None:
            cur = cur[cur["game_date"].isin([pd.Timestamp(d) for d in only_dates])]
        t0 = time.time()
        for (gid, team), g in cur.groupby(["game_id", "team"], sort=False):
            day = model.day_num(g["game_date"].iloc[0])
            avail = g[g["status"].isin(["played", "dnp_cd"])]
            if len(avail) < 5:
                continue
            listed_out = set(g.loc[g["status"] == "out", "player_id"])
            th = hist.team_before(team, day)
            recent = set()
            if th is not None and len(th["game_id"]):
                for rg in th["game_id"][-10:]:
                    recent |= played_by_tg.get((rg, team), set())
            out_ids = (listed_out | recent) - set(avail["player_id"])
            roster = [{"player_id": int(r.player_id), "name": r.name, "pos": float(r.pos),
                       "starter": bool(r.starter) if use_starters else None}
                      for r in avail.itertuples()]
            vegas = (vegas_lookup or {}).get((g["game_date"].iloc[0], team))
            rows = model.project_team(hist, team, g["opp"].iloc[0], bool(g["is_home"].iloc[0]),
                                      day, season, roster, out_ids, pos_means, P, vegas)
            res = pd.DataFrame(rows)
            res["game_id"], res["game_date"], res["season"] = gid, g["game_date"].iloc[0], season
            act = avail.set_index("player_id")[["min"] + STATS + ["status"]].add_prefix("a_")
            res = res.join(act, on="player_id")
            out.append(res)
        if not quiet:
            print(f"  season {season}: {time.time() - t0:.0f}s")
    res = pd.concat(out, ignore_index=True)
    a = res[[f"a_{s}" for s in STATS]].rename(columns=lambda c: c[2:])
    res["a_fp_DK"], res["a_fp_FD"] = actual_fp(a)
    return res


def naive_baseline(pg, res):
    """Average DK points in the player's last 10 games played (season to date, else last season)."""
    pg = pg.sort_values("game_date")
    played = pg[pg["status"] == "played"].copy()
    played["fp"], _ = actual_fp(played)
    played["naive"] = (played.groupby(["player_id", "season"])["fp"]
                       .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()))
    played["naive_min"] = (played.groupby(["player_id", "season"])["min"]
                           .transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean()))
    # value going INTO each game, carried to DNP rows too
    allg = pg[["player_id", "game_id", "season", "game_date"]].merge(
        played[["player_id", "game_id", "naive", "naive_min"]], on=["player_id", "game_id"], how="left")
    allg = allg.sort_values("game_date")
    last = played.groupby(["player_id", "season"])["fp"].mean().rename("last_season").reset_index()
    last["season"] += 1
    allg = allg.merge(last, on=["player_id", "season"], how="left")
    # for DNP rows, carry forward the previous naive
    allg["naive"] = allg.groupby(["player_id", "season"])["naive"].transform(lambda s: s.ffill())
    allg["naive"] = allg["naive"].fillna(allg["last_season"])
    return res.merge(allg[["player_id", "game_id", "naive", "naive_min"]], on=["player_id", "game_id"], how="left")


def report(res, label=""):
    r = res.copy()
    r["err"] = r["fp_DK"] - r["a_fp_DK"]
    r["min_err"] = r["proj_min"] - r["a_min"]
    r["month"] = pd.to_datetime(r["game_date"]).dt.strftime("%Y-%m")
    has_naive = r["naive"].notna()
    r["n_err"] = r["naive"] - r["a_fp_DK"]

    def agg(d):
        hn = d["naive"].notna()
        return pd.Series({
            "players": len(d),
            "DK_MAE": d["err"].abs().mean(),
            "DK_RMSE": np.sqrt((d["err"] ** 2).mean()),
            "naive_MAE": d.loc[hn, "n_err"].abs().mean(),
            "model_MAE_same_rows": d.loc[hn, "err"].abs().mean(),
            "min_MAE": d["min_err"].abs().mean(),
            "bias": d["err"].mean(),
        })

    print(f"\n=== {label} ===")
    print(r.groupby("season").apply(agg, include_groups=False).round(3).to_string())
    print("\nby month:")
    print(r.groupby("month").apply(agg, include_groups=False).round(3).to_string())
    big = r[r["a_min"] >= 20]
    print(f"\nplayers who played 20+ min: model MAE {big['err'].abs().mean():.3f}, "
          f"naive MAE {big.loc[big['naive'].notna(), 'n_err'].abs().mean():.3f}")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[CURRENT_SEASON - 1, CURRENT_SEASON])
    ap.add_argument("--no-starters", action="store_true")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split("=")
        params[k] = float(v)
    pg, tg = load_all(a.seasons)
    res = run(a.seasons, params, use_starters=not a.no_starters, pg=pg, tg=tg)
    res = naive_baseline(pg, res)
    tag = "no_starters" if a.no_starters else "with_starters"
    report(res, tag)
    OUTPUT.mkdir(exist_ok=True)
    res.to_csv(OUTPUT / f"backtest_{tag}.csv", index=False)
