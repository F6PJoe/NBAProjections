"""Score past nights' projections against the real box scores.

  python grade.py                 # every graded-able date in output/
  python grade.py --date 2026-11-06

Uses the LAST projections file saved before that night's first tip-off (what the
site was showing at lock) and appends one line per date to output/accuracy_log.csv.
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

import sdv
from backtest import actual_fp
from config import CURRENT_SEASON, OUTPUT
from history import load_player_games


def pick_file(folder):
    files = sorted(folder.glob("projections_all_*.csv"))
    if not files:
        return None
    first = pd.read_csv(files[-1], usecols=["tip"])["tip"].dropna()
    if first.empty:
        return files[-1]
    lock = pd.to_datetime(first, utc=True).min().tz_convert("America/New_York")
    before = [f for f in files if f.stem[-4:] < lock.strftime("%H%M")]
    return before[-1] if before else files[0]


def grade(date, pg):
    folder = OUTPUT / f"{date:%Y-%m-%d}"
    f = pick_file(folder)
    if f is None:
        return None
    proj = pd.read_csv(f)
    act = pg[pg["game_date"] == pd.Timestamp(date)]
    if act.empty:
        return None
    act = act.assign(a_DK=actual_fp(act)[0], a_FD=actual_fp(act)[1])
    act = act.rename(columns={"min": "a_min", "status": "a_status"})
    m = proj.merge(act[["player_id", "a_min", "a_status", "a_DK", "a_FD"]], on="player_id", how="inner")
    m = m[m["a_status"].isin(["played", "dnp_cd"]) & (m["Status"].fillna("") != "OUT")]
    e_dk = m["fp_DK"] - m["a_DK"]
    big = m["proj_min"] >= 20
    row = {"date": f"{date:%Y-%m-%d}", "file": f.name, "players": len(m),
           "DK_MAE": e_dk.abs().mean(), "FD_MAE": (m["fp_FD"] - m["a_FD"]).abs().mean(),
           "DK_MAE_proj20plus": e_dk[big].abs().mean(), "min_MAE": (m["proj_min"] - m["a_min"]).abs().mean(),
           "DK_bias": e_dk.mean(),
           "missed_scratches": int(((m["a_status"] == "dnp_cd") & (m["proj_min"] >= 15)).sum())}
    m = fill_benchmarks(m, date)
    for src in ("DFF", "FP", "BC"):
        for site, actual in (("DK", "a_DK"), ("FD", "a_FD")):
            col = f"bench_{src}_{site}"
            if col not in m.columns or not (m[col] > 0).any():
                continue
            b = m[m[col] > 0]
            row[f"{src}_{site}_players"] = len(b)
            row[f"ours_{site}_MAE_vs_{src}"] = (b[f"fp_{site}"] - b[actual]).abs().mean()
            row[f"{src}_{site}_MAE"] = (b[col] - b[actual]).abs().mean()
    m.assign(err_DK=e_dk).to_csv(folder / "graded.csv", index=False)
    return row


def fill_benchmarks(m, date):
    """Projection files saved before a source was added can still be graded against it,
    for sources that keep past dates (DFF, FantasyPros). BlueCollar cannot be back-filled."""
    import benchmark
    from names import Matcher
    mt = Matcher(zip(m["player_id"], m["name"], m["team"]))
    for site in ("DK", "FD"):
        col = f"bench_DFF_{site}"
        if col not in m.columns or not (m[col] > 0).any():
            df = benchmark.dff(date, site)
            if not df.empty:
                ids = [mt.match(n, t) for n, t in zip(df["name"], df["team"])]
                vals = {i: v for i, v in zip(ids, df["ppg_proj"]) if i is not None}
                m[col] = m["player_id"].map(vals)
    if "bench_FP_DK" not in m.columns or not (m["bench_FP_DK"] > 0).any():
        fp = benchmark.fantasypros(date, CURRENT_SEASON)
        if not fp.empty:
            fp = fp.drop_duplicates("player_id").set_index("player_id")
            m["bench_FP_DK"] = m["player_id"].map(fp["fp_DK"])
            m["bench_FP_FD"] = m["player_id"].map(fp["fp_FD"])
    return m


def regrade(date, use_starters=True):
    """Re-project a past night as if the final inactive list (and, by default, confirmed
    starters) had been known, with that day's saved Vegas lines; score vs benchmarks."""
    import backtest
    import benchmark
    import live
    import model
    from names import Matcher

    pg, tg = backtest.load_all([CURRENT_SEASON])
    games = live.saved_games(date)
    lookup = {}
    for g in games.itertuples():
        if g.total and g.spread_home is not None:
            lookup[(pd.Timestamp(date), g.home)] = {"total": g.total, "spread": g.spread_home}
            lookup[(pd.Timestamp(date), g.away)] = {"total": g.total, "spread": -g.spread_home}
    if not lookup:   # no run saved that day: use the lines DailyFantasyFuel lists per game
        for site in ("DK", "FD"):
            df = benchmark.dff(date, site)
            if not df.empty:
                for t, sp, ou in df.dropna(subset=["spread", "ou"])[["team", "spread", "ou"]].drop_duplicates("team").values:
                    lookup.setdefault((pd.Timestamp(date), t), {"total": ou, "spread": sp})
        games = pd.DataFrame({"home": sorted({k[1] for k in lookup})[::2]}) if lookup else games
    r = backtest.run([CURRENT_SEASON], pg=pg, tg=tg, quiet=True, vegas_lookup=lookup,
                     use_starters=use_starters, only_dates=[date])
    if r.empty:
        return None
    r["a_fp_DK"], r["a_fp_FD"] = r["a_fp_DK"], r["a_fp_FD"]
    mt = Matcher(zip(r["player_id"], r["name"], r["team"]))

    def by_name(df, col):
        ids = [mt.match(n, t) for n, t in zip(df["name"], df["team"])]
        return {i: v for i, v in zip(ids, df[col]) if i is not None}

    notes = {}
    for site in ("DK", "FD"):
        d = benchmark.dff(date, site)
        r[f"bench_DFF_{site}"] = r["player_id"].map(by_name(d, "ppg_proj")) if not d.empty else np.nan
        bc, stamp = benchmark.bluecollar_saved(date, site)
        r[f"bench_BC_{site}"] = r["player_id"].map(by_name(bc, "proj")) if not bc.empty else np.nan
        if stamp:
            notes["BC"] = f"BlueCollar snapshot saved {stamp[:2]}:{stamp[2:]} ET"
    fp = benchmark.fantasypros(date, CURRENT_SEASON)
    if not fp.empty:
        fp = fp.drop_duplicates("player_id").set_index("player_id")
        r["bench_FP_DK"], r["bench_FP_FD"] = r["player_id"].map(fp["fp_DK"]), r["player_id"].map(fp["fp_FD"])
    import props
    r, pstamp = props.attach(r, date)
    if pstamp:
        notes["props"] = f"props pulled {pstamp[:2]}:{pstamp[2:]} ET"
    r.to_csv(OUTPUT / f"{date:%Y-%m-%d}" / "regraded_injuries_known.csv", index=False)
    return r, notes, len(lookup) // 2, len(games)


def print_regrade(date, r, notes, n_lines, n_games):
    e = r["fp_DK"] - r["a_fp_DK"]
    print(f"\n=== {date} re-graded as if injuries were known "
          f"(real inactives + starters, Vegas lines for {n_lines}/{n_games} games) ===")
    print(f"all {len(r)} players who dressed: DK MAE {e.abs().mean():.2f} | FD MAE "
          f"{(r['fp_FD'] - r['a_fp_FD']).abs().mean():.2f} | minutes MAE "
          f"{(r['proj_min'] - r['a_min']).abs().mean():.2f} | bias {e.mean():+.2f}")
    for src, label in (("BC", "BlueCollar"), ("DFF", "DailyFantasyFuel"), ("FP", "FantasyPros")):
        for site in ("DK", "FD"):
            col = f"bench_{src}_{site}"
            if col not in r.columns or not (r[col] > 0).any():
                continue
            b = r[r[col] > 0]
            ours = (b[f"fp_{site}"] - b[f"a_fp_{site}"]).abs()
            theirs = (b[col] - b[f"a_fp_{site}"]).abs()
            print(f"  vs {label:16s} {site}: same {len(b):3d} players | ours {ours.mean():.2f}  "
                  f"theirs {theirs.mean():.2f} | we were closer on {100 * (ours < theirs).mean():.0f}%")
    for v in notes.values():
        print(f"  ({v})")
    print_props_test(r)


def print_props_test(r):
    """TEST: would moving our pts/reb/ast/3s toward the prop-implied averages have helped?"""
    import props
    if "prop_pts" not in r.columns:
        return
    b = r.dropna(subset=["prop_pts"])
    if b.empty:
        return
    print(f"  props test ({len(b)} players with a points prop; 24+ min = {int((b['a_min'] >= 24).sum())}):")
    for s in ("pts", "reb", "ast", "fg3m"):
        c = b.dropna(subset=[f"prop_{s}"])
        print(f"    {s:4s} {len(c):3d}: ours miss {(c[s] - c[f'a_{s}']).abs().mean():.2f}  "
              f"props miss {(c[f'prop_{s}'] - c[f'a_{s}']).abs().mean():.2f}")
    for w in (0, 0.25, 0.5, 0.75, 1.0):
        e = props.blend_dk(b, w) - b["a_fp_DK"]
        big = b["a_min"] >= 24
        print(f"    DK with {int(w * 100):3d}% toward props: all {e.abs().mean():.2f} | 24+ min {e[big].abs().mean():.2f}")


def main(dates=None):
    sdv.refresh([CURRENT_SEASON])
    pg = load_player_games([CURRENT_SEASON])
    if not dates:
        dates = sorted(dt.date.fromisoformat(p.name) for p in OUTPUT.glob("20*-*-*") if p.is_dir())
    log_f = OUTPUT / "accuracy_log.csv"
    log = pd.read_csv(log_f) if log_f.exists() else pd.DataFrame()
    rows = []
    for d in dates:
        r = grade(d, pg)
        if r:
            rows.append(r)
            print(f"{r['date']}: {r['players']} players | DK MAE {r['DK_MAE']:.2f} "
                  f"(proj 20+ min {r['DK_MAE_proj20plus']:.2f}) | min MAE {r['min_MAE']:.2f}")
            names = {"DFF": "DailyFantasyFuel", "FP": "FantasyPros", "BC": "BlueCollar"}
            for src, label in names.items():
                for site in ("DK", "FD"):
                    if r.get(f"{src}_{site}_players"):
                        print(f"    vs {label:16s} {site}: same {r[f'{src}_{site}_players']:3d} players | "
                              f"ours {r[f'ours_{site}_MAE_vs_{src}']:.2f}  theirs {r[f'{src}_{site}_MAE']:.2f}")
            out = regrade(d)
            if out:
                print_regrade(d, *out)
        else:
            print(f"{d}: no box scores yet")
    if rows:
        new = pd.DataFrame(rows)
        log = pd.concat([log[~log.get("date", pd.Series(dtype=str)).isin(new["date"])] if len(log) else log, new])
        log.sort_values("date").round(3).to_csv(log_f, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    main([dt.date.fromisoformat(a.date)] if a.date else None)
