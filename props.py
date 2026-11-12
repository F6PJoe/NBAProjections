"""Player prop lines from SportsGameOdds (free plan) - TEST ONLY, not a model input yet.

  python props.py                  # pull tonight's NBA props (if the pull rules allow)
  python props.py --date 2026-11-11

Free plan = 2,500 "objects" a month; one game returned = one object, however many
props it carries. A night of 8 games costs ~8, so one or two pulls a day is cheap.
Their terms are fine for evaluation; ask them before props drive published numbers.

Each prop (fair line + fair over odds, bookmaker margin removed) is turned into an
implied average: the mean of a normal(mu, c*sqrt(mu)) that puts that much chance
above the line (c = the model's sd_coef). grade.py compares those with our numbers.
"""
import argparse
import datetime as dt
import gzip
import json
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

from config import LIVE, PARAMS, secret, team_code

URL = "https://api.sportsgameodds.com/v2/events"
ET = ZoneInfo("America/New_York")
# their statID -> our stat name
STAT_MAP = {"points": "pts", "rebounds": "reb", "assists": "ast", "threePointersMade": "fg3m",
            "steals": "stl", "blocks": "blk", "turnovers": "tov"}
SD_COEF = {**PARAMS["sd_coef"], "fg3m": 1.1, "tov": 1.0}
FIRST_PULL_HOUR = 12  # books post most player props by midday; earlier pulls cost the same and return little
REUSE_MIN = 180       # before the first tip: refresh every 3 hours (noon, 3 PM)
NEAR_TIP_MIN = 75     # + one pull in the 75 minutes before each tip group ...
GROUP_GAP_MIN = 120   # ... where games within 2 hours of the first tip of a group count as one group
# normal 7 PM night: noon, 3 PM, ~5:55 PM (+ ~8:55 PM if there are 10 PM games) = ~3-4 pulls,
# ~10 games each -> ~1,000 objects a month (free plan: 2,500)


def _american_to_prob(o):
    o = float(o)
    return -o / (-o + 100) if o < 0 else 100 / (o + 100)


def implied_mean(line, p_over, c):
    """Mean of normal(mu, c*sqrt(mu)) with P(X > line) = p_over."""
    p_over = min(max(p_over, 0.03), 0.97)
    f = lambda mu: (1 - norm.cdf((line - mu) / (c * np.sqrt(mu)))) - p_over
    try:
        return brentq(f, 0.05, max(3 * line + 10, 10))
    except ValueError:
        return np.nan


def fetch(date):
    key = secret("SPORTSGAMEODDS_API_KEY")
    if not key:
        raise SystemExit("SPORTSGAMEODDS_API_KEY missing from .env")
    start = dt.datetime.combine(date, dt.time(6), ET)          # 6 AM ET .. 6 AM ET next day
    params = {"leagueID": "NBA", "oddsAvailable": "true", "limit": 50,
              # games already under way are skipped: their lines are live in-game lines (and cost objects)
              "startsAfter": max(start, dt.datetime.now(ET)).astimezone(dt.timezone.utc).isoformat(),
              "startsBefore": (start + dt.timedelta(days=1)).astimezone(dt.timezone.utc).isoformat()}
    events, cursor = [], None
    for _ in range(5):
        if cursor:
            params["cursor"] = cursor
        r = requests.get(URL, params=params, headers={"x-api-key": key}, timeout=60)
        r.raise_for_status()
        j = r.json()
        events += j.get("data", [])
        cursor = j.get("nextCursor")
        if not cursor or not j.get("data"):
            break
        time.sleep(7)                                               # 10 requests/minute
    # keep only games that start on this ET date
    keep = [e for e in events
            if dt.datetime.fromisoformat(e["status"]["startsAt"].replace("Z", "+00:00")).astimezone(ET).date() == date]
    return keep


def tidy(events):
    rows = []
    for ev in events:
        st = ev.get("status", {})
        if st.get("started") or st.get("live"):        # in-game lines already know the score - never use
            continue
        teams = {side: team_code(ev["teams"][side]["names"]["short"]) for side in ("home", "away")}
        by_id = {ev["teams"][s]["teamID"]: t for s, t in teams.items()}
        players = ev.get("players", {})
        for oid, o in ev.get("odds", {}).items():
            if (o.get("betTypeID") != "ou" or o.get("sideID") != "over" or o.get("periodID") != "game"
                    or o.get("started")):
                continue
            stat = STAT_MAP.get(o.get("statID"))
            pid = o.get("playerID")
            if not stat or not pid or not o.get("fairOddsAvailable"):
                continue
            try:
                line, p = float(o["fairOverUnder"]), _american_to_prob(o["fairOdds"])
            except (KeyError, TypeError, ValueError):
                continue
            books = o.get("byBookmaker", {})
            pl = players.get(pid, {})
            rows.append({"event": ev["eventID"], "tip": ev["status"]["startsAt"],
                         "home": teams["home"], "away": teams["away"],
                         "team": by_id.get(pl.get("teamID")), "name": pl.get("name", pid),
                         "sgo_player": pid, "stat": stat, "line": line, "p_over": round(p, 4),
                         "book_line": o.get("bookOverUnder"), "n_books": len(books),
                         "mean": implied_mean(line, p, SD_COEF[stat])})
    return pd.DataFrame(rows)


def _due(folder, date, now, tips):
    if date != now.date():
        return False
    import live
    files = sorted(folder.glob("props_[0-9]*.csv"))
    last = None
    if files:
        hhmm = files[-1].stem[-4:]
        last = pd.Timestamp(dt.datetime.combine(date, dt.time(int(hhmm[:2]), int(hhmm[2:])), ET))
    now = pd.Timestamp(now)
    if tips is None or not len(tips):
        return now.hour >= FIRST_PULL_HOUR and (last is None or (now - last).total_seconds() >= REUSE_MIN * 60)
    first = live.tip_groups(tips, GROUP_GAP_MIN)[0]
    if (now.hour >= FIRST_PULL_HOUR and now < first
            and (last is None or (now - last).total_seconds() >= REUSE_MIN * 60)):
        return True                    # midday refreshes (an early tip still gets its near-tip pull below)
    return live.pull_due_before_tip(tips, now, last, NEAR_TIP_MIN, GROUP_GAP_MIN)


def pull(date=None, force=False, tips=None):
    """Tonight's props (one row per player x stat), pulled only when due; saved under cache/live/<date>/."""
    now = dt.datetime.now(ET)
    date = date or now.date()
    folder = LIVE / f"{date:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    if not force and not _due(folder, date, now, tips):
        return saved(date)[0]
    events = fetch(date)
    stamp = f"{now:%H%M}"
    with gzip.open(folder / f"props_raw_{stamp}.json.gz", "wt") as f:
        json.dump(events, f)
    df = tidy(events)
    df.to_csv(folder / f"props_{stamp}.csv", index=False)
    return df


def saved(date, before=None):
    """All of a date's pulls combined, latest line per player x stat (games drop out of the
    feed once they tip, so the last pull alone misses early games). Optionally only pulls before HHMM."""
    files = sorted((LIVE / f"{date:%Y-%m-%d}").glob("props_[0-9]*.csv"))
    if before:
        files = [f for f in files if f.stem[-4:] < before] or files[:1]
    if not files:
        return pd.DataFrame(), None
    df = pd.concat([pd.read_csv(f, dtype={"pulled": str}).assign(pulled=f.stem[-4:]) for f in files])
    # each player's line from the last pull BEFORE his own game tipped
    tip_hhmm = pd.to_datetime(df["tip"], utc=True).dt.tz_convert(ET).dt.strftime("%H%M")
    df = df[df["pulled"] < tip_hhmm]
    df = df.sort_values("pulled").drop_duplicates(["sgo_player", "stat"], keep="last")
    return df, files[-1].stem[-4:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date) if a.date else None
    df = pull(d, a.force)
    if df.empty:
        print("no props found")
    else:
        print(f"{df['event'].nunique()} games, {df['sgo_player'].nunique()} players, {len(df)} player-stat lines")
        print(df.groupby("stat").size().to_string())


def attach(df, date, before=None):
    """Add prop_<stat> (implied averages) and prop_line_<stat> columns to a table of our players."""
    from names import Matcher
    p, stamp = saved(date, before)
    if p.empty:
        return df, None
    mt = Matcher(zip(df["player_id"], df["name"], df["team"]))
    p = p.assign(player_id=[mt.match(n, t) for n, t in zip(p["name"], p["team"])]).dropna(subset=["player_id"])
    df = df.copy()
    for stat, g in p.groupby("stat"):
        g = g.drop_duplicates("player_id").set_index("player_id")
        df[f"prop_{stat}"] = df["player_id"].map(g["mean"])
        df[f"prop_line_{stat}"] = df["player_id"].map(g["line"])
    return df, stamp


def blend_dk(df, w, stats=("pts", "reb", "ast", "fg3m"), ours="", site="DK"):
    """Our DK points with each stat moved w of the way toward the prop's implied average
    (only where a prop exists). Double-double bonus left as projected."""
    from config import SCORING
    fp = df[f"fp_{site}"].copy()
    for s in stats:
        c = f"prop_{s}"
        if c in df.columns:
            d = (df[c] - df[f"{ours}{s}"]).fillna(0)
            fp += w * SCORING[site][s] * d
    return fp
