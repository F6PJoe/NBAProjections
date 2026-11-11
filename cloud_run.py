"""One scheduled run (GitHub Actions calls this every 15 minutes; also fine to run by hand).

  python cloud_run.py            # run if one is due
  python cloud_run.py --force    # run now regardless

Decides whether a run is worth it, so the 15-minute trigger can fire all day:
  - no games today, or every game has tipped        -> skip
  - more than 4 hours before the first tip          -> at most one run an hour
  - otherwise                                       -> every trigger (skips if a run finished < 10 min ago)
Then trims today's raw input files to the newest of each kind so the data repo stays small
(projections, lines.csv, props and BlueCollar snapshots are all kept).
"""
import argparse
import datetime as dt
import re
import sys
from zoneinfo import ZoneInfo

import requests

import live
from config import LIVE, OUTPUT

ET = ZoneInfo("America/New_York")
EARLY_HOURS = 4
EARLY_EVERY_MIN = 55
DEDUPE_MIN = 10
KEEP_ALL = re.compile(r"^(props_\d{4}\.csv|lines\.csv)$")


def todays_tips(date):
    r = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                     params={"dates": f"{date:%Y%m%d}"}, timeout=45)
    r.raise_for_status()
    return live.parse_scoreboard(r.json())


def minutes_since_last_run(date, now):
    files = sorted((OUTPUT / f"{date:%Y-%m-%d}").glob("projections_all_*.csv"))
    if not files:
        return None
    hhmm = files[-1].stem[-4:]
    last = dt.datetime.combine(date, dt.time(int(hhmm[:2]), int(hhmm[2:])), ET)
    return (now - last).total_seconds() / 60


def due(now):
    date = now.date()
    g = todays_tips(date)
    if g.empty:
        return False, "no NBA games today"
    first, last = g["tip"].min(), g["tip"].max()
    if now >= last:
        return False, f"every game has tipped (last tip {last:%I:%M %p})"
    since = minutes_since_last_run(date, now)
    if now < first - dt.timedelta(hours=EARLY_HOURS):
        if since is not None and since < EARLY_EVERY_MIN:
            return False, f"early in the day, last run {since:.0f} min ago (first tip {first:%I:%M %p})"
    elif since is not None and since < DEDUPE_MIN:
        return False, f"last run only {since:.0f} min ago"
    return True, f"{len(g)} games, first tip {first:%I:%M %p} ET"


def prune(date):
    """Keep only the newest raw file of each kind for today (raw inputs are ~2.5 MB a run)."""
    folder = LIVE / f"{date:%Y-%m-%d}"
    if not folder.exists():
        return
    groups = {}
    for f in folder.rglob("*"):
        if not f.is_file() or KEEP_ALL.match(f.name):
            continue
        kind = (f.parent, re.sub(r"_\d{4}(\.json(\.gz)?|\.csv|\.pdf)$", "", f.name))
        groups.setdefault(kind, []).append(f)
    for files in groups.values():
        for f in sorted(files)[:-1]:
            f.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    now = dt.datetime.now(ET)
    ok, why = due(now)
    print(f"{now:%Y-%m-%d %I:%M %p} ET: {'RUN' if ok or a.force else 'skip'} - {why}")
    if not (ok or a.force):
        return 0
    import project
    project.main(now.date())
    prune(now.date())
    return 0


if __name__ == "__main__":
    sys.exit(main())
