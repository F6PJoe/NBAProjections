"""Download SportsDataverse (hoopR) NBA files - the NBA version of nflverse.

Files are re-downloaded only when older than their time-to-live, so this is
cheap to call on every run.
"""
import time

import requests

from config import SDV, SDV_URL, CURRENT_SEASON

# release tag, file stem, seasons it is needed for
DATASETS = [
    ("nba_stats_player_boxscores", "player_boxscores_{yr}"),
    ("nba_stats_schedules", "nba_schedule_{yr}"),
    ("nba_stats_rosters", "rosters_{yr}"),
]

TTL_CURRENT = 3 * 3600          # current season: refresh every 3 hours
TTL_PAST = 30 * 24 * 3600       # finished seasons barely change


def fetch(tag, stem, yr, force=False):
    SDV.mkdir(parents=True, exist_ok=True)
    name = stem.format(yr=yr) + ".parquet"
    dest = SDV / name
    ttl = TTL_CURRENT if yr >= CURRENT_SEASON else TTL_PAST
    if dest.exists() and not force and time.time() - dest.stat().st_mtime < ttl:
        return dest
    url = f"{SDV_URL}/{tag}/{name}"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print(f"  [sdv] {name}: HTTP {r.status_code} (keeping old copy)" if dest.exists()
              else f"  [sdv] {name}: HTTP {r.status_code}")
        return dest if dest.exists() else None
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    print(f"  [sdv] {name}: {len(r.content) / 1e6:.1f} MB")
    return dest


def refresh(seasons=None, force=False):
    seasons = seasons or [CURRENT_SEASON - 1, CURRENT_SEASON]
    for yr in seasons:
        for tag, stem in DATASETS:
            fetch(tag, stem, yr, force)


if __name__ == "__main__":
    import sys
    yrs = [int(a) for a in sys.argv[1:]] or None
    refresh(yrs)
