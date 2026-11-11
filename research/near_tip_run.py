"""A 'just before tip' run done late: live injury report + confirmed lineups, but the
PRE-GAME Vegas lines from an earlier saved scoreboard (lines move once games start).
Saved to output/<date>/replay/ so grading never mistakes it for the real pre-lock run.

  python research/near_tip_run.py 2026-11-09 1724      # date, stamp of the scoreboard to reuse
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

import live  # noqa: E402
import project  # noqa: E402

date = dt.date.fromisoformat(sys.argv[1])
stamp = sys.argv[2]
sb = json.loads((ROOT / "cache" / "live" / f"{date:%Y-%m-%d}" / f"espn_scoreboard_{stamp}.json")
                .read_text(encoding="utf-8"))
live.games = lambda d: live.parse_scoreboard(sb)

now = pd.Timestamp.now(tz="America/New_York").strftime("%H%M")
out = ROOT / "output" / f"{date:%Y-%m-%d}" / "replay"
out.mkdir(parents=True, exist_ok=True)
project.OUTPUT = out / "_runs"
allp = project.main(date)
allp.to_csv(out / f"projections_neartip_{now}.csv", index=False)
print(f"saved {out / f'projections_neartip_{now}.csv'}")
