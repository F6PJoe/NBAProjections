"""Re-run a past live run with TODAY's code, using only the inputs that run saved
(scoreboard/lines, injury PDF, ESPN injuries, NBA lineups) and history before that date.

  python research/replay_day.py 2026-11-08 1501

Writes output/<date>/replay/projections_replay_<stamp>.csv (never mixed with the real runs).
"""
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

import benchmark  # noqa: E402
import history  # noqa: E402
import injury_pdf  # noqa: E402
import live  # noqa: E402
import project  # noqa: E402
import sdv  # noqa: E402

date = dt.date.fromisoformat(sys.argv[1])
stamp = sys.argv[2]
folder = ROOT / "cache" / "live" / f"{date:%Y-%m-%d}"


def saved(name):
    return json.loads((folder / f"{name}_{stamp}.json").read_text(encoding="utf-8"))


def fake_get_json(url, **kw):
    if "scoreboard" in url:
        return saved("espn_scoreboard")
    if "injuries" in url:
        return saved("espn_injuries")
    if "daily_lineups" in url:
        return saved("nba_lineups")
    raise RuntimeError(f"replay: no saved copy of {url}")


pdfs = sorted((folder / "injury_pdf").glob("*.pdf"))
live._get_json = fake_get_json
live._save = lambda *a, **k: None
injury_pdf.latest = lambda *a, **k: (pdfs[-1], pd.Timestamp(date))
live.draftkings = lambda d: (pd.DataFrame(), None)
live.yahoo = lambda d: pd.DataFrame()
live.fanduel = lambda d: pd.DataFrame()
benchmark.attach = lambda allp, *a, **k: allp
sdv.refresh = lambda *a, **k: None

_load = history.load_player_games
project.load_player_games = lambda seasons: (lambda pg: pg[pg["game_date"] < pd.Timestamp(date)])(_load(seasons))

out = ROOT / "output" / f"{date:%Y-%m-%d}" / "replay"
out.mkdir(parents=True, exist_ok=True)
project.OUTPUT = out / "_runs"
allp = project.main(date, quiet=True)
allp.to_csv(out / f"projections_replay_{stamp}.csv", index=False)
print(f"replayed {len(allp)} players -> {out / f'projections_replay_{stamp}.csv'}")
