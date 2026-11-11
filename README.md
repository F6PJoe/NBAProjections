# NBA DFS Projections (Fantasy Six Pack)

Our own nightly NBA projections for DraftKings, FanDuel and Yahoo, built to replace
the BlueCollar DFS feed. Minutes, injuries and lineups are handled automatically;
a hand override is only needed for news the feeds don't carry.

## Run it

```
python project.py                  # today's slate (Eastern date)
python project.py --date 2026-11-07
```
or double-click `Run projections.bat`. Safe to run as often as you like — every
run pulls the newest injury report, lineups, lines and salaries.

Output goes to `output/<date>/` and the newest copy to `output/latest/`:

| File | What's in it |
|---|---|
| `DK.csv`, `Yahoo.csv`, `FD.csv` | Site tables — Player, Proj, Salary, Value, Pos, Team, Opp, Min, Status (same first 7 columns as the current Google Sheet tabs) |
| `projections_all.csv` | Every player: minutes, each stat, double/triple-double odds, DK/FD/Yahoo points, team total, injury status, notes |

## Where the data comes from (all free)

| Need | Source |
|---|---|
| Box-score history (2 seasons) | SportsDataverse / hoopR — the NBA version of nflverse, updated daily |
| Games, spread, over/under | ESPN scoreboard (DraftKings line) |
| Injury statuses | NBA official injury report PDF (re-posted every 15 min) |
| Starting lineups | NBA.com daily lineups (Expected → Confirmed ~30 min before tip) |
| Minutes-restriction news | ESPN player notes (flag; a cap is applied only when a number is given) |
| DraftKings salaries | DK public draftables feed |
| Yahoo salaries | Yahoo DFS public players feed |
| FanDuel salaries | **No public feed** — save FD's "Download players list" CSV into `data/fanduel/` |

Every run saves a time-stamped copy of these inputs under `cache/live/<date>/`, so
later backtests can replay real lines and injury reports.

## How it projects

1. **Minutes**: recent games the player was available for (newer games count more,
   coach's-decision DNPs count as 0, games he left hurt are skipped), mixed with last
   season early on. Tonight's role (starting vs bench) shifts it.
2. **Team minutes forced to 240 (+ overtime)**: minutes freed by players who are out
   go mostly to teammates at the same position; when a player returns, minutes come
   back mostly from the bench. Players back from a long absence start a bit lower.
3. **Per-minute production**: this season + last season, pulled toward the position
   average when the sample is small.
4. **Game environment**: Vegas total and spread set expected pace and team points;
   team totals are kept realistic so a missing star's shots and assists get spread
   around the active players.
5. **Scoring**: DK (incl. double/triple-double bonus odds), FD, Yahoo.

Questionable players are projected as if they play (Status `Q`), the way most DFS
sites show them; re-run once they're ruled in/out. Doubtful = out.

**Teams that haven't filed today's report yet** (the PDF says NOT YET SUBMITTED — common
early in the day and on back-to-backs): ESPN's injury status fills in for that team only.
A long-term ESPN "Out" (weeks, surgery, 2+ days old) counts as out; a fresh one-game "Out"
or "Day-To-Day" becomes Q. The run prints which teams used ESPN, and `status_source` in
`projections_all.csv` says where each status came from. Once a team files, the official
report wins.

**Expected vs confirmed lineups**: a confirmed lineup is trusted completely. An *expected*
lineup can promote a player to starter, but it can't bench someone who usually starts and
isn't ruled out — his own history decides until the lineup is confirmed.

## Hand overrides — `data/overrides.csv`

```
player,team,action,value,note
LeBron James,PHI,cap,28,minutes limit per beat writer
Jarrett Allen,CLE,out,,ruled out after report
Peyton Watson,CLE,play,,Q but warming up
Someone,BOS,minutes,30,
```
`out` / `play` override the injury report, `minutes` sets minutes exactly, `cap` limits them.
Delete the line when it no longer applies.

`data/name_aliases.csv` (`source_name,nba_name`) fixes a name the matcher misses;
each run prints any unmatched names.

## Publishing to the site

`python publish_sheet.py` writes the site tables into the same Google Sheet the
BlueCollar script uses, on TEST tabs (`DK NBA DFS (new)` etc.) so the live site is
untouched. `--live` writes the real tabs + the "NBA Update Time" stamp. Needs the
service-account key in `GCP_SHEETS_KEY_B64` or `GCP_SHEETS_KEY_PATH`.

## Private keys — `.env`

Keys live in `.env` in this folder (never committed — it's in `.gitignore`):
```
FANTASYPROS_API_KEY=...     # filled in (same key as FF_ADP/ff_rankings)
BLUECOLLAR_API_KEY=...      # benchmark only, until you cancel. Sent as "Authorization: ApiKey <key>".
                            # 200 requests/day per key across everything that uses it; we reuse a
                            # fetch for 30 min, so at most ~4 calls/hour from here
GCP_SHEETS_KEY_PATH=...     # Google service-account JSON, for publish_sheet.py later
```

## Benchmarks (comparison only — the model never reads them)

Every run saves outside projections next to ours in `projections_all.csv`:
`bench_DFF_DK/FD` (DailyFantasyFuel, free), `bench_FP_DK/FD` (FantasyPros daily stat
lines, needs the key), `bench_BC_DK/FD` (BlueCollar, needs the key). A source that's
down just leaves blank columns. `grade.py` scores each one on the same players as ours.

Season to date (2026-27, walk-forward, same players only):

| vs | player-games | ours DK MAE | theirs |
|---|---|---|---|
| DailyFantasyFuel (`research/compare_dff.py`) | 1,909 | 7.45 | 7.23 |
| FantasyPros (`research/compare_fp.py`) | 1,767 | 7.72 | 8.17 |

FantasyPros' stats are whole numbers, so part of their error is rounding.

## Grading (how did last night go?)

```
python grade.py                   # every past date in output/
```
Two grades per night:
1. **As run** — the last projections saved before the first tip-off, against the box scores
   (`output/<date>/graded.csv`, one line per night in `output/accuracy_log.csv`).
2. **Injuries known** — re-projects the night as if the final inactive list and starters
   had been known, with that day's saved Vegas lines (`regraded_injuries_known.csv`). This
   is the fair test of the model itself when runs can't be timed right before lock.
Both are scored against BlueCollar / DailyFantasyFuel / FantasyPros on the same players.
BlueCollar only serves current slates, so its grade uses the last pull saved that day.

## Backtest

```
python backtest.py                # walk-forward over 2025-26 and 2026-27 to date
python backtest.py --no-starters  # as if run before lineups are known
python research/tune.py           # re-tune settings in config.py
```
Settings live in `config.py` → `PARAMS`.

## Automation (GitHub Actions)

- **Code:** this repo (public; keys are GitHub secrets, never in the code).
- **Results:** private repo `F6PJoe/NBAProjections-data` (it holds other sites' numbers that can't be republished).
- `projections.yml` fires every 15 min; `cloud_run.py` decides if a run is due (hourly early in the day,
  every 15 min in the 4 hours before the first tip, stops after the last tip) and trims raw inputs.
- `grade.yml` grades last night every morning -> `output/<date>/grade_report.txt` in the data repo.
- Locally, `.env` has `NBA_DATA_DIR` pointing at `store/` (a clone of the data repo); `git -C store pull` to see cloud results.
- Secrets: `FANTASYPROS_API_KEY`, `BLUECOLLAR_API_KEY`, `SPORTSGAMEODDS_API_KEY`, `DATA_DEPLOY_KEY` (write deploy key on the data repo).
