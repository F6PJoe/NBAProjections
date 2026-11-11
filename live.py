"""Fetch tonight's facts. Each fetch saves a time-stamped copy under cache/live/<date>/
so later backtests can replay real Vegas lines, injury reports and lineups.

Sources (all free, no login):
  ESPN scoreboard      games, tip times, spread and over/under (DraftKings line)
  NBA injury report    official statuses (PDF, every 15 minutes)
  NBA daily lineups    expected / confirmed starters
  ESPN injuries        news notes (minutes restrictions)
  DraftKings           salaries + positions (public draftables feed)
  Yahoo DFS            salaries + positions (public players feed)
  FanDuel              no public feed: drop FD's "players list" CSV in data/fanduel/
"""
import datetime as dt
import json
import re
from zoneinfo import ZoneInfo

import pandas as pd
import requests

import injury_pdf
from config import DATA, LIVE, team_code
from names import flip_last_first

ET = ZoneInfo("America/New_York")
S = requests.Session()
S.headers["Accept"] = "application/json"


def _folder(date):
    f = LIVE / f"{date:%Y-%m-%d}"
    f.mkdir(parents=True, exist_ok=True)
    return f


def _save(date, name, obj):
    stamp = dt.datetime.now(ET).strftime("%H%M")
    p = _folder(date) / f"{name}_{stamp}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _get_json(url, **kw):
    r = S.get(url, timeout=45, **kw)
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ games ----

def games(date):
    d = _get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                  params={"dates": f"{date:%Y%m%d}"})
    _save(date, "espn_scoreboard", d)
    g = parse_scoreboard(d)
    _remember_lines(date, g)
    return g


def _remember_lines(date, g):
    """lines.csv: the last pre-game total/spread seen for each game today."""
    f = _folder(date) / "lines.csv"
    old = pd.read_csv(f, dtype={"espn_id": str}) if f.exists() else pd.DataFrame()
    new = g.loc[g["total"].notna(), ["espn_id", "home", "away", "total", "spread_home", "line_text"]]
    new = new.assign(espn_id=new["espn_id"].astype(str))
    if new.empty:
        return
    keep = old[~old["espn_id"].isin(new["espn_id"])] if len(old) else old
    pd.concat([keep, new]).to_csv(f, index=False)


def saved_games(date):
    """Games + lines from the last scoreboard saved that day (for re-grading past nights)."""
    files = sorted(_folder(date).glob("espn_scoreboard_*.json"))
    if not files:
        return pd.DataFrame()
    g = parse_scoreboard(json.loads(files[-1].read_text(encoding="utf-8")))
    f = _folder(date) / "lines.csv"
    if f.exists() and len(g):
        ln = pd.read_csv(f, dtype={"espn_id": str}).set_index("espn_id")
        ids = g["espn_id"].astype(str)
        miss = g["total"].isna() & ids.isin(ln.index)
        g.loc[miss, "total"] = ids[miss].map(ln["total"])
        g.loc[miss, "spread_home"] = ids[miss].map(ln["spread_home"])
    return g


def parse_scoreboard(d):
    out = []
    for e in d.get("events", []):
        c = e["competitions"][0]
        teams = {t["homeAway"]: team_code(t["team"]["abbreviation"]) for t in c["competitors"]}
        odds = (c.get("odds") or [{}])[0]
        total = odds.get("overUnder")
        spread_home = None
        det = odds.get("details") or ""
        m = re.match(r"([A-Z]+)\s+(-?\d+(\.\d+)?)", det)
        if m:
            fav, line = team_code(m.group(1)), float(m.group(2))
            spread_home = line if fav == teams["home"] else -line
        elif det.upper() in ("EVEN", "PK", "PICK"):
            spread_home = 0.0
        tip = pd.Timestamp(e["date"]).tz_convert(ET)
        out.append({"espn_id": e["id"], "home": teams["home"], "away": teams["away"],
                    "tip": tip, "status": c["status"]["type"]["name"],
                    "total": total, "spread_home": spread_home, "line_text": det})
    return pd.DataFrame(out)


# --------------------------------------------------------------- injuries ----

def injury_report(date):
    """Returns (player rows, report time, set of teams that have NOT filed yet)."""
    path, ts = injury_pdf.latest(_folder(date) / "injury_pdf")
    empty = pd.DataFrame(columns=["team", "player", "status", "reason"])
    if path is None:
        return empty, None, set()
    df = pd.DataFrame(injury_pdf.parse(path))
    if df.empty:
        return empty, ts, set()
    df = df[df["game_date"] == f"{date:%m/%d/%Y}"].copy()
    df["team"] = df["team"].map(injury_pdf.team_tricode)
    not_filed = set(df.loc[df["status"] == "NOT_SUBMITTED", "team"].dropna())
    df = df[df["status"] != "NOT_SUBMITTED"].copy()
    df["player"] = df["player"].map(lambda n: flip_last_first(injury_pdf.split_camel(n)))
    return df, ts, not_filed


LONG_TERM = re.compile(r"weeks|months|surgery|indefinitely|season|re-evaluated|reevaluated|"
                       r"no timetable|torn|tear|fracture", re.I)


def espn_statuses(date):
    """ESPN injury statuses, used only for teams whose official report isn't in yet.
    'Out' counts as out only when it is long-term or 2+ days old; a fresh one-game 'Out'
    (e.g. sat last night) or 'Day-To-Day' becomes Questionable for tonight."""
    files = sorted(_folder(date).glob("espn_injuries_*.json"))
    d = json.loads(files[-1].read_text(encoding="utf-8")) if files else _get_json(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    out = []
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            st = i.get("status") or ""
            when = pd.Timestamp(i["date"]).tz_convert(ET).date() if i.get("date") else None
            note = f"{i.get('shortComment') or ''} {i.get('longComment') or ''}"
            if st == "Out" and (LONG_TERM.search(note) or (when and (date - when).days >= 2)):
                status = "Out"
            elif st in ("Out", "Day-To-Day"):
                status = "Questionable"
            else:
                continue
            out.append({"player": i["athlete"]["displayName"], "status": status, "espn_status": st,
                        "note_date": str(when), "note": (i.get("shortComment") or "")[:200]})
    return pd.DataFrame(out)


RESTRICT = re.compile(r"minutes? (restriction|limit|cap)|limited minutes|restricted minutes|"
                      r"on a (strict )?minutes|ramp(ing)? up|pitch count|limited role in his return",
                      re.I)
MIN_NUMBER = re.compile(r"(?:around|about|roughly|to|under|limited to|capped at)\s+(\d{2})\s+minutes", re.I)


def espn_notes(date):
    """Recent ESPN player notes that mention a minutes restriction."""
    d = _get_json("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries")
    _save(date, "espn_injuries", d)
    out = []
    since = pd.Timestamp(date, tz="UTC") - pd.Timedelta(days=3)
    for t in d.get("injuries", []):
        for i in t.get("injuries", []):
            txt = f"{i.get('shortComment') or ''} {i.get('longComment') or ''}"
            when = pd.Timestamp(i.get("date")) if i.get("date") else None
            if when is not None and when.tzinfo is None:
                when = when.tz_localize("UTC")
            if RESTRICT.search(txt) and (when is None or when >= since):
                num = MIN_NUMBER.search(txt)
                out.append({"player": i["athlete"]["displayName"],
                            "espn_status": i.get("status"),
                            "note_date": str(when)[:10],
                            "minutes_cap": float(num.group(1)) if num else None,
                            "note": (i.get("shortComment") or "")[:300]})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- lineups ----

def lineups(date):
    try:
        d = _get_json(f"https://stats.nba.com/js/data/leaders/00_daily_lineups_{date:%Y%m%d}.json")
    except Exception as e:  # noqa: BLE001
        print(f"  [lineups] not available ({e.__class__.__name__})")
        return pd.DataFrame(columns=["player_id", "team", "lineup_status"])
    _save(date, "nba_lineups", d)
    rows = []
    for g in d.get("games", []):
        for side in ("homeTeam", "awayTeam"):
            t = g[side]
            for p in t.get("players", []):
                rows.append({"player_id": int(p["personId"]), "team": team_code(t["teamAbbreviation"]),
                             "name": p.get("playerName"), "lineup_status": p.get("lineupStatus"),
                             "roster_status": p.get("rosterStatus"), "lineup_pos": p.get("position")})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- DFS sites ----

def draftkings(date):
    lobby = _get_json("https://www.draftkings.com/lobby/getcontests", params={"sport": "NBA"})
    groups = [g for g in lobby.get("DraftGroups", [])
              if g.get("GameTypeId") == 70 and str(g.get("StartDateEst", "")).startswith(f"{date:%Y-%m-%d}")]
    if not groups:
        return pd.DataFrame(), None
    main = max(groups, key=lambda g: (g.get("ContestStartTimeSuffix") is None, g.get("GameCount", 0)))
    rows = {}
    for g in sorted(groups, key=lambda g: g is not main):  # main slate first wins
        d = _get_json(f"https://api.draftkings.com/draftgroups/v1/draftgroups/{g['DraftGroupId']}/draftables")
        if g is main:
            _save(date, "dk_draftables_main", d)
        for x in d.get("draftables", []):
            key = x["playerId"]
            if key in rows:
                continue
            opp = x["competition"]["name"].replace(" ", "").split("@")
            team = team_code(x["teamAbbreviation"])
            opp = [team_code(o) for o in opp if team_code(o) != team]
            rows[key] = {"dk_id": x["playerId"], "name": x["displayName"], "team": team,
                         "opp": opp[0] if opp else None, "pos": x["position"],
                         "salary": x["salary"], "dk_status": x.get("status"),
                         "main_slate": g is main}
    return pd.DataFrame(rows.values()), main["DraftGroupId"]


def yahoo(date):
    d = _get_json("https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nba")
    _save(date, "yahoo_players", d)
    rows = []
    for p in d.get("players", {}).get("result", []):
        tip = pd.to_datetime(p["gameStartTime"], format="%a, %d %b %Y %H:%M:%S %z").tz_convert(ET)
        if tip.date() != date:
            continue
        team = team_code(p["team"])
        home, away = team_code(p["homeTeam"]), team_code(p["awayTeam"])
        rows.append({"yahoo_id": p["playerCode"], "name": p["name"], "team": team,
                     "opp": away if team == home else home, "pos": p["position"],
                     "salary": p["salary"], "yahoo_status": p.get("status")})
    return pd.DataFrame(rows)


def fanduel(date):
    """FanDuel's lineup page has a 'Download players list' button. Save that CSV into
    data/fanduel/ (any file name). The newest file whose games are today is used."""
    folder = DATA / "fanduel"
    folder.mkdir(parents=True, exist_ok=True)
    best = None
    for f in sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001
            continue
        if {"First Name", "Last Name", "Salary", "Team"} <= set(df.columns):
            best = df
            break
    if best is None:
        return pd.DataFrame()
    df = best
    name = (df["First Name"].fillna("") + " " + df["Last Name"].fillna("")).str.strip()
    out = pd.DataFrame({"fd_id": df.get("Id"), "name": name, "team": df["Team"].map(team_code),
                        "opp": df.get("Opponent", pd.Series([None] * len(df))).map(team_code),
                        "pos": df["Position"], "salary": df["Salary"],
                        "fd_status": df.get("Injury Indicator")})
    return out
