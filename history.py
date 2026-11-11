"""Turn the SportsDataverse (hoopR) downloads into one clean player-game table.

One row per player per regular-season game he was listed in, with:
  status  'played'  - logged minutes
          'dnp_cd'  - dressed but coach kept him out (counts as 0 minutes of his role)
          'out'     - did not dress / not with team (injury, rest, suspension)
Players who were hurt for a long stretch are often missing from the box score
entirely; for the model that is the same as 'out'.
"""
import numpy as np
import pandas as pd

from config import SDV, team_code

BOX_COLS = {
    "points": "pts", "three_pointers_made": "fg3m", "rebounds_total": "reb",
    "assists": "ast", "steals": "stl", "blocks": "blk", "turnovers": "tov",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "rebounds_offensive": "oreb", "rebounds_defensive": "dreb",
}


def _minutes(x):
    """'33:15' -> 33.25 ; '' / None -> 0."""
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    if ":" in s:
        m, sec = s.split(":")[:2]
        try:
            return int(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    if s.startswith("PT"):  # ISO 8601 'PT33M15.00S'
        m = s[2:].split("M")[0]
        sec = s.split("M")[1].rstrip("S") if "M" in s else "0"
        return float(m or 0) + float(sec or 0) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_schedule(seasons):
    frames = []
    for yr in seasons:
        f = SDV / f"nba_schedule_{yr}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    s = pd.concat(frames, ignore_index=True)
    s["game_date"] = pd.to_datetime(s["game_date"]).dt.normalize()
    s["home"] = s["home_team_abbreviation"].map(team_code)
    s["away"] = s["away_team_abbreviation"].map(team_code)
    return s[["game_id", "season", "season_type", "game_date", "home", "away",
              "home_pts", "away_pts"]].drop_duplicates("game_id")


def load_rosters(season):
    f = SDV / f"rosters_{season}.parquet"
    if not f.exists():
        return pd.DataFrame(columns=["player_id", "player", "team", "position"])
    r = pd.read_parquet(f)
    tri = load_team_ids()
    r["team"] = r["team_id"].map(tri)
    return r.rename(columns={"player_id": "player_id"})[["player_id", "player", "team", "position"]]


_TEAM_IDS = None


def load_team_ids():
    """NBA team_id -> tricode, from any box score file we have."""
    global _TEAM_IDS
    if _TEAM_IDS is None:
        m = {}
        for f in sorted(SDV.glob("player_boxscores_*.parquet"))[-2:]:
            b = pd.read_parquet(f, columns=["team_id", "team_tricode"])
            m.update(dict(zip(b.team_id.astype(int), b.team_tricode.map(team_code))))
        _TEAM_IDS = m
    return _TEAM_IDS


ESPN_SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def _espn_fill(b, sched, seasons, days_back=5):
    """Finished games from the last few days that the daily SportsDataverse file doesn't
    have yet, taken from ESPN's box scores (posted right after each game). Returned in the
    SportsDataverse column layout so the rest of the pipeline can't tell the difference."""
    import json
    import requests
    from config import CACHE, CURRENT_SEASON
    from names import Matcher
    if CURRENT_SEASON not in seasons:
        return pd.DataFrame()
    have = set(b["game_id"].astype(str))
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    s = sched[(sched["season_type"] == "regular-season") & (sched["game_date"] < today)
              & (sched["game_date"] >= today - pd.Timedelta(days=days_back))]
    s = s[~s["game_id"].astype(str).isin(have)]
    if s.empty:
        return pd.DataFrame()
    # NBA ids for ESPN names: everyone seen in box scores or on a roster
    people = pd.DataFrame({"pid": b["person_id"].astype(int), "team": b["team_tricode"].map(team_code),
                           "name": (b["first_name"].fillna("") + " " + b["family_name"].fillna("")).str.strip()})
    r = load_rosters(CURRENT_SEASON)
    people = pd.concat([people, pd.DataFrame({"pid": r["player_id"], "team": r["team"], "name": r["player"]})])
    people = people.dropna().drop_duplicates(["pid", "team"])
    mt = Matcher(zip(people["pid"], people["name"], people["team"]))
    tri_to_id = {v: k for k, v in load_team_ids().items()}
    folder = CACHE / "espn_box"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for day, games in s.groupby("game_date"):
        try:
            sb = requests.get(ESPN_SB, params={"dates": f"{day:%Y%m%d}"}, timeout=40).json()
        except Exception:  # noqa: BLE001
            continue
        for e in sb.get("events", []):
            comp = e["competitions"][0]
            if not comp["status"]["type"].get("completed"):
                continue
            side = {t["homeAway"]: team_code(t["team"]["abbreviation"]) for t in comp["competitors"]}
            g = games[(games["home"] == side["home"]) & (games["away"] == side["away"])]
            if g.empty:
                continue
            gid = str(g["game_id"].iloc[0])
            f = folder / f"{e['id']}.json"
            if not f.exists():
                try:
                    f.write_text(requests.get(ESPN_SUM, params={"event": e["id"]}, timeout=40).text,
                                 encoding="utf-8")
                except Exception:  # noqa: BLE001
                    continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for tb in d.get("boxscore", {}).get("players", []):
                team = team_code(tb["team"]["abbreviation"])
                st = tb["statistics"][0]
                lab = st["labels"]
                for a in st["athletes"]:
                    name = a["athlete"]["displayName"]
                    pid = mt.match(name, team)
                    if pid is None:
                        pid = -int(a["athlete"]["id"])          # unknown to NBA files yet
                    v = dict(zip(lab, a.get("stats") or []))

                    def num(k, part=None):
                        x = v.get(k, "0") or "0"
                        if part is not None:
                            x = x.split("-")[part] if "-" in x else "0"
                        try:
                            return float(x)
                        except ValueError:
                            return 0.0
                    first, _, last = name.partition(" ")
                    dnp = a.get("didNotPlay") or not v
                    reason = (a.get("reason") or "").upper()
                    rows.append({
                        "game_id": gid, "person_id": pid, "first_name": first, "family_name": last,
                        "team_tricode": team, "team_id": tri_to_id.get(team, 0),
                        "position": "S" if a.get("starter") else "",
                        "comment": ("DNP - Coach's Decision" if "COACH" in reason else f"DND - {reason}")
                        if dnp else "",
                        "minutes": "" if dnp else str(num("MIN")),
                        "points": num("PTS"), "three_pointers_made": num("3PT", 0),
                        "rebounds_total": num("REB"), "assists": num("AST"), "steals": num("STL"),
                        "blocks": num("BLK"), "turnovers": num("TO"),
                        "field_goals_made": num("FG", 0), "field_goals_attempted": num("FG", 1),
                        "free_throws_made": num("FT", 0), "free_throws_attempted": num("FT", 1),
                        "rebounds_offensive": num("OREB"), "rebounds_defensive": num("DREB"),
                        "season_type_id": "2", "season": CURRENT_SEASON,
                    })
    if rows:
        print(f"  [espn] filled {len({r['game_id'] for r in rows})} recent games not in the daily file yet")
    return pd.DataFrame(rows)


def load_player_games(seasons):
    """Clean regular-season player-game table for the given SDV season labels."""
    sched = load_schedule(seasons)
    frames = []
    for yr in seasons:
        f = SDV / f"player_boxscores_{yr}.parquet"
        if not f.exists():
            continue
        b = pd.read_parquet(f)
        b = b[b["season_type_id"].astype(str) == "2"]
        frames.append(b)
    b = pd.concat(frames, ignore_index=True)

    b = pd.concat([b, _espn_fill(b, sched, seasons)], ignore_index=True)
    out = pd.DataFrame({
        "game_id": b["game_id"].astype(str),
        "player_id": b["person_id"].astype(int),
        "name": (b["first_name"].fillna("") + " " + b["family_name"].fillna("")).str.strip(),
        "team": b["team_tricode"].map(team_code),
        "starter": b["position"].fillna("").str.len() > 0,
        "comment": b["comment"].fillna(""),
        "min": b["minutes"].map(_minutes),
    })
    for src, dst in BOX_COLS.items():
        out[dst] = pd.to_numeric(b[src], errors="coerce").fillna(0.0).values

    c = out["comment"].str.upper()
    out["status"] = np.where(out["min"] > 0, "played",
                     np.where(c.str.contains("COACH"), "dnp_cd", "out"))
    out.loc[(out["status"] == "dnp_cd") & (out["min"] > 0), "status"] = "played"

    out = out.merge(sched, on="game_id", how="inner")
    out["opp"] = np.where(out["team"] == out["home"], out["away"], out["home"])
    out["is_home"] = out["team"] == out["home"]
    tp = np.where(out["is_home"], out["home_pts"], out["away_pts"])
    op = np.where(out["is_home"], out["away_pts"], out["home_pts"])
    out["team_pts"], out["opp_pts"] = tp, op
    # schedule scores can lag a day: fall back to the box-score totals
    box_pts = out.groupby(["game_id", "team"])["pts"].transform("sum")
    opp_box = out.groupby("game_id")["pts"].transform("sum") - box_pts
    out["team_pts"] = out["team_pts"].fillna(box_pts)
    out["opp_pts"] = out["opp_pts"].fillna(opp_box)
    out["margin"] = out["team_pts"] - out["opp_pts"]
    out = out.drop(columns=["home", "away", "home_pts", "away_pts", "season_type"])
    out = out.sort_values(["game_date", "game_id", "team", "starter"],
                          ascending=[True, True, True, False]).reset_index(drop=True)
    out["left_hurt"] = _left_hurt(out)
    return out


def _left_hurt(out):
    """Played well under his usual minutes AND missed the team's next game -> likely got hurt.
    Those minutes say nothing about his role, so the minutes model skips them."""
    tg = out.drop_duplicates(["game_id", "team"])[["game_id", "team", "game_date"]]
    tg = tg.sort_values("game_date")
    tg["seq"] = tg.groupby("team").cumcount()
    o = out.merge(tg[["game_id", "team", "seq"]], on=["game_id", "team"], how="left")
    o = o.sort_values(["player_id", "game_date"])
    played_min = o["min"].where(o["status"] == "played")
    o["usual"] = (played_min.groupby(o["player_id"])
                  .transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()))
    avail = o["status"].isin(["played", "dnp_cd"])
    nxt_seq = o["seq"].where(avail).groupby(o["player_id"]).shift(-1)
    nxt_team = o["team"].where(avail).groupby(o["player_id"]).shift(-1)
    missed_next = (nxt_seq.isna() | (nxt_seq - o["seq"] > 1) | (nxt_team != o["team"]))
    # last game of the data set: unknown, not hurt
    last_seq = o.groupby("team")["seq"].transform("max")
    missed_next &= o["seq"] < last_seq
    flag = (o["status"] == "played") & (o["min"] < 0.6 * o["usual"]) & missed_next
    return flag.reindex(out.index).fillna(False).values


def team_games(pg):
    """One row per team per game with team totals (from the player rows)."""
    played = pg[pg["status"] == "played"]
    agg = played.groupby(["game_id", "team"], as_index=False)[
        ["min", "pts", "fg3m", "reb", "ast", "stl", "blk", "tov", "fga", "fta", "oreb"]].sum()
    meta = pg.drop_duplicates(["game_id", "team"])[
        ["game_id", "team", "opp", "season", "game_date", "is_home", "team_pts", "opp_pts", "margin"]]
    tg = meta.merge(agg, on=["game_id", "team"], how="left")
    tg["game_total"] = tg["team_pts"] + tg["opp_pts"]
    return tg.sort_values(["team", "game_date"]).reset_index(drop=True)


def position_index(pos):
    """Roster position text -> a 1 (point guard) .. 5 (center) scale."""
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return 3.0
    p = str(pos).upper().replace(" ", "")
    table = {"PG": 1.0, "G": 1.5, "SG": 2.0, "G-F": 2.5, "PG/SG": 1.5, "SG/SF": 2.5,
             "F-G": 2.75, "SF": 3.0, "F": 3.5, "SF/PF": 3.5, "PF": 4.0, "F-C": 4.25,
             "PF/C": 4.5, "C-F": 4.5, "C": 5.0, "PG/SF": 2.0, "SG/PF": 3.0}
    if p in table:
        return table[p]
    parts = [table.get(x) for x in p.replace("/", "-").split("-") if x in table]
    return float(np.mean(parts)) if parts else 3.0
