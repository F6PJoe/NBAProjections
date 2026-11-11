"""Tonight's projections, start to finish. Safe to run as often as you like.

  python project.py                 # today (Eastern time)
  python project.py --date 2026-11-07

Writes output/<date>/ :
  projections_all_<HHMM>.csv   every player, every stat, minutes, status, notes
  DK.csv / FD.csv / Yahoo.csv  site tables (Player, Proj, Salary, Value, Pos, Team, Opp, ...)
and copies the newest set to output/latest/.

Hand overrides: data/overrides.csv  (player,team,action,value,note)
  action = out | play | minutes | cap     e.g.  LeBron James,LAL,cap,28,minutes limit
Questionable players are projected as if they play, with Status "Q".
Doubtful players are treated as out.
"""
import argparse
import datetime as dt
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import benchmark
import live
import model
import sdv
from config import CURRENT_SEASON, DATA, OUTPUT, PARAMS, SITE_TEAM_CODES, team_code
from history import load_player_games, load_rosters, position_index, team_games
from names import Matcher

ET = ZoneInfo("America/New_York")
OUT_STATUSES = {"Out", "Doubtful"}


def load_overrides():
    f = DATA / "overrides.csv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("player,team,action,value,note\n", encoding="utf-8")
    o = pd.read_csv(f, dtype=str).fillna("")
    o["team"] = o["team"].map(lambda t: team_code(t) if t else "")
    return o


def build_rosters(pg, season):
    """Current team for every player: today's roster file, plus anyone who has played
    for a team this season and does not show up on another roster."""
    r = load_rosters(season)
    r = r.dropna(subset=["team"])
    r["player_id"] = r["player_id"].astype(int)
    on_file = set(r["player_id"])
    last = (pg[(pg["season"] == season) & pg["status"].isin(["played", "dnp_cd"])]
            .sort_values("game_date").groupby("player_id").tail(1))
    extra = last[~last["player_id"].isin(on_file)]
    extra = pd.DataFrame({"player_id": extra["player_id"], "player": extra["name"],
                          "team": extra["team"], "position": None})
    return pd.concat([r[["player_id", "player", "team", "position"]], extra], ignore_index=True)


def starter_flag(pid, team, starters, lineup_state, start_rate):
    """True/False from the lineup feed. An EXPECTED lineup that leaves out a player who
    usually starts (and isn't ruled out) is not trusted to bench him: None = use his history."""
    if team not in starters:
        return None
    if pid in starters[team]:
        return True
    if lineup_state.get(team) != "Confirmed" and start_rate.get(pid, 0.0) >= 0.6:
        return None
    return False


def main(date, params=PARAMS, quiet=False):
    say = (lambda *a: None) if quiet else print
    season = CURRENT_SEASON
    say(f"NBA projections for {date} (season {season - 1}-{str(season)[2:]})")
    say("refreshing history ...")
    sdv.refresh([season - 1, season])
    pg = load_player_games([season - 1, season])
    rosters = build_rosters(pg, season)
    pos = {}
    for yr in (season - 1, season):
        rr = load_rosters(yr)
        pos.update(dict(zip(rr["player_id"].astype(int), rr["position"].map(position_index))))
    pg["pos"] = pg["player_id"].map(pos).fillna(3.0)
    model.set_positions(dict(zip(pg["player_id"], pg["pos"])))
    tg = team_games(pg)
    hist = model.History(pg, tg)
    pos_means = model.position_means(pg[pg["season"] == season - 1])

    say("fetching tonight's games, injuries, lineups, salaries ...")
    games = live.games(date)
    if games.empty:
        say("no games today")
        return None
    inj, inj_time, not_filed = live.injury_report(date)
    notes = live.espn_notes(date)
    espn_st = live.espn_statuses(date)
    lineups = live.lineups(date)
    dk, dk_group = live.draftkings(date)
    yh = live.yahoo(date)
    fd = live.fanduel(date)
    overrides = load_overrides()

    matcher = Matcher(zip(rosters["player_id"], rosters["player"], rosters["team"]))
    status, reason, unmatched = {}, {}, []
    for r in inj.itertuples():
        pid = matcher.match(r.player, r.team)
        if pid is None:
            unmatched.append(f"{r.player} ({r.team}, {r.status})")
            continue
        status[pid], reason[pid] = r.status, r.reason
    status_src = {pid: "official" for pid in status}
    # teams that haven't filed today's report: ESPN fills the silence
    team_of = dict(zip(rosters["player_id"], rosters["team"]))
    for r in espn_st.itertuples():
        pid = matcher.match(r.player)
        if pid is None or team_of.get(pid) not in not_filed or pid in status:
            continue
        status[pid], reason[pid] = r.status, f"ESPN {r.espn_status} ({r.note_date}): {r.note}"
        status_src[pid] = "ESPN (team report not filed yet)"
    # recent start rate, so an unconfirmed lineup can't bench a regular starter
    recent = (pg[(pg["season"] == season) & pg["status"].isin(["played", "dnp_cd"])]
              .sort_values("game_date").groupby("player_id").tail(8))
    start_rate = recent.groupby("player_id")["starter"].mean().to_dict()
    caps = {}
    for r in notes.itertuples():
        pid = matcher.match(r.player)
        if pid is not None and r.minutes_cap:
            caps[pid] = (r.minutes_cap, r.note)
    restriction_note = {matcher.match(r.player): r.note for r in notes.itertuples()}

    starters = {}
    lineup_state = {}
    for team, g in lineups.groupby("team"):
        # Expected lineups list just the 5 starters; once CONFIRMED, NBA.com lists the whole
        # roster and only starters carry a position. Active/Inactive is then the official list.
        five = g[g["lineup_pos"].fillna("").str.len() > 0]
        if len(five) == 0 and len(g) <= 5:
            five = g
        if len(five) >= 5:
            starters[team] = set(five["player_id"])
            lineup_state[team] = "Confirmed" if (five["lineup_status"] == "Confirmed").all() else "Expected"
        if lineup_state.get(team) == "Confirmed":
            for r in g[g["roster_status"] == "Inactive"].itertuples():
                if status.get(r.player_id) not in ("Out",):
                    status[r.player_id], reason[r.player_id] = "Out", "NBA inactive list"
                    status_src[r.player_id] = "NBA confirmed lineup (inactive)"

    ov = {}
    for r in overrides.itertuples():
        if not r.player:
            continue
        pid = matcher.match(r.player, r.team or None)
        if pid is None:
            unmatched.append(f"override: {r.player}")
            continue
        ov[pid] = (r.action.strip().lower(), r.value, r.note)

    recent_played = {}
    for team, d in pg[(pg["season"] == season) & (pg["status"] == "played")].groupby("team"):
        gids = d.sort_values("game_date")["game_id"].unique()[-10:]
        recent_played[team] = set(d.loc[d["game_id"].isin(gids), "player_id"])

    day = model.day_num(date)
    rows = []
    for g in games.itertuples():
        for team, opp, is_home in ((g.home, g.away, True), (g.away, g.home, False)):
            vegas = None
            if g.total and g.spread_home is not None:
                vegas = {"total": g.total, "spread": g.spread_home if is_home else -g.spread_home}
            members = rosters[rosters["team"] == team]
            roster, out_ids, info = [], set(), {}
            for m in members.itertuples():
                pid = int(m.player_id)
                st = status.get(pid, "")
                act, val, note = ov.get(pid, ("", "", ""))
                is_out = (st in OUT_STATUSES and act != "play") or act == "out"
                info[pid] = {"status": st, "status_source": status_src.get(pid, ""),
                             "reason": reason.get(pid, ""), "override": act and f"{act} {val}".strip(),
                             "override_note": note, "restriction_note": restriction_note.get(pid, "")}
                if is_out:
                    out_ids.add(pid)
                    continue
                entry = {"player_id": pid, "name": m.player,
                         "pos": pos.get(pid, position_index(m.position)),
                         "starter": starter_flag(pid, team, starters, lineup_state, start_rate),
                         "fixed_min": None, "cap_min": None}
                if act == "minutes" and val:
                    entry["fixed_min"] = float(val)
                elif act == "cap" and val:
                    entry["cap_min"] = float(val)
                elif pid in caps:
                    entry["cap_min"] = caps[pid][0]
                roster.append(entry)
            out_ids |= recent_played.get(team, set()) - {e["player_id"] for e in roster}
            res = model.project_team(hist, team, opp, is_home, day, season, roster, out_ids,
                                     pos_means, params, vegas)
            for r in res:
                r.update(info.get(r["player_id"], {}))
                r["tip"] = g.tip
                r["lineup"] = lineup_state.get(team, "none")
                r["vegas_line"] = g.line_text
            rows.extend(res)
            for pid in out_ids & set(info):
                rows.append({"player_id": pid, "name": members.loc[members["player_id"] == pid, "player"].iloc[0],
                             "team": team, "opp": opp, "proj_min": 0.0, "tip": g.tip,
                             **info[pid]})
    allp = pd.DataFrame(rows)
    allp["Status"] = np.select([allp["status"].isin(OUT_STATUSES) | (allp.get("override") == "out"),
                                allp["status"] == "Questionable", allp["status"] == "Probable"],
                               ["OUT", "Q", "P"], "")
    for c in ("fp_DK", "fp_FD", "fp_Yahoo", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m"):
        allp[c] = allp[c].fillna(0.0)

    # ---- site tables ----
    stamp = dt.datetime.now(ET)
    folder = OUTPUT / f"{date:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    keep = ["player_id", "name", "team", "opp", "Status", "proj_min", "starter_tonight", "pts", "reb",
            "ast", "stl", "blk", "tov", "fg3m", "p_dd", "p_td", "fp_DK", "fp_FD", "fp_Yahoo",
            "team_total", "game_total", "env_source", "lineup", "status", "reason",
            "status_source", "restriction_note", "override", "override_note", "games",
            "missed_in_row", "tip"]
    allp = allp.reindex(columns=keep).sort_values("fp_DK", ascending=False)
    allp = benchmark.attach(allp, matcher, date, season, tips=games["tip"])   # comparison only
    try:                                                    # prop lines: saved for grading only (TEST)
        import props
        pr = props.pull(date, tips=games["tip"])            # pulls only when due (free plan quota)
        say(f"props (test only): {pr['sgo_player'].nunique() if len(pr) else 0} players")
    except Exception as e:                                  # never let the test break a run
        say(f"props (test only): skipped - {e}")
    allp.to_csv(folder / f"projections_all_{stamp:%H%M}.csv", index=False)

    unmatched_sites = {}
    sites = {}
    fd_has_salary = fd is not None and not fd.empty
    if not fd_has_salary and dk is not None and not dk.empty:
        # no FanDuel salary file yet: list FD projections for the same player pool, salary blank
        fd = dk.drop(columns=["dk_id", "dk_status", "main_slate"]).assign(salary=np.nan)
    for site, df, fp, per in (("DK", dk, "fp_DK", 1000), ("FD", fd, "fp_FD", 1000), ("Yahoo", yh, "fp_Yahoo", 1)):
        if df is None or df.empty:
            continue
        df = df.copy()
        df["player_id"] = [matcher.match(n, t) for n, t in zip(df["name"], df["team"])]
        unmatched_sites[site] = df.loc[df["player_id"].isna(), "name"].tolist()
        m = df.merge(allp[["player_id", fp, "proj_min", "Status", "lineup"]], on="player_id", how="left")
        m["Proj"] = m[fp].fillna(0.0).round(2)
        m["Value"] = (m["Proj"] / (m["salary"] / per)).round(2)
        t = pd.DataFrame({
            "Player": m["name"], "Proj": m["Proj"], "Salary": m["salary"], "Value": m["Value"],
            "Pos": m["pos"], "Team": m["team"].map(lambda x: SITE_TEAM_CODES.get(x, x)),
            "Opp": m["opp"].map(lambda x: SITE_TEAM_CODES.get(x, x) if isinstance(x, str) else x),
            "Min": m["proj_min"].round(1), "Status": m["Status"].fillna(""),
        })
        if site == "DK":
            t["Slate"] = np.where(m["main_slate"], "Main", "Other")
        t = t[t["Proj"] > 0].sort_values("Proj", ascending=False)
        t.to_csv(folder / f"{site}.csv", index=False)
        sites[site] = t

    latest = OUTPUT / "latest"
    latest.mkdir(exist_ok=True)
    allp.to_csv(latest / "projections_all.csv", index=False)
    for site, t in sites.items():
        t.to_csv(latest / f"{site}.csv", index=False)

    # ---- run summary ----
    say(f"\ninjury report: {inj_time:%I:%M %p ET}" if inj_time else "\ninjury report: NOT FOUND")
    tonight = set(games["home"]) | set(games["away"])
    if not_filed & tonight:
        say("  not filed yet (ESPN statuses used): " + ", ".join(sorted(not_filed & tonight)))
    say("lineups: " + ", ".join(f"{k} {v}" for k, v in sorted(lineup_state.items())) if lineup_state
        else "lineups: none posted")
    say(f"Vegas lines: {games['total'].notna().sum()} of {len(games)} games")
    say(f"site tables: " + ", ".join(f"{s} {len(t)} players" for s, t in sites.items())
        + ("" if fd_has_salary else "  (FD salaries blank: no players-list CSV in data/fanduel/)"))
    if unmatched:
        say("injury/override names not matched: " + "; ".join(unmatched))
    for s, u in unmatched_sites.items():
        if u:
            say(f"{s} names not matched ({len(u)}): " + ", ".join(u[:15]))
    say(f"\nwrote {folder}")
    return allp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: today, Eastern)")
    a = ap.parse_args()
    d = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(ET).date()
    main(d)
