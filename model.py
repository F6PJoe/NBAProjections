"""The projection model. Pure functions over history + tonight's facts; no network.

For each team in a game:
  1. Minutes baseline per active player (recent games he was available for,
     newer games count more; coach's-decision DNPs count as 0).
  2. Tonight's role (starting vs bench) shifts the baseline.
  3. Team minutes are forced to ~240. Minutes freed by players who are out go
     mostly to teammates at the same position; extra minutes are taken back
     mostly from the bench.
  4. Per-minute rates (this season + last season, pulled toward the position
     average when the sample is small).
  5. Game environment: expected pace and team points (Vegas when available)
     scale the stats, and team totals are kept realistic so a missing star's
     shots and assists get spread around.
"""
import math

import numpy as np
import pandas as pd

from config import PARAMS, SCORING, STATS

DAY0 = np.datetime64("2000-01-01", "D")


def day_num(d):
    return int((np.datetime64(pd.Timestamp(d).date(), "D") - DAY0).astype(int))


class History:
    """Fast as-of lookups over the player-game table."""

    def __init__(self, pg, tg):
        self.players = {}
        pg = pg.sort_values(["game_date", "game_id"])
        days = ((pg["game_date"].values.astype("datetime64[D]") - DAY0).astype(int))
        pg = pg.assign(_day=days)
        for pid, d in pg.groupby("player_id", sort=False):
            self.players[pid] = {
                "day": d["_day"].values,
                "season": d["season"].values,
                "team": d["team"].values,
                "status": d["status"].values,
                "min": d["min"].values.astype(float),
                "starter": d["starter"].values.astype(bool),
                "margin": d["margin"].values.astype(float),
                "left_hurt": d["left_hurt"].values.astype(bool),
                "x": d[STATS].values.astype(float),
                "game_id": d["game_id"].values,
                "name": d["name"].values[-1],
            }
        self.min_by_game = {(g, p): m for g, p, m in
                            zip(pg["game_id"].values, pg["player_id"].values, pg["min"].values)}
        # rotation size: players with 10+ minutes in each team game
        rot = pg[pg["min"] >= 10].groupby(["game_id", "team"]).size()
        self.teams = {}
        tg = tg.sort_values(["team", "game_date"])
        tdays = ((tg["game_date"].values.astype("datetime64[D]") - DAY0).astype(int))
        tg = tg.assign(_day=tdays)
        for team, d in tg.groupby("team", sort=False):
            self.teams[team] = {
                "day": d["_day"].values,
                "season": d["season"].values,
                "game_id": d["game_id"].values,
                "pts": d["team_pts"].values.astype(float),
                "opp_pts": d["opp_pts"].values.astype(float),
                "total": d["game_total"].values.astype(float),
                "x": d[STATS].values.astype(float),
                "rot": np.array([rot.get((g, team), np.nan) for g in d["game_id"].values], dtype=float),
            }
        # league per-game team averages by season, for early-season shrinkage
        self.league_team_avg = tg.groupby("season")[STATS + ["team_pts", "game_total"]].mean()
        # what each DEFENSE allowed, by the scorer's position group (guards / wings / bigs)
        pl = pg[(pg["status"] == "played") & (pg["min"] > 0)]
        if "pos" in pl.columns:
            grp = np.where(pl["pos"] <= 2.0, 0, np.where(pl["pos"] <= 3.6, 1, 2))
            dk = (pl["pts"] + 0.5 * pl["fg3m"] + 1.25 * pl["reb"] + 1.5 * pl["ast"] + 2 * pl["stl"]
                  + 2 * pl["blk"] - 0.5 * pl["tov"]).values
            d = pd.DataFrame({"def": pl["opp"].values, "day": pl["_day"].values, "season": pl["season"].values,
                              "grp": grp, "dk": dk, "min": pl["min"].values}).sort_values("day")
            self.defense = {t: {k: x[k].values for k in ("day", "season", "grp", "dk", "min")}
                            for t, x in d.groupby("def")}
            self.league_grp_rate = {(sn, g): x["dk"].sum() / x["min"].sum()
                                    for (sn, g), x in d.groupby(["season", "grp"])}
        else:
            self.defense, self.league_grp_rate = {}, {}

    def team_before(self, team, day):
        t = self.teams.get(team)
        if t is None:
            return None
        n = np.searchsorted(t["day"], day, side="left")
        return {k: v[:n] for k, v in t.items()}

    def player_before(self, pid, day):
        h = self.players.get(pid)
        if h is None:
            return None
        n = np.searchsorted(h["day"], day, side="left")
        if n == 0:
            return None
        return {k: (v[:n] if isinstance(v, np.ndarray) else v) for k, v in h.items()}


# ---------------------------------------------------------------- minutes ----

def minutes_baseline(h, team, season, P, game_w=None):
    """Return (baseline minutes, weighted start rate, games used). game_w: optional
    {game_id: weight} (e.g. games played while a teammate who is back tonight was out)."""
    if h is None:
        return None, 0.0, 0, {}
    avail = ((h["status"] == "played") | (h["status"] == "dnp_cd")) & ~h["left_hurt"]
    cur = avail & (h["season"] == season)
    idx = np.flatnonzero(cur)[::-1][: P["min_window"]]
    prior = None
    last = avail & (h["season"] == season - 1)
    if last.sum() >= 5:
        prior = float(h["min"][last][-40:].mean())
    elif P.get("older_seasons", 0) > 0:          # missed (nearly) all of last season: use the last full one
        older = avail & (h["season"] < season - 1) & (h["season"] >= season - 1 - P["older_seasons"])
        if older.sum() >= 5:
            prior = float(h["min"][older][-40:].mean()) * P.get("older_min_factor", 1.0)
            last = older
    if len(idx) == 0:
        if prior is None:
            return None, 0.0, 0, {}
        st = h["starter"][last][-20:].mean() if last.any() else 0.0
        nw = P.get("new_team_prior_w", 1.0)
        if nw < 1.0 and last.any() and not (h["team"][last][-20:] == team).any():
            # all his recent minutes came for another team: lean toward a new-player default
            prior = nw * prior + (1 - nw) * P["no_history_min"]
            st *= nw
        return prior, float(st), 0, {}
    k = np.arange(len(idx))
    w = 0.5 ** (k / P["min_half_life"])
    w = w * np.where(h["team"][idx] == team, 1.0, P["min_other_team_w"])
    w = w * np.where(np.abs(h["margin"][idx]) >= P["blowout_margin"], P["blowout_w"], 1.0)
    w = w * np.where(h["status"][idx] == "dnp_cd", P.get("dnp_cd_w", 1.0), 1.0)
    if game_w:
        w = w * np.array([game_w.get(g, 1.0) for g in h["game_id"][idx]])
    m = h["min"][idx]
    base = float((w * m).sum() / w.sum())
    if prior is not None:
        kp = P["min_prior_games"]
        base = float(((w * m).sum() + kp * prior) / (w.sum() + kp))
    jk, jw = int(P.get("role_jump_k", 2)), P.get("role_jump_w", 0.0)
    jumped = False
    if jw > 0 and len(idx) >= jk + 2:          # clear recent jump in minutes = role change (off by default)
        recent = float(m[:jk].mean())
        if recent - base >= P.get("role_jump_min", 6.0):
            base += jw * (recent - base)
            jumped = True
    dw = P.get("role_drop_w", 0.0)
    if dw > 0 and len(idx) >= jk + 2:          # clear recent drop (e.g. out of the rotation) (off by default)
        recent = float(m[:jk].mean())
        if base - recent >= P.get("role_drop_min", 6.0):
            base -= dw * (base - recent)
    st = h["starter"][idx]
    start_rate = float((w * st).sum() / w.sum())
    role = {}
    for flag in (True, False):
        sel = st == flag
        role[flag] = (float((w[sel] * m[sel]).sum() / w[sel].sum()), int(sel.sum())) if sel.any() else (None, 0)
    role["jumped"] = jumped
    return base, start_rate, len(idx), role


def filled_in_weights(hist, team, day, season, roster, P):
    """Games a player logged while a teammate who is ACTIVE tonight was out count less
    (e.g. Kobe Brown's 20-38 min while Obi Toppin sat). Off unless P['fill_in_scale'] > 0.
    Returns {player_id: {game_id: weight}}."""
    scale = P.get("fill_in_scale", 0.0)
    if scale <= 0:
        return {}
    ids = [r["player_id"] for r in roster]
    usual, games, base_all = {}, {}, {}
    for pid in ids:
        h = hist.player_before(pid, day)
        if h is None:
            continue
        b = minutes_baseline(h, team, season, P)[0]
        base_all[pid] = b or 0.0
        if b and b >= P.get("fill_in_min_usual", 15.0):
            usual[pid] = b
        cur = (h["season"] == season) & (h["team"] == team)
        games[pid] = h["game_id"][cur][-P["min_window"]:]
    out = {}
    floor = P.get("fill_in_floor", 0.2)
    for pid in ids:
        gw = {}
        for g in games.get(pid, []):
            absent = sum(u for j, u in usual.items() if j != pid and hist.min_by_game.get((g, j), 0.0) <= 0)
            if absent > 0:
                up = P.get("fill_in_only_up")
                if up is not None and hist.min_by_game.get((g, pid), 0.0) < base_all.get(pid, 0.0) + up:
                    continue
                gw[g] = max(floor, 1.0 - absent / scale)
        if gw:
            out[pid] = gw
    return out


def _missed_game(h, g):
    """True if this player was with the team for game g but did not play (out / rested)."""
    if h is None:
        return False
    idx = np.where(h["game_id"] == g)[0]
    return len(idx) > 0 and h["status"][idx[-1]] != "played"


def _recent_role(p, rows, P):
    """Did he earn real minutes lately with the regulars ahead of him available?
    Games where a regular who is active tonight sat out are skipped when rot_exempt_clean is on
    (those minutes were probably fill-in). Needs rot_recent_need of the last rot_recent_k clean
    games at rot_recent_min+."""
    h = p.get("h")
    if h is None or not len(h["min"]):
        return False
    need_min = P["rot_recent_min"]
    k = int(P.get("rot_recent_k", 1))
    need = int(P.get("rot_recent_need", 1))
    clean = P.get("rot_exempt_clean", False)
    regs = [q for q in rows if q is not p and (q["starter_tonight"] or q["r"] >= P.get("rot_reg_min", 20.0))]
    seen = hits = 0
    for i in range(len(h["min"]) - 1, -1, -1):
        if seen >= k:
            break
        if h["status"][i] == "out":
            continue                                   # he was hurt: not evidence either way
        g = h["game_id"][i]
        if clean and any(_missed_game(q.get("h"), g) for q in regs):
            continue                                   # a regular sat: fill-in minutes
        seen += 1
        hits += h["status"][i] == "played" and h["min"][i] >= need_min
    return hits >= need


def rotation_cut(rows, team_hist, season, P):
    """The coach's usual rotation size (players getting 10+ min, recent games). Healthy players
    ranked below it tonight usually DNP or mop up: shrink them (off by default)."""
    f = P.get("rot_cut_factor", 1.0)
    if f >= 1.0 or team_hist is None or "rot" not in team_hist:
        return
    sel = (team_hist["season"] == season) & ~np.isnan(team_hist["rot"])
    r = team_hist["rot"][sel][-P.get("rot_games", 10):]
    if len(r) < P.get("rot_min_games", 3):
        return
    size = int(round(float(np.mean(r)))) + int(P.get("rot_slack", 0))
    free = sorted((p for p in rows if p["fixed"] is None), key=lambda p: -p["r"])
    nfixed = sum(1 for p in rows if p["fixed"] is not None)
    recent = P.get("rot_recent_min", 0.0)        # >0: exempt players who earned real minutes lately
    soft = int(P.get("rot_soft_n", 0))            # >0: first `soft` players past the line get a partial cut
    k = 0
    for i, p in enumerate(free):
        if i + nfixed >= size and not p.get("starter_tonight"):
            if recent and _recent_role(p, rows, P):
                p["rot_exempt"] = True
                p["r_cut"] = p["r"] * f
                continue
            ff = f if k >= soft else 1 - (1 - f) * (k + 1) / (soft + 1)
            k += 1
            p["r"] *= ff
            p["rot_cut"] = ff <= f + 1e-9 or P.get("rot_soft_fill_cut", True)


def protect_regulars(rows, vac, vac_pos, P):
    """Minutes kept by rotation-exempt bench players come out of the other bench players,
    not the regulars: regulars keep what they'd get if the exempt players had been cut."""
    ex = [p for p in rows if p.get("rot_exempt")]
    if not ex:
        return
    ghost = [dict(p) for p in rows]
    for g in ghost:
        if g.get("rot_exempt"):
            g["r"], g["rot_cut"] = g["r_cut"], True
    allocate_minutes(ghost, vac, vac_pos, P)
    reg_min = P.get("rot_reg_min", 20.0)
    owed = 0.0
    for p, g in zip(rows, ghost):
        if p["fixed"] is None and not p.get("rot_exempt") and (p["starter_tonight"] or g["proj_min"] >= reg_min):
            owed += g["proj_min"] - p["proj_min"]
            p["proj_min"] = g["proj_min"]
            p["is_reg"] = True
    pool = [p for p in rows if p["fixed"] is None and not p.get("is_reg") and p["proj_min"] > 0]
    tot = sum(p["proj_min"] for p in pool)
    if tot > 0 and owed:
        f = max(0.0, 1 - owed / tot)
        for p in pool:
            p["proj_min"] *= f


def is_ironman(h, P):
    """Plays heavy minutes almost every night (recent close games): exempt from the high-minute trim."""
    rate = P.get("ironman_rate", 2.0)
    if rate > 1 or h is None:
        return False
    ok = (h["status"] == "played") & ~h["left_hurt"] & (np.abs(h["margin"]) < P["blowout_margin"])
    m = h["min"][ok][-P.get("ironman_games", 10):]
    return len(m) >= 5 and float((m >= P.get("ironman_min", 34.0)).mean()) >= rate


def games_missed_in_row(h, team_hist):
    """Team games since the player was last available (any team)."""
    if h is None or team_hist is None:
        return 0
    avail = (h["status"] == "played") | (h["status"] == "dnp_cd")
    if not avail.any():
        return 0
    last_day = h["day"][avail][-1]
    return int((team_hist["day"] > last_day).sum())


def vacated_minutes(hist, team_hist, out_ids, P, n_games=10):
    """Minutes per recent team game that players who are out tonight had been playing."""
    if team_hist is None or len(team_hist["game_id"]) == 0 or not out_ids:
        return {}
    gids = team_hist["game_id"][-n_games:][::-1]
    w = 0.5 ** (np.arange(len(gids)) / P["min_half_life"])
    out = {}
    for pid in out_ids:
        m = np.array([hist.min_by_game.get((g, pid), 0.0) for g in gids])
        v = float((w * m).sum() / w.sum())
        if v > 0.5:
            out[pid] = v
    return out


def allocate_minutes(players, vac, vac_pos, P):
    """players: list of dicts with r (raw minutes), pos, starter_tonight, fixed (or None)."""
    target = P["team_minutes"]
    fixed = sum(p["fixed"] for p in players if p["fixed"] is not None)
    free = [p for p in players if p["fixed"] is None]
    for p in players:
        p["proj_min"] = p["fixed"] if p["fixed"] is not None else p["r"]
    if not free:
        return players
    need = target - fixed - sum(p["r"] for p in free)

    if need > 0:
        tot_vac = sum(vac.values())
        for p in free:
            if tot_vac > 0:
                p["posw"] = sum(v * math.exp(-abs(p["pos"] - vac_pos[j]) / P["pos_tau"])
                                for j, v in vac.items()) / tot_vac
            else:
                p["posw"] = 1.0
            p["cap"] = min(P["max_minutes"], max(p["r"] * P["fill_cap_mult"], p["r"] + P["fill_cap_add"]))
        for stage in (0, 1):
            for _ in range(12):
                room = [p for p in free if p["cap"] - p["proj_min"] > 1e-6]
                if need <= 1e-6 or not room:
                    break
                wts = np.array([(p["r"] + (1.0 if stage else 0.25)) ** P["fill_power"] * p["posw"]
                                * (P.get("fill_cut_w", 1.0) if p.get("rot_cut") else 1.0)
                                for p in room])
                if wts.sum() <= 0:
                    break
                give = need * wts / wts.sum()
                used = 0.0
                for p, g in zip(room, give):
                    add = min(g, p["cap"] - p["proj_min"])
                    p["proj_min"] += add
                    used += add
                need -= used
            for p in free:  # second stage: anyone can go to the hard max
                p["cap"] = P["max_minutes"]
    elif need < 0:
        excess = -need
        for _ in range(12):
            room = [p for p in free if p["proj_min"] > 1e-6]
            if excess <= 1e-6 or not room:
                break
            wts = np.array([p["proj_min"] * (P["excess_starter_w"] if p["starter_tonight"] else 1.0)
                            for p in room])
            take = excess * wts / wts.sum()
            used = 0.0
            for p, t in zip(room, take):
                cut = min(t, p["proj_min"])
                p["proj_min"] -= cut
                used += cut
            excess -= used
    return players


# ------------------------------------------------------------------ rates ----

def position_means(pg_before):
    """Per-minute league averages for guards / wings / bigs."""
    d = pg_before[pg_before["status"] == "played"]
    grp = pd.cut(d["pos"], [0, 2.0, 3.6, 9], labels=["G", "W", "B"])
    s = d.groupby(grp, observed=True)[STATS + ["min"]].sum()
    return {g: (s.loc[g, STATS] / s.loc[g, "min"]).to_dict() for g in s.index}


def pos_group(pos):
    return "G" if pos <= 2.0 else ("W" if pos <= 3.6 else "B")


def per_minute_rates(h, season, pos, pos_means, P):
    mu = pos_means[pos_group(pos)]
    K = {k: v * P.get("rate_prior_scale", 1.0) for k, v in P["rate_prior_minutes"].items()}
    if h is None:
        return {s: mu[s] for s in STATS}, 0.0
    back = 1 + int(P.get("older_seasons", 0))   # seasons before last one (players who missed all of last season)
    played = (h["status"] == "played") & (h["season"] >= season - back) & (h["min"] > 0)
    idx = np.flatnonzero(played)[::-1][: P["rate_window"]]
    if len(idx) == 0:
        return {s: mu[s] for s in STATS}, 0.0
    w = 0.5 ** (np.arange(len(idx)) / P["rate_half_life"])
    w = w * np.where(h["season"][idx] == season, 1.0,
                     np.where(h["season"][idx] == season - 1, P["rate_last_season_w"], P.get("rate_older_w", 0.4)))
    mins = h["min"][idx]
    wm = float((w * mins).sum())
    out = {}
    for j, s in enumerate(STATS):
        x = float((w * h["x"][idx, j]).sum())
        out[s] = (x + K[s] * mu[s]) / (wm + K[s])
    return out, wm


# ------------------------------------------------------------ environment ----

def team_usual(hist, team, day, season, P, k_shrink=3.0):
    """A team's usual per-game totals (pts, game total, each stat), shrunk to league average."""
    t = hist.team_before(team, day)
    lg_season = season if season in hist.league_team_avg.index else hist.league_team_avg.index.max()
    lg_prev = season - 1 if season - 1 in hist.league_team_avg.index else lg_season
    lg = hist.league_team_avg.loc[lg_prev]
    keys = STATS + ["team_pts", "game_total", "opp_pts"]
    base = {k: float(lg[k]) if k in lg else float(lg["team_pts"]) for k in keys}
    if t is None or len(t["day"]) == 0:
        return base
    sel = t["season"] >= season - 1
    if not sel.any():
        return base
    idx = np.flatnonzero(sel)[::-1][:30]
    w = 0.5 ** (np.arange(len(idx)) / P["team_half_life"])
    w = w * np.where(t["season"][idx] == season, 1.0, 0.35)
    out = {}
    vals = {"team_pts": t["pts"], "game_total": t["total"], "opp_pts": t["opp_pts"]}
    for j, s in enumerate(STATS):
        vals[s] = t["x"][:, j]
    for k in keys:
        v = vals[k][idx]
        out[k] = float(((w * v).sum() + k_shrink * base[k]) / (w.sum() + k_shrink))
    return out


def expected_environment(hist, team, opp, is_home, day, season, P, vegas=None):
    """Expected team points and game total. Vegas wins when present."""
    tu = team_usual(hist, team, day, season, P)
    ou = team_usual(hist, opp, day, season, P)
    if vegas and vegas.get("total") and vegas.get("spread") is not None:
        total = float(vegas["total"])
        # spread is from THIS team's view (negative = favored)
        team_pts = (total - float(vegas["spread"])) / 2.0
        src = "vegas"
    else:
        lga = hist.league_team_avg
        lg_pts = lga.loc[season - 1, "team_pts"] if season - 1 in lga.index else lga["team_pts"].iloc[0]
        edge = P["home_edge"] / 2.0 * (1 if is_home else -1)
        team_pts = tu["team_pts"] + ou["opp_pts"] - lg_pts + edge
        opp_pts = ou["team_pts"] + tu["opp_pts"] - lg_pts - edge
        total = team_pts + opp_pts
        src = "model"
    return {"team_pts": team_pts, "total": total, "usual": tu, "source": src}


def defense_factor(hist, opp, day, season, pos, P):
    """How much more (or less) fantasy production per minute this defense allows to the
    player's position group than the league does, shrunk toward 1. ^ P["dvp_strength"]."""
    g = P.get("dvp_strength", 0.0)
    d = hist.defense.get(opp) if g else None
    if not g or d is None:
        return 1.0
    grp = 0 if pos <= 2.0 else (1 if pos <= 3.6 else 2)
    n = np.searchsorted(d["day"], day, side="left")
    sel = (d["grp"][:n] == grp) & (d["season"][:n] >= season - 1)
    if not sel.any():
        return 1.0
    w = np.where(d["season"][:n][sel] == season, 1.0, P.get("dvp_last_season_w", 0.35))
    lg = hist.league_grp_rate.get((season - 1, grp)) or hist.league_grp_rate.get((season, grp))
    if not lg:
        return 1.0
    K = P.get("dvp_prior_minutes", 1500.0)
    rate = ((w * d["dk"][:n][sel]).sum() + K * lg) / ((w * d["min"][:n][sel]).sum() + K)
    return (rate / lg) ** g


def rest_adjust(rows, hist, team, day, P):
    """Second night of a back-to-back: starters play a little less, the bench a little more."""
    k = P.get("b2b_starter_min", 0.0)
    th = hist.team_before(team, day)
    if not k or th is None or not len(th["day"]) or day - th["day"][-1] != 1:
        return
    starters = [p for p in rows if p["starter_tonight"] and p["fixed"] is None and p["proj_min"] > 20]
    bench = [p for p in rows if not p["starter_tonight"] and p["fixed"] is None]
    if not starters or not bench:
        return
    moved = 0.0
    for p in starters:
        cut = min(k, p["proj_min"] * 0.15)
        p["proj_min"] -= cut
        moved += cut
    wts = np.array([p["proj_min"] + 2.0 for p in bench])
    for p, gv in zip(bench, moved * wts / wts.sum()):
        p["proj_min"] += gv


def spread_adjust(rows, env, P):
    """Lopsided games: starters sit late, the bench plays more."""
    k = P["spread_min_k"]
    if k <= 0:
        return
    margin = abs(2 * env["team_pts"] - env["total"])
    excess = max(0.0, margin - P["spread_min_start"])
    if excess <= 0:
        return
    starters = [p for p in rows if p["starter_tonight"] and p["fixed"] is None and p["proj_min"] > 20]
    bench = [p for p in rows if not p["starter_tonight"] and p["fixed"] is None]
    if not starters or not bench:
        return
    moved = 0.0
    for p in starters:
        cut = k * excess * p["proj_min"] / 36.0
        p["proj_min"] -= cut
        moved += cut
    wts = np.array([p["proj_min"] + 2.0 for p in bench])
    for p, g in zip(bench, moved * wts / wts.sum()):
        p["proj_min"] += g


# ---------------------------------------------------------------- scoring ----

def _p_at_least_10(mean, coef):
    if mean <= 0:
        return 0.0
    sd = max(coef * math.sqrt(mean), 0.5)
    z = (9.5 - mean) / sd
    return 0.5 * math.erfc(z / math.sqrt(2))


def dd_td_probs(row, P):
    ps = [_p_at_least_10(row[s], P["sd_coef"][s]) for s in ("pts", "reb", "ast", "stl", "blk")]
    # probability that at least 2 / at least 3 of the five categories reach 10
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for k, q in enumerate(dist):
            new[k] += q * (1 - p)
            new[k + 1] += q * p
        dist = new
    return sum(dist[2:]), sum(dist[3:])


def fantasy_points(row, site):
    s = SCORING[site]
    fp = sum(s[k] * row[k] for k in STATS)
    return fp + s["dd"] * row.get("p_dd", 0.0) + s["td"] * row.get("p_td", 0.0)


# ------------------------------------------------------------- team-game ----

def project_team(hist, team, opp, is_home, day, season, roster, out_ids, pos_means,
                 P=PARAMS, vegas=None):
    """roster: list of dicts {player_id, name, pos, starter (True/False/None),
    fixed_min (None or number), cap_min (None or number), status_note}.
    Only players expected to be ACTIVE belong in roster; out_ids are the ones who are not."""
    team_hist = hist.team_before(team, day)
    game_w = filled_in_weights(hist, team, day, season, roster, P)
    rows = []
    for r in roster:
        h = hist.player_before(r["player_id"], day)
        base, start_rate, n, role = minutes_baseline(h, team, season, P, game_w.get(r["player_id"]))
        st = r.get("starter")
        if base is None:
            base = P["no_history_start_min"] if st else P["no_history_min"]
            start_rate = 1.0 if st else 0.0
        elif st is not None:
            delta = P.get("starter_delta_up" if st else "starter_delta_down", P["starter_delta"])
            shifted = base + delta * ((1.0 if st else 0.0) - start_rate)
            mean_r, n_r = role.get(bool(st), (None, 0))
            skip = role.get("jumped") and P.get("jump_skips_role_blend", False)
            if P["role_blend_k"] > 0 and n_r > 0 and not skip:
                wr = n_r / (n_r + P["role_blend_k"])
                base = wr * mean_r + (1 - wr) * shifted
            else:
                base = shifted
        if st and P.get("starter_min_floor", 0) > 0:     # a confirmed starter rarely plays less (off by default)
            base = max(base, P["starter_min_floor"])
        missed = games_missed_in_row(h, team_hist)
        if missed >= P["return_games"] and h is not None:
            base *= P["return_factor"]
        if base > P["high_min_start"] and not is_ironman(h, P):
            base -= P["high_min_shrink"] * (base - P["high_min_start"])
        base = max(base, 0.0)
        fixed = r.get("fixed_min")
        if r.get("cap_min") is not None and fixed is None:
            fixed = min(base, float(r["cap_min"]))
        rows.append({"player_id": r["player_id"], "name": r.get("name") or (h or {}).get("name"),
                     "pos": r["pos"], "r": base, "start_rate": start_rate, "games": n,
                     "starter_tonight": bool(st) if st is not None else start_rate >= 0.5,
                     "fixed": fixed, "missed_in_row": missed, "h": h,
                     "note": r.get("status_note", "")})

    rotation_cut(rows, team_hist, season, P)
    vac = vacated_minutes(hist, team_hist, out_ids, P)
    vac_pos = {}
    for j in vac:
        vac_pos[j] = next((r["pos"] for r in roster if r["player_id"] == j), None)
        if vac_pos[j] is None:
            vac_pos[j] = hist_pos(hist, j)
    allocate_minutes(rows, vac, vac_pos, P)
    if P.get("rot_exempt_protect", False):
        protect_regulars(rows, vac, vac_pos, P)

    env = expected_environment(hist, team, opp, is_home, day, season, P, vegas)
    spread_adjust(rows, env, P)
    rest_adjust(rows, hist, team, day, P)
    pace = (env["total"] / env["usual"]["game_total"]) ** P["pace_alpha"]
    opp_b2b = 1.0
    if P.get("opp_b2b_factor", 1.0) != 1.0:
        oh = hist.team_before(opp, day)
        if oh is not None and len(oh["day"]) and day - oh["day"][-1] == 1:
            opp_b2b = P["opp_b2b_factor"]

    for p in rows:
        rates, _ = per_minute_rates(p["h"], season, p["pos"], pos_means, P)
        dvp = defense_factor(hist, opp, day, season, p["pos"], P)
        p["raw_dk"] = sum(rates[s] * p["proj_min"] * SCORING["DK"][s] for s in STATS)
        p["adj_pace"], p["adj_dvp"] = pace, dvp
        for s in STATS:
            p[s] = rates[s] * p["proj_min"] * pace * dvp * opp_b2b

    # keep team totals realistic (spreads a missing star's production around)
    for s in STATS:
        proj_sum = sum(p[s] for p in rows)
        if proj_sum <= 0:
            continue
        expect = env["team_pts"] if s == "pts" else env["usual"][s] * pace * (
            env["team_pts"] / env["usual"]["team_pts"]) ** P.get("implied_other", 0.0)
        f = (expect / proj_sum) ** P["team_norm_beta"][s]
        eta = P.get("usage_concentration", 0.0)
        mode = P.get("usage_weight_mode", "rate")
        stat_ok = s in (("pts", "ast", "fg3m", "tov") if mode == "rate" else STATS)
        if f > 1 and eta != 0 and stat_ok:
            # extra production (e.g. a missing star's) goes mostly to the players who
            # already produce the most of THIS stat ("total": projected amount tonight,
            # "rate": per minute), not evenly to everyone
            if mode == "total":
                rates = {id(p): p[s] for p in rows}
            else:
                rates = {id(p): p[s] / max(p["proj_min"], 1.0) for p in rows}
            mean_rate = sum(rates.values()) / len(rates) or 1.0
            wts = {k: (max(v, 0.05 * mean_rate) / mean_rate) ** eta for k, v in rates.items()}
            denom = sum(p[s] * wts[id(p)] for p in rows)
            k = proj_sum * (f - 1) / denom if denom > 0 else 0.0
            for p in rows:
                p[s] *= 1 + k * wts[id(p)]
        elif f < 1 and P.get("cut_protect", 0.0) > 0 and stat_ok:
            # a cut (team projected above its total) comes less out of the top producers
            rates = {id(p): p[s] / max(p["proj_min"], 1.0) for p in rows}
            mean_rate = sum(rates.values()) / len(rates) or 1.0
            wts = {k: (max(v, 0.05 * mean_rate) / mean_rate) ** -P["cut_protect"] for k, v in rates.items()}
            denom = sum(p[s] * wts[id(p)] for p in rows)
            k = proj_sum * (1 - f) / denom if denom > 0 else 0.0
            for p in rows:
                p[s] *= max(1 - k * wts[id(p)], 0.5)
        else:
            for p in rows:
                p[s] *= f

    out = []
    for p in rows:
        rec = {k: p[k] for k in ("player_id", "name", "pos", "proj_min", "r", "start_rate",
                                 "games", "starter_tonight", "missed_in_row", "note",
                                 "raw_dk", "adj_pace", "adj_dvp")}
        rec.update({s: p[s] for s in STATS})
        rec["p_dd"], rec["p_td"] = dd_td_probs(rec, P)
        for site in SCORING:
            rec[f"fp_{site}"] = fantasy_points(rec, site)
        rec.update({"team": team, "opp": opp, "is_home": is_home,
                    "team_total": env["team_pts"], "game_total": env["total"],
                    "env_source": env["source"]})
        out.append(rec)
    return out


_POS_CACHE = {}


def hist_pos(hist, pid):
    return _POS_CACHE.get(pid, 3.0)


def set_positions(pos_map):
    _POS_CACHE.clear()
    _POS_CACHE.update(pos_map)
