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

def minutes_baseline(h, team, season, P):
    """Return (baseline minutes, weighted start rate, games used)."""
    if h is None:
        return None, 0.0, 0, {}
    avail = ((h["status"] == "played") | (h["status"] == "dnp_cd")) & ~h["left_hurt"]
    cur = avail & (h["season"] == season)
    idx = np.flatnonzero(cur)[::-1][: P["min_window"]]
    prior = None
    last = avail & (h["season"] == season - 1)
    if last.sum() >= 5:
        prior = float(h["min"][last][-40:].mean())
    if len(idx) == 0:
        if prior is None:
            return None, 0.0, 0, {}
        st = h["starter"][last][-20:].mean() if last.any() else 0.0
        return prior, float(st), 0, {}
    k = np.arange(len(idx))
    w = 0.5 ** (k / P["min_half_life"])
    w = w * np.where(h["team"][idx] == team, 1.0, P["min_other_team_w"])
    w = w * np.where(np.abs(h["margin"][idx]) >= P["blowout_margin"], P["blowout_w"], 1.0)
    w = w * np.where(h["status"][idx] == "dnp_cd", P.get("dnp_cd_w", 1.0), 1.0)
    m = h["min"][idx]
    base = float((w * m).sum() / w.sum())
    if prior is not None:
        kp = P["min_prior_games"]
        base = float(((w * m).sum() + kp * prior) / (w.sum() + kp))
    st = h["starter"][idx]
    start_rate = float((w * st).sum() / w.sum())
    role = {}
    for flag in (True, False):
        sel = st == flag
        role[flag] = (float((w[sel] * m[sel]).sum() / w[sel].sum()), int(sel.sum())) if sel.any() else (None, 0)
    return base, start_rate, len(idx), role


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
    played = (h["status"] == "played") & (h["season"] >= season - 1) & (h["min"] > 0)
    idx = np.flatnonzero(played)[::-1][: P["rate_window"]]
    if len(idx) == 0:
        return {s: mu[s] for s in STATS}, 0.0
    w = 0.5 ** (np.arange(len(idx)) / P["rate_half_life"])
    w = w * np.where(h["season"][idx] == season, 1.0, P["rate_last_season_w"])
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
    rows = []
    for r in roster:
        h = hist.player_before(r["player_id"], day)
        base, start_rate, n, role = minutes_baseline(h, team, season, P)
        st = r.get("starter")
        if base is None:
            base = P["no_history_start_min"] if st else P["no_history_min"]
            start_rate = 1.0 if st else 0.0
        elif st is not None:
            delta = P.get("starter_delta_up" if st else "starter_delta_down", P["starter_delta"])
            shifted = base + delta * ((1.0 if st else 0.0) - start_rate)
            mean_r, n_r = role.get(bool(st), (None, 0))
            if P["role_blend_k"] > 0 and n_r > 0:
                wr = n_r / (n_r + P["role_blend_k"])
                base = wr * mean_r + (1 - wr) * shifted
            else:
                base = shifted
        missed = games_missed_in_row(h, team_hist)
        if missed >= P["return_games"] and h is not None:
            base *= P["return_factor"]
        if base > P["high_min_start"]:
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

    vac = vacated_minutes(hist, team_hist, out_ids, P)
    vac_pos = {}
    for j in vac:
        vac_pos[j] = next((r["pos"] for r in roster if r["player_id"] == j), None)
        if vac_pos[j] is None:
            vac_pos[j] = hist_pos(hist, j)
    allocate_minutes(rows, vac, vac_pos, P)

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
        else:
            for p in rows:
                p[s] *= f

    out = []
    for p in rows:
        rec = {k: p[k] for k in ("player_id", "name", "pos", "proj_min", "r", "start_rate",
                                 "games", "starter_tonight", "missed_in_row", "note")}
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
