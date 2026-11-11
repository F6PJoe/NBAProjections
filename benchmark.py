"""Outside projections to measure ourselves against. BENCHMARK ONLY - nothing in the
model reads these (same rule as the NFL model: our numbers stay our own).

  DFF          DailyFantasyFuel, free DK / FD main-slate projections, past dates kept
  FantasyPros  daily stat-line projections (needs FANTASYPROS_API_KEY in .env); stats are
               whole numbers, so their fantasy points carry a little rounding noise
  BlueCollar   the feed the site shows today (needs BLUECOLLAR_API_KEY); current slates only
"""
import html
import json
import os
import re
import time

import pandas as pd
import requests

from config import CACHE, secret, team_code

DFF = "https://www.dailyfantasyfuel.com/nba/projections/{site}/{date}"
SITES = {"DK": "draftkings", "FD": "fanduel"}
ATTRS = ["start_date", "name", "pos", "team", "opp", "salary", "ppg_proj", "spread", "ou",
         "inj", "l5_avg", "l10_avg", "szn_avg", "depth_rank"]


def parse_dff(page):
    rows = []
    for tag in re.findall(r'<tr[^>]*class="[^"]*projections-listing[^"]*"[^>]*>', page):
        rec = {}
        for a in ATTRS:
            m = re.search(rf'data-{a}=\s*"([^"]*)"', tag)
            rec[a] = html.unescape(m.group(1)) if m else None
        rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ("salary", "ppg_proj", "spread", "ou", "l5_avg", "l10_avg", "szn_avg"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["team"] = df["team"].map(team_code)
    df["opp"] = df["opp"].map(team_code)
    return df


DFF_SLATES = "https://www.dailyfantasyfuel.com/data/slates/recent/NBA/{site}"


def _dff_page(date, site, slate, refresh):
    folder = CACHE / "dff"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"{SITES[site]}_{date:%Y-%m-%d}{'_' + slate if slate else ''}.html"
    today = pd.Timestamp.now(tz="America/New_York").date()
    if not f.exists() or refresh or date >= today:
        r = requests.get(DFF.format(site=SITES[site], date=f"{date:%Y-%m-%d}"),
                         params={"slate": slate} if slate else None, timeout=60)
        time.sleep(1.0)
        if r.status_code != 200:
            return pd.DataFrame()
        f.write_text(r.text, encoding="utf-8")
    df = parse_dff(f.read_text(encoding="utf-8"))
    return df[df["start_date"] == f"{date:%Y-%m-%d}"] if not df.empty else df


def dff_slates(date, site="DK"):
    """Every CLASSIC (non-showdown) slate DFF has for that date: [(slate id, label)]."""
    folder = CACHE / "dff"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"slates_{SITES[site]}_{date:%Y-%m-%d}.json"
    today = pd.Timestamp.now(tz="America/New_York").date()
    if not f.exists() or date >= today:
        r = requests.get(DFF_SLATES.format(site=SITES[site]),
                         params={"date": f"{date:%Y-%m-%d}", "url": ""}, timeout=60)
        time.sleep(1.0)
        if r.status_code != 200 or not r.text.startswith("{"):
            return []
        f.write_text(r.text, encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    return [(s["url"], f"{s.get('game_count')} games {s.get('slate_type') or 'Main'}")
            for s in d.get("slates", []) if not s.get("showdown_flag")]


def dff(date, site="DK", refresh=False):
    """DFF projections for EVERY game that day: all classic slates merged (the default page
    is only the main slate). Cached; past dates never re-downloaded."""
    frames = [_dff_page(date, site, None, refresh)]
    for slate, _ in dff_slates(date, site):
        frames.append(_dff_page(date, site, slate, refresh))
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["name", "team"])


# ------------------------------------------------------------ FantasyPros ----

FP_URL = "https://api.fantasypros.com/public/v2/json/nba/{season}/projections"


def fantasypros(date, season, refresh=False):
    """FantasyPros daily projections -> player_id (NBA.com id), stats, fp_DK, fp_FD."""
    key = secret("FANTASYPROS_API_KEY")
    if not key:
        return pd.DataFrame()
    folder = CACHE / "fantasypros"
    folder.mkdir(parents=True, exist_ok=True)
    f = folder / f"daily_{date:%Y-%m-%d}.json"
    today = pd.Timestamp.now(tz="America/New_York").date()
    if not f.exists() or refresh or date >= today:
        r = requests.get(FP_URL.format(season=season), headers={"x-api-key": key},
                         params={"type": "daily", "date": f"{date:%Y-%m-%d}"}, timeout=60)
        time.sleep(1.1)
        if r.status_code != 200 or not r.text.startswith("{"):
            return pd.DataFrame()
        f.write_text(r.text, encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    if d.get("date") != f"{date:%Y-%m-%d}":
        return pd.DataFrame()
    rows = []
    for p in d.get("player") or []:
        if not p.get("nbacom_player_id") or not p.get("minutes"):
            continue
        st = {k: float(p.get(src) or 0) for k, src in
              (("pts", "points"), ("reb", "rebounds"), ("ast", "assists"), ("stl", "steals"),
               ("blk", "blocks"), ("tov", "turnovers"), ("fg3m", "three_points_made"), ("min", "minutes"))}
        tens = sum(st[k] >= 10 for k in ("pts", "reb", "ast", "stl", "blk"))
        st["fp_DK"] = (st["pts"] + 0.5 * st["fg3m"] + 1.25 * st["reb"] + 1.5 * st["ast"] + 2 * st["stl"]
                       + 2 * st["blk"] - 0.5 * st["tov"] + 1.5 * (tens >= 2) + 3 * (tens >= 3))
        st["fp_FD"] = st["pts"] + 1.2 * st["reb"] + 1.5 * st["ast"] + 3 * st["stl"] + 3 * st["blk"] - st["tov"]
        rows.append({"player_id": int(p["nbacom_player_id"]), "name": p.get("name"),
                     "team": team_code(p.get("team_id")), **st})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- BlueCollar ----

BC_URL = "https://bluecollardfs.com/api/nba_{site}"
BC_SITES = {"DK": "draftkings", "FD": "fanduel"}


def _bc_rows(players):
    """BlueCollar players come as 10 fields: start, opp, player, id, pos, team, proj, salary, beta, value."""
    out = []
    for p in players or []:
        if isinstance(p, dict):
            low = {str(k).lower(): v for k, v in p.items()}
            name = low.get("player") or low.get("name")
            team = low.get("team")
            proj = low.get("proj") or low.get("projection") or low.get("fpts")
            if name is None:  # unnamed fields: fall back to position
                vals = list(p.values())
                name, team, proj = (vals[2], vals[5], vals[6]) if len(vals) >= 7 else (None, None, None)
        else:
            vals = list(p)
            name, team, proj = (vals[2], vals[5], vals[6]) if len(vals) >= 7 else (None, None, None)
        if name:
            out.append({"name": name, "team": team_code(team), "proj": pd.to_numeric(proj, errors="coerce")})
    return out


BC_TTL_MIN = 30   # the key allows 200 requests/day across all endpoints (shared with anything
                  # else using it), so reuse a fetch for 30 minutes: at most ~96 calls/day here


def bluecollar(date, site="DK"):
    """BlueCollar slates -> name, team, proj. Auth per bluecollardfs.com/developers:
    'Authorization: ApiKey <key>'. Response: {"slates": [{"slate", "info": [players]}]}."""
    key = secret("BLUECOLLAR_API_KEY")
    if not key:
        return pd.DataFrame()
    folder = CACHE / "bluecollar"
    folder.mkdir(parents=True, exist_ok=True)
    recent = sorted(folder.glob(f"{BC_SITES[site]}_{date:%Y-%m-%d}_*.json"))
    now = pd.Timestamp.now(tz="America/New_York")
    ttl = float(os.environ.get("BC_TTL_MIN") or BC_TTL_MIN)
    age = None
    if recent:
        hhmm = recent[-1].stem[-4:]
        saved_at = pd.Timestamp(f"{date:%Y-%m-%d} {hhmm[:2]}:{hhmm[2:]}", tz="America/New_York")
        age = (now - saved_at).total_seconds() / 60
    if recent and (age < ttl or now.date() > date):
        text = recent[-1].read_text(encoding="utf-8")
    else:
        r = requests.get(BC_URL.format(site=BC_SITES[site]),
                         headers={"Authorization": f"ApiKey {key}"}, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"BlueCollar {site} HTTP {r.status_code}: {r.text[:120]}")
        text = r.text
        stamp = pd.Timestamp.now(tz="America/New_York").strftime("%H%M")
        (folder / f"{BC_SITES[site]}_{date:%Y-%m-%d}_{stamp}.json").write_text(text, encoding="utf-8")
    d = json.loads(text)
    rows = []
    slates = [s for s in d.get("slates") or [] if isinstance(s, dict)]
    slates.sort(key=lambda s: s.get("slate_type") != "classic")   # classic numbers win over showdown
    for sl in slates:
        info = sl.get("info")
        if isinstance(info, dict):          # older shape: info.players
            info = info.get("players")
        rows += _bc_rows(info)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["name", "team"]) if not df.empty else df


def bluecollar_saved(date, site="DK", before=None):
    """BlueCollar pulls saved on that date, combined: the latest number per player (games leave
    their slates once they tip, so the last pull alone misses early games). Optionally only
    pulls before HHMM."""
    files = sorted((CACHE / "bluecollar").glob(f"{BC_SITES[site]}_{date:%Y-%m-%d}_*.json"))
    if before:
        files = [f for f in files if f.stem[-4:] < before] or files[:1]
    if not files:
        return pd.DataFrame(), None
    frames = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        slates = [s for s in d.get("slates") or [] if isinstance(s, dict)]
        slates.sort(key=lambda s: s.get("slate_type") != "classic")
        rows = []
        for sl in slates:
            info = sl.get("info")
            if isinstance(info, dict):
                info = info.get("players")
            rows += _bc_rows(info)
        if rows:
            frames.append(pd.DataFrame(rows).drop_duplicates(["name", "team"]))
    if not frames:
        return pd.DataFrame(), files[-1].stem[-4:]
    df = pd.concat(frames).drop_duplicates(["name", "team"], keep="last")
    return df, files[-1].stem[-4:]


# ----------------------------------------------------------------- attach ----

def attach(allp, matcher, date, season=None):
    """Add outside numbers as bench_<source>_<site> columns (benchmark only).
    Never fails the run: any problem just leaves that source's columns blank."""
    def by_name(df, col):
        ids = [matcher.match(n, t) for n, t in zip(df["name"], df["team"])]
        vals = dict(zip(ids, df[col]))
        vals.pop(None, None)
        return vals

    for site in SITES:
        col = f"bench_DFF_{site}"
        allp[col] = float("nan")
        try:
            df = dff(date, site, refresh=True)
            if not df.empty:
                allp[col] = allp["player_id"].map(by_name(df, "ppg_proj"))
        except Exception as e:  # noqa: BLE001
            print(f"  [benchmark] DFF {site} unavailable ({e.__class__.__name__})")

    allp["bench_FP_DK"] = float("nan")
    allp["bench_FP_FD"] = float("nan")
    if season and secret("FANTASYPROS_API_KEY"):
        try:
            fp = fantasypros(date, season, refresh=True)
            if not fp.empty:
                fp = fp.drop_duplicates("player_id").set_index("player_id")
                allp["bench_FP_DK"] = allp["player_id"].map(fp["fp_DK"])
                allp["bench_FP_FD"] = allp["player_id"].map(fp["fp_FD"])
        except Exception as e:  # noqa: BLE001
            print(f"  [benchmark] FantasyPros unavailable ({e.__class__.__name__})")

    for site in BC_SITES:
        col = f"bench_BC_{site}"
        allp[col] = float("nan")
        if not secret("BLUECOLLAR_API_KEY"):
            continue
        try:
            bc = bluecollar(date, site)
            if not bc.empty:
                allp[col] = allp["player_id"].map(by_name(bc, "proj"))
        except Exception as e:  # noqa: BLE001
            print(f"  [benchmark] BlueCollar {site} unavailable ({str(e)[:120]})")
    return allp
