"""Download and read the NBA's official injury report (a PDF re-posted every 15 minutes).

The files live at predictable addresses, so we ask for the newest time slot and
step back 15 minutes at a time until one exists.
"""
import datetime as dt
import re
from zoneinfo import ZoneInfo

import pdfplumber
import requests

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{t}.pdf"
ET = ZoneInfo("America/New_York")
STATUSES = ["Out", "Doubtful", "Questionable", "Probable", "Available"]


def slot_name(ts):
    h = ts.hour % 12 or 12
    return f"{h:02d}_{ts.minute:02d}{'AM' if ts.hour < 12 else 'PM'}"


def latest(folder, now=None, max_back=48):
    """Save the newest report into folder; return (path, report time) or (None, None)."""
    now = now or dt.datetime.now(ET)
    ts = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    folder.mkdir(parents=True, exist_ok=True)
    for _ in range(max_back):
        name = f"Injury-Report_{ts:%Y-%m-%d}_{slot_name(ts)}.pdf"
        dest = folder / name
        if dest.exists() and dest.stat().st_size > 1000:
            return dest, ts
        try:
            r = requests.get(URL.format(d=f"{ts:%Y-%m-%d}", t=slot_name(ts)), timeout=30)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200 and r.content[:4] == b"%PDF":
            dest.write_bytes(r.content)
            return dest, ts
        ts -= dt.timedelta(minutes=15)
    return None, None


def parse(path):
    """Rows: game_date, matchup, team (full name), player ('Last, First'), status, reason."""
    rows = []
    cur = {"game_date": None, "matchup": None, "team": None}
    with pdfplumber.open(path) as pdf:
        cols = None
        for page in pdf.pages:
            words = page.extract_words(keep_blank_chars=False, use_text_flow=False,
                                       extra_attrs=["size"])
            # header positions (same on every page)
            hdr_words = [w for w in words if w["text"] in
                         ("GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason")]
            hdr = {w["text"]: w["x0"] for w in hdr_words}
            if len(hdr) == 7:
                cols = hdr
            title_top = max((w["top"] for w in words if w["text"] == "Report:"), default=0)
            header_top = max([w["top"] for w in hdr_words] + [title_top])
            if not cols:
                continue
            edges = sorted(cols.items(), key=lambda kv: kv[1])
            names = [k for k, _ in edges]
            xs = [x for _, x in edges]

            def col_of(x):
                i = 0
                for j, cx in enumerate(xs):
                    if x >= cx - 2:
                        i = j
                return names[i]

            lines = {}
            for w in words:
                if w["top"] <= header_top + 1 or w["text"].startswith("Page"):
                    continue
                key = round(w["top"])
                lines.setdefault(key, []).append(w)
            for top in sorted(lines):
                cells = {}
                for w in sorted(lines[top], key=lambda w: w["x0"]):
                    c = col_of(w["x0"])
                    cells[c] = (cells.get(c, "") + " " + w["text"]).strip()
                if cells.get("GameDate") and re.match(r"\d\d/\d\d/\d{4}", cells["GameDate"]):
                    cur["game_date"] = cells["GameDate"][:10]
                if cells.get("Matchup") and "@" in cells["Matchup"]:
                    cur["matchup"] = cells["Matchup"].split()[0]
                if cells.get("Team"):
                    cur["team"] = cells["Team"]
                player = cells.get("PlayerName", "")
                status = cells.get("CurrentStatus", "")
                if "NOTYETSUBMITTED" in (cells.get("Reason", "") + cells.get("PlayerName", "")
                                         ).replace(" ", "").upper() and not status:
                    rows.append({**cur, "player": None, "status": "NOT_SUBMITTED", "reason": ""})
                    continue
                if player and "," in player and status in STATUSES:
                    rows.append({**cur, "player": player, "status": status,
                                 "reason": cells.get("Reason", "")})
                elif (rows and not player and cells.get("Reason") and not status
                      and "NOTYETSUBMITTED" not in cells["Reason"].replace(" ", "").upper()):
                    rows[-1]["reason"] = (rows[-1]["reason"] + " " + cells["Reason"]).strip()
    return rows


TEAM_NAMES = {
    "AtlantaHawks": "ATL", "BostonCeltics": "BOS", "BrooklynNets": "BKN", "CharlotteHornets": "CHA",
    "ChicagoBulls": "CHI", "ClevelandCavaliers": "CLE", "DallasMavericks": "DAL",
    "DenverNuggets": "DEN", "DetroitPistons": "DET", "GoldenStateWarriors": "GSW",
    "HoustonRockets": "HOU", "IndianaPacers": "IND", "LAClippers": "LAC", "LosAngelesClippers": "LAC",
    "LosAngelesLakers": "LAL", "MemphisGrizzlies": "MEM", "MiamiHeat": "MIA", "MilwaukeeBucks": "MIL",
    "MinnesotaTimberwolves": "MIN", "NewOrleansPelicans": "NOP", "NewYorkKnicks": "NYK",
    "OklahomaCityThunder": "OKC", "OrlandoMagic": "ORL", "Philadelphia76ers": "PHI",
    "PhoenixSuns": "PHX", "PortlandTrailBlazers": "POR", "SacramentoKings": "SAC",
    "SanAntonioSpurs": "SAS", "TorontoRaptors": "TOR", "UtahJazz": "UTA", "WashingtonWizards": "WAS",
}


def team_tricode(full):
    return TEAM_NAMES.get((full or "").replace(" ", ""))


def split_camel(name):
    """The PDF drops spaces: 'CenacJr.,Christopher' -> 'Cenac Jr., Christopher'."""
    last, _, first = name.partition(",")
    fix = lambda s: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("Jr.", " Jr.").strip()
    return f"{fix(last)}, {fix(first)}".replace("  ", " ")
