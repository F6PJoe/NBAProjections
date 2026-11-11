"""Match player names from DK / FD / Yahoo / injury report onto NBA player ids."""
import csv
import difflib
import re
import unicodedata

from config import DATA

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name):
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    toks = [t for t in re.split(r"\s+", s) if t and t not in SUFFIXES]
    return " ".join(toks)


def flip_last_first(name):
    """'Gilgeous-Alexander, Shai' -> 'Shai Gilgeous-Alexander' (injury report style)."""
    if "," in name:
        parts = [p.strip() for p in name.split(",") if p.strip()]
        return " ".join([parts[-1]] + parts[:-1])
    return name


def load_aliases():
    """data/name_aliases.csv: source_name,nba_name  (for the odd one the matcher misses)."""
    f = DATA / "name_aliases.csv"
    out = {}
    if f.exists():
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out[norm(row["source_name"])] = norm(row["nba_name"])
    return out


class Matcher:
    """people: iterable of (player_id, name, team)."""

    def __init__(self, people):
        self.aliases = load_aliases()
        self.by_team = {}
        self.all = {}
        for pid, name, team in people:
            n = norm(name)
            self.by_team.setdefault(team, {})[n] = pid
            self.all.setdefault(n, set()).add(pid)

    def match(self, name, team=None):
        n = norm(name)
        n = self.aliases.get(n, n)
        pool = self.by_team.get(team, {}) if team else {}
        if n in pool:
            return pool[n]
        if n in self.all and len(self.all[n]) == 1:
            return next(iter(self.all[n]))
        # same last name + first initial on the team
        toks = n.split()
        if pool and len(toks) >= 2:
            cands = [pid for nm, pid in pool.items()
                     if nm.split()[-1] == toks[-1] and nm[:1] == n[:1]]
            if len(cands) == 1:
                return cands[0]
        if pool:
            close = difflib.get_close_matches(n, list(pool), n=1, cutoff=0.85)
            if close:
                return pool[close[0]]
        return None
