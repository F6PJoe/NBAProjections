"""FantasyPros NBA news (their own staff write-ups) - COLLECTED FOR TESTING ONLY, not a model input.

  python news.py            # save any new items (also runs inside every projection run)

Their feed only keeps ~1-2 days, so every run appends new items to cache/news/fp_news_<YYYY-MM>.csv
(private data repo). After a few weeks, research tests whether rotation phrases ("out of the
rotation", "increased role", "minutes restriction") would have caught the minutes we miss.
FantasyPros' site terms bar commercial use unless authorized: confirm Joe's API agreement covers
it before news ever drives published numbers.
ESPN's player news is NOT used: most blurbs are type "Rotowire" (RotoWire terms forbid it).
"""
import datetime as dt
import re

import pandas as pd
import requests

from config import CACHE, secret

URL = "https://api.fantasypros.com/public/v2/json/nba/news"
FOLDER = CACHE / "news"
COLS = ["id", "created", "player_id", "team_id", "title", "desc", "impact", "categories", "author"]

# phrases worth flagging (used by the research test; kept here so live and test agree)
FLAGS = {
    "out_of_rotation": r"out of the rotation|fell out of|dnp[- ]cd|coach'?s decision|did not play|didn't play|healthy scratch|won't see the floor|glued to the bench",
    "more_minutes": r"increased (?:role|playing time|minutes|usage)|uptick in (?:usage|minutes|playing time)|expanded role|bigger role|more minutes|candidates? for (?:increased|more)|step into|elevated role|move into the starting",
    "fewer_minutes": r"reduced role|decreased role|fewer minutes|less playing time|limited role|lost (?:his|a) spot|moved to the bench|bench role",
    "restriction": r"minutes? (?:restriction|limit)|limited minutes|on a pitch count|managed minutes",
    "returning": r"(?:will|set to|expected to) return|cleared to play|will make his (?:season )?debut|back in the lineup",
}


def fetch(pages=3):
    key = secret("FANTASYPROS_API_KEY")
    if not key:
        return pd.DataFrame(columns=COLS)
    rows = []
    for page in range(1, pages + 1):
        r = requests.get(URL, headers={"x-api-key": key}, params={"limit": 100, "page": page}, timeout=30)
        if r.status_code != 200:
            break
        items = r.json().get("items", [])
        if not items:
            break
        rows += [{c: (", ".join(it.get(c) or []) if c == "categories" else it.get(c)) for c in COLS} for it in items]
    return pd.DataFrame(rows, columns=COLS)


def pull():
    """Append any items not saved yet. Returns how many were new."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    df = fetch()
    if df.empty:
        return 0
    df = df.drop_duplicates("id")
    df["created"] = pd.to_datetime(df["created"])
    new = 0
    for month, g in df.groupby(df["created"].dt.strftime("%Y-%m")):
        f = FOLDER / f"fp_news_{month}.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLS)
        add = g[~g["id"].isin(old["id"])]
        if len(add):
            both = add if old.empty else pd.concat([old, add])
            both.drop_duplicates("id").sort_values("created").to_csv(f, index=False)
            new += len(add)
    return new


def load(start=None, end=None):
    files = sorted(FOLDER.glob("fp_news_*.csv"))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("id")
    df["created"] = pd.to_datetime(df["created"])
    if start is not None:
        df = df[df["created"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["created"] < pd.Timestamp(end)]
    text = (df["title"].fillna("") + " " + df["desc"].fillna("") + " " + df["impact"].fillna("")).str.lower()
    for k, pat in FLAGS.items():
        df[k] = text.str.contains(pat, regex=True)
    return df


if __name__ == "__main__":
    n = pull()
    df = load()
    print(f"{n} new items; {len(df)} saved since {df['created'].min():%Y-%m-%d %H:%M} UTC" if len(df) else "nothing saved")
    if len(df):
        print(df[list(FLAGS)].sum().to_string())
