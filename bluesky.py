"""Bluesky posts from NBA beat writers and news accounts - COLLECTED FOR TESTING ONLY, not a model input.

  python bluesky.py              # save any new posts (also runs inside every projection run)
  python bluesky.py 2026-10-01   # backfill: page back to this date

Reads each account in bluesky_accounts.csv through Bluesky's official public API (no login) and
appends new posts to cache/bsky/bsky_<YYYY-MM>.csv in the PRIVATE data repo (post text is the
authors' own; never commit it to the public repo). After a few weeks, research tests whether
rotation news in these posts ("out of the rotation", "Toppin back") would have caught the bench
minutes we miss. RotoWire's account is left out on purpose (their terms bar this use).
"""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from config import CACHE

API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
ACCOUNTS = Path(__file__).with_name("bluesky_accounts.csv")
FOLDER = CACHE / "bsky"
COLS = ["uri", "created", "team", "handle", "text", "reply", "repost_of", "link"]


def _feed(team, handle, limit=30, since=None):
    """Latest posts; with `since` (ISO date) keeps paging back until posts are older than it."""
    rows, cursor = [], None
    while True:
        page = _page(team, handle, limit if since is None else 100, cursor)
        if page is None:
            return rows
        got, cursor = page
        rows += got
        if since is None or not cursor or not got or min(r["created"] for r in got) < since:
            return rows


def _page(team, handle, limit, cursor):
    try:
        params = {"actor": handle, "limit": limit, **({"cursor": cursor} if cursor else {})}
        r = requests.get(API, params=params, timeout=20)
        if r.status_code != 200:
            return None
        rows = []
        data = r.json()
        for it in data.get("feed", []):
            p = it["post"]
            rec = p.get("record", {})
            repost = it.get("reason", {}).get("$type", "").endswith("reasonRepost")
            if p["author"]["handle"] != handle and not repost:
                continue
            emb = (p.get("embed") or {}).get("external") or {}
            rows.append({"uri": p["uri"] if not repost else f"{p['uri']}#by:{handle}",
                         "created": rec.get("createdAt"), "team": team, "handle": handle,
                         "text": rec.get("text", ""), "reply": bool(rec.get("reply")),
                         "repost_of": p["author"]["handle"] if repost else "",
                         "link": emb.get("uri", "")})
        return rows, data.get("cursor")
    except Exception:
        return None


def fetch(since=None):
    acc = pd.read_csv(ACCOUNTS)
    with ThreadPoolExecutor(8) as ex:
        parts = ex.map(lambda r: _feed(r.team, r.handle, since=since), acc.itertuples())
    return pd.DataFrame([row for part in parts for row in part], columns=COLS)


def pull(since=None):
    """Append any posts not saved yet (since=ISO date backfills). Returns how many were new."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    df = fetch(since)
    if df.empty:
        return 0
    df = df.drop_duplicates("uri")
    df["created"] = pd.to_datetime(df["created"], utc=True, format="ISO8601")
    new = 0
    for month, g in df.groupby(df["created"].dt.strftime("%Y-%m")):
        f = FOLDER / f"bsky_{month}.csv"
        old = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=COLS)
        old["created"] = pd.to_datetime(old["created"], utc=True, format="ISO8601")
        add = g[~g["uri"].isin(old["uri"])]
        if len(add):
            both = add if old.empty else pd.concat([old, add])
            both.drop_duplicates("uri").sort_values("created").to_csv(f, index=False)
            new += len(add)
    return new


def load(start=None, end=None):
    """Saved posts, flagged with the same rotation phrases as the FantasyPros news test."""
    from news import FLAGS
    files = sorted(FOLDER.glob("bsky_*.csv"))
    if not files:
        return pd.DataFrame(columns=COLS)
    df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("uri")
    df["created"] = pd.to_datetime(df["created"], utc=True, format="ISO8601")
    if start is not None:
        df = df[df["created"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["created"] < pd.Timestamp(end, tz="UTC")]
    text = df["text"].fillna("").str.lower()
    for k, pat in FLAGS.items():
        df[k] = text.str.contains(pat, regex=True)
    return df


if __name__ == "__main__":
    import sys
    n = pull(sys.argv[1] if len(sys.argv) > 1 else None)
    df = load()
    print(f"{n} new posts; {len(df)} saved from {df['handle'].nunique()} accounts" if len(df) else "nothing saved")
