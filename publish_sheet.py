"""Push output/latest/ site tables into the Google Sheet that feeds wpDataTables.

Same layout the BlueCollar R script writes (Player, Proj, Salary, Value, Pos, Team, Opp)
plus the "NBA Update Time" stamp, so the site tables keep working unchanged.

  python publish_sheet.py            # writes to the TEST tabs ("... (new)")
  python publish_sheet.py --live     # writes to the tabs the site reads

Google login: a service-account key, from either
  env GCP_SHEETS_KEY_B64  (base64 of the key JSON - what the R script / GitHub use), or
  env GCP_SHEETS_KEY_PATH (path to the key JSON file).
The sheet must be shared with the service account's e-mail.
"""
import argparse
import base64
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import gspread
import pandas as pd

from config import OUTPUT

SHEET_ID = "1dWsEg3HLa9KY1YES31P1Mam0vLFK9zrR91rOsDSKsA8"
TABS = {"DK": "DK NBA DFS", "FD": "FD NBA DFS", "Yahoo": "Yahoo NBA DFS"}
TIME_TAB = "NBA Update Time"
COLS = ["Player", "Proj", "Salary", "Value", "Pos", "Team", "Opp"]
ET = ZoneInfo("America/New_York")


def client():
    b64 = os.environ.get("GCP_SHEETS_KEY_B64")
    if b64:
        return gspread.service_account_from_dict(json.loads(base64.b64decode(b64)))
    path = os.environ.get("GCP_SHEETS_KEY_PATH")
    if path:
        return gspread.service_account(filename=path)
    raise SystemExit("No Google key: set GCP_SHEETS_KEY_B64 or GCP_SHEETS_KEY_PATH")


def write_tab(sh, title, df):
    try:
        ws = sh.worksheet(title)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=max(len(df) + 10, 100), cols=len(df.columns) + 2)
    ws.clear()
    values = [list(df.columns)] + df.astype(object).where(df.notna(), "").values.tolist()
    ws.update(values, "A1")


def main(live=False, min_proj=5.0):
    sh = client().open_by_key(SHEET_ID)
    latest = OUTPUT / "latest"
    for site, tab in TABS.items():
        f = latest / f"{site}.csv"
        if not f.exists():
            print(f"  {site}: no file, skipped")
            continue
        df = pd.read_csv(f)
        df = df[df["Proj"] >= min_proj][COLS]
        name = tab if live else f"{tab} (new)"
        write_tab(sh, name, df)
        print(f"  {site}: {len(df)} rows -> '{name}'")
    now = dt.datetime.now(ET)
    if live:
        ws = sh.worksheet(TIME_TAB)
        ws.update([[now.strftime("%B %d, %Y"), now.strftime("%I:%M %p ET")]], "A2:B2")
        print(f"  stamped {TIME_TAB}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    main(ap.parse_args().live)
