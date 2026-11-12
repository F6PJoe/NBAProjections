"""Paths, team codes and model settings for the NBA DFS projections.

Every number the model uses lives in PARAMS so the backtest can tune it and the
live run uses exactly the same values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")      # private keys (FantasyPros, BlueCollar, Google) - never committed


def secret(name):
    """A key from .env / the environment, or None when blank."""
    v = os.environ.get(name, "").strip()
    return v or None


# Where downloads and results go. Locally = this folder; in GitHub Actions = the private data repo.
DATA_HOME = Path(os.environ.get("NBA_DATA_DIR") or ROOT)
CACHE = DATA_HOME / "cache"     # downloaded raw inputs (safe to delete; refresh.py rebuilds)
SDV = CACHE / "sdv"             # SportsDataverse (hoopR) parquet files
LIVE = CACHE / "live"           # today's slate inputs, one folder per date
DATA = ROOT / "data"            # hand-kept files (overrides, name aliases)
OUTPUT = DATA_HOME / "output"

SDV_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

# SportsDataverse labels a season by the year it ENDS: 2027 = the 2026-27 season.
CURRENT_SEASON = 2027

# NBA tricodes are the house standard; every other source is mapped onto them.
NBA_TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
             "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
             "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]
TEAM_ALIASES = {
    "GS": "GSW", "NY": "NYK", "NO": "NOP", "NOR": "NOP", "SA": "SAS", "SAN": "SAS",
    "PHO": "PHX", "UTAH": "UTA", "UTH": "UTA", "WSH": "WAS", "BRK": "BKN", "BKLYN": "BKN",
    "CHO": "CHA", "CHH": "CHA",
}
# How the site's Google Sheet has always written team codes (BlueCollar style).
SITE_TEAM_CODES = {"GSW": "GS", "NYK": "NY", "NOP": "NO", "SAS": "SA"}


def team_code(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    return TEAM_ALIASES.get(x, x)


# Stats the model projects per minute.
STATS = ["pts", "fg3m", "reb", "ast", "stl", "blk", "tov"]

# Tuned 2026-11-06 by research/tune.py (two passes; see research/tune_log_round*.txt).
PARAMS = {
    # ---- minutes ----
    "min_half_life": 5.0,        # games: weight halves every N available games back
    "min_window": 30,            # most recent available games used for minutes
    "min_prior_games": 0.5,      # pseudo-games of last season's minutes mixed in
    "min_other_team_w": 0.3,     # weight on games played for a previous team
    "blowout_margin": 20,        # final margin at/over which a game is down-weighted
    "blowout_w": 0.15,
    "starter_delta": 4.0,        # minutes gained/lost when tonight's role differs from usual
    "starter_delta_up": 5.5,     # 2026-11-09: bench player starting tonight (3 seasons: promoted min bias -0.8 -> -0.1)
    "starter_delta_down": 4.0,   # starter coming off the bench: already unbiased (-0.1 min); bigger cuts tested worse
    "b2b_starter_min": 1.0,      # 2026-11-10: night 2 of a back-to-back, starters -1 min (to the bench); better on
                                 #   b2b starters + 24+ min players all 3 seasons. opp b2b / defense-by-position: no gain, off
    "role_blend_k": 3.0,         # >0: lean on his own minutes in that role (pseudo-games); 0 = off
    "spread_min_k": 0.0,         # starter minutes cut per point of spread beyond spread_min_start
    "spread_min_start": 6.0,
    "team_minutes": 241.5,       # 240 + expected overtime
    "max_minutes": 34.0,         # 2026-11-12: was 36; 33+ projections ran 1.7 min high (3 seasons: better all, 24+ -0.04)
    "starter_min_floor": 20.0,   # 2026-11-12: confirmed starter projects >= 20 min (deep-bench spot starters ran far low)
    "role_jump_k": 2,            # 2026-11-12: last 2 games >= role_jump_min above his baseline = role change ...
    "role_jump_min": 6.0,
    "role_jump_w": 0.5,          # ... move halfway to those games (research/test_min_calibration.py)
    "fill_cap_mult": 1.5,        # a player can absorb freed minutes up to max(r * mult, r + add)
    "fill_cap_add": 8.0,
    "excess_starter_w": 0.25,     # when minutes must come back, starters give up less
    "high_min_start": 32.0,      # minutes above this are pulled back a little (blowouts, fouls)
    "high_min_shrink": 0.0,
    "pos_tau": 1.25,             # how strongly freed minutes stay at the same position
    "fill_power": 0.25,           # freed minutes split by (baseline minutes ^ power)
    "return_games": 5,           # missed this many team games in a row = returning
    "return_factor": 0.85,       # first game back: share of normal minutes
    "no_history_min": 8.0,       # minutes for a player with no NBA history
    "no_history_start_min": 24.0,
    # ---- per-minute rates ----
    "rate_half_life": 45.0,      # games
    "rate_last_season_w": 0.6,   # weight on last-season games relative to this season
    "rate_window": 82,
    "older_seasons": 1,          # 2026-11-12: missed all of last season -> use the season before for minutes + rates
    "rate_older_w": 0.4,         #   (Haliburton/Lillard/Kyrie were ~20 min "unknowns" on opening night; 2026-27 -0.018, 24+ -0.025)
    "rate_prior_scale": 0.25,    # multiplies rate_prior_minutes (how hard rates are pulled to the position average)
    "rate_prior_minutes": {"pts": 150, "fg3m": 250, "reb": 200, "ast": 200,
                           "stl": 500, "blk": 400, "tov": 300},
    # ---- game environment ----
    "pace_alpha": 0.5,           # counting stats scale with (expected game total / usual) ^ alpha
    "team_norm_beta": {"pts": 1.0, "fg3m": 0.3, "reb": 0.6, "ast": 0.8,
                       "stl": 0.2, "blk": 0.2, "tov": 0.3},   # 2026-11-08: leans on Vegas team totals
    "implied_other": 0.5,        # >0: non-scoring team totals also follow the Vegas team total (^ this)
    "team_half_life": 10.0,      # games, for a team's usual per-game totals
    "home_edge": 1.5,            # points, used only when there is no Vegas line
    # ---- double-double / triple-double spread (sd = c * sqrt(mean)) ----
    "sd_coef": {"pts": 1.6, "reb": 1.15, "ast": 1.15, "stl": 1.0, "blk": 1.1},
}

SCORING = {
    "DK": {"pts": 1, "fg3m": 0.5, "reb": 1.25, "ast": 1.5, "stl": 2, "blk": 2, "tov": -0.5,
           "dd": 1.5, "td": 3.0},
    "FD": {"pts": 1, "fg3m": 0, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1,
           "dd": 0, "td": 0},
    "Yahoo": {"pts": 1, "fg3m": 0.5, "reb": 1.2, "ast": 1.5, "stl": 3, "blk": 3, "tov": -1,
              "dd": 0, "td": 0},
}
