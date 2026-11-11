"""Straight comparison: our per-stat projections vs prop-implied averages for a night.
  python research/compare_props.py 2026-11-11"""
import sys, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import props
from config import OUTPUT

d = dt.date.fromisoformat(sys.argv[1])
f = sorted((OUTPUT / f"{d:%Y-%m-%d}").glob("projections_all_*.csv"))[-1]
df = pd.read_csv(f)
df = df[(df["Status"].fillna("") != "OUT") & (df["proj_min"] > 0)]
df, stamp = props.attach(df, d)
print(f"{f.name} vs props pulled {stamp}")
for s in ("pts", "reb", "ast", "fg3m", "stl", "blk"):
    b = df.dropna(subset=[f"prop_{s}"])
    gap = b[s] - b[f"prop_{s}"]
    print(f"  {s:4s} {len(b):3d} players | ours avg {b[s].mean():5.2f} props {b[f'prop_{s}'].mean():5.2f} | "
          f"we're {gap.mean():+.2f} | avg abs gap {gap.abs().mean():.2f} | corr {b[s].corr(b[f'prop_{s}']):.3f}")
b = df.dropna(subset=["prop_pts", "prop_reb", "prop_ast"]).copy()
b["gap_pts"] = b["pts"] - b["prop_pts"]
cols = ["name", "team", "proj_min", "starter_tonight", "pts", "prop_pts", "prop_line_pts", "gap_pts"]
print("\nbiggest points gaps (ours minus props):")
print(b.reindex(b["gap_pts"].abs().sort_values(ascending=False).index)[cols].head(15).round(1).to_string(index=False))
