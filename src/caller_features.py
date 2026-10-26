"""Adds the play-caller's habits next to the head coach's: caller_* columns in data/features_v3.parquet.

Builds the caller-keyed feature passes (features.py with CALLER=1, then v2 and v3, into data/caller/; src/callers.py says
who called the plays) and merges their coach-keyed columns, renamed caller_*, plus caller_is_hc, into the head-coach table.
Experiment only (not in production): the live model gained +0.0003 [-0.0002, +0.0007] once the participation-derived
caller_pf_proe was dropped. Run after features_v3.py; idempotent.

usage: caller_features.py [raw_dir]
"""
import os
import subprocess
import sys
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
D = os.environ.get("DATA", "data")
SRC = os.path.dirname(os.path.abspath(__file__))
COACH = ["coach_dd_proe", "coach_neutral_pass", "coach_pf_proe", "coach_proe", "coach_proe_after_fail",
         "coach_proe_after_pass", "coach_proe_after_run", "coach_switch_rate"]


def main():
    os.makedirs(f"{D}/caller", exist_ok=True)
    py, env = sys.executable, {**os.environ, "CALLER": "1"}
    subprocess.run([py, f"{SRC}/features.py", RAW, f"{D}/caller/features.parquet"], check=True, env=env, stdout=subprocess.DEVNULL)
    subprocess.run([py, f"{SRC}/features_v2.py", RAW, f"{D}/caller/features.parquet", f"{D}/caller/features_v2.parquet"],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([py, f"{SRC}/features_v3.py", RAW, f"{D}/caller/features_v2.parquet", f"{D}/caller/features_v3.parquet"],
                   check=True, stdout=subprocess.DEVNULL)
    hc = pd.read_parquet(f"{D}/features_v3.parquet")
    hc = hc.drop(columns=[c for c in hc.columns if c.startswith("caller")])
    ca = pd.read_parquet(f"{D}/caller/features_v3.parquet", columns=["game_id", "play_id", "coach"] + COACH)
    ca = ca.rename(columns={"coach": "_caller", **{c: "caller" + c[len("coach"):] for c in COACH}})
    out = hc.merge(ca, on=["game_id", "play_id"], how="left")
    out["caller_is_hc"] = (out["_caller"] == out["coach"]).astype(float)
    out = out.drop(columns="_caller")
    out.to_parquet(f"{D}/features_v3.parquet", index=False)
    t = out[out.is_target == 1]
    print(f"caller features added; caller isn't the head coach on {1 - t.caller_is_hc.mean():.0%} of plays")


if __name__ == "__main__":
    main()
