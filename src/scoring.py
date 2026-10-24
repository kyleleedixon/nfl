"""What counts toward a public read rate (game cards, weekly rankings).

RPOs are not scored: the play-caller calls "RPO" and the QB picks pass or run after the snap, so the outcome isn't the
play-caller's decision. FTN tags them (2022+). They stay on the card as unscored marks. Play action is scored: that is
the play-caller choosing a pass and disguising it.
"""
import os
import pandas as pd

D = os.environ.get("DATA", "data")
NOTE = "RPOs not scored: the QB picks pass or run after the snap."


def mark_rpo(P, season):
    """Adds a boolean `rpo` column (False where FTN hasn't charted the play)."""
    f = pd.read_parquet(f"{D}/raw/ftn_{season}.parquet", columns=["nflverse_game_id", "nflverse_play_id", "is_rpo"])
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}).drop_duplicates(["game_id", "play_id"])
    f["play_id"] = f.play_id.astype(float)
    P = P.drop(columns="rpo", errors="ignore").assign(play_id=P.play_id.astype(float))
    P = P.merge(f, on=["game_id", "play_id"], how="left")
    return P.assign(rpo=P.pop("is_rpo").astype("boolean").fillna(False).astype(bool))
