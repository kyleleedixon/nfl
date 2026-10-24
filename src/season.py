"""The season the pipeline is on: the NFL season that most recently started (Sep-Dec: this year; Jan-Aug: last year).
`SEASON` in the environment overrides. Run as a script to print it (refresh.sh, season_close.sh)."""
import datetime
import os


def current(today=None):
    if os.environ.get("SEASON"):
        return int(os.environ["SEASON"])
    d = today or datetime.date.today()
    return d.year if d.month >= 9 else d.year - 1


if __name__ == "__main__":
    print(current())
