"""Check every team's hashtag against the team's own recent posts on X, and update src/hashtags.json.

For each team account (brand.AT): read its last 50 original posts (no retweets/replies) through the X API and count
hashtags, skipping matchup tags (#WASvsSF, #KCatSEA) and #NFL. Then:
- our tag appears                     -> confirmed: "checked" = SEASON
- ours is absent and another tag clearly took over (4+ uses) -> switch to it ("previous" keeps the old one)
- the account uses no slogan tags     -> kept, marked "unverified" (team accounts mostly tag matchups now; nothing to
                                         check against), so no warning
- the account couldn't be read        -> left alone; the run warns
Runs each August 1 and mid-September (preseason-check.yml). Reads cost a little (pay-per-use); about 1,600 per run.

usage: check_hashtags.py [--dry-run]
"""
import collections
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brand import AT, TAG_FILE
from season import current

API = "https://api.x.com/2"
KEYS = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"]
MATCHUP = re.compile(r"^[a-z]{2,4}(vs|at|v|x)[a-z]{2,4}$")
SKIP = {"nfl", "nflfootball", "nflplayoffs", "superbowl", "football"}


def session():
    from requests_oauthlib import OAuth1Session
    if all(os.environ.get(k) for k in KEYS):
        c = {k: os.environ[k].strip() for k in KEYS}
    else:
        env = os.path.expanduser("~/.config/telegraphed/x.env")
        c = dict(l.strip().split("=", 1) for l in open(env) if "=" in l and not l.lstrip().startswith("#"))
    return OAuth1Session(c["X_API_KEY"], c["X_API_SECRET"], c["X_ACCESS_TOKEN"], c["X_ACCESS_SECRET"])


def team_tags(x, handle):
    r = x.get(f"{API}/users/by/username/{handle.lstrip('@')}")
    r.raise_for_status()
    uid = r.json()["data"]["id"]
    t = x.get(f"{API}/users/{uid}/tweets", params={"max_results": 50, "tweet.fields": "entities", "exclude": "retweets,replies"})
    t.raise_for_status()
    count, spelled = collections.Counter(), {}
    for tw in t.json().get("data", []):
        for h in (tw.get("entities") or {}).get("hashtags") or []:
            k = h["tag"].lower()
            if k in SKIP or MATCHUP.match(k):
                continue
            count[k] += 1
            spelled.setdefault(k, collections.Counter())[h["tag"]] += 1
    return count, {k: v.most_common(1)[0][0] for k, v in spelled.items()}


def main(dry=False):
    S = current()
    data = json.load(open(TAG_FILE))
    x = session()
    report = []
    for team, v in data["teams"].items():
        try:
            count, spelled = team_tags(x, AT[team])
        except Exception as e:   # one account failing (renamed, suspended) shouldn't stop the rest
            report.append(f"{team}: couldn't read {AT[team]} ({e})")
            continue
        ours = v["tag"].lstrip("#").lower()
        top = count.most_common(3)
        tops = ", ".join(f"#{spelled[k]} {n}" for k, n in top) or "no hashtags"
        v.pop("unverified", None)
        if count[ours] >= 1:
            v["checked"] = S
            report.append(f"{team}: {v['tag']} confirmed ({tops})")
        elif top and top[0][1] >= 4 and count[ours] == 0:
            new = "#" + spelled[top[0][0]]
            v.update(previous=v["tag"], tag=new, checked=S)
            report.append(f"{team}: CHANGED {v['previous']} -> {new} ({tops})")
        else:
            v.update(checked=S, unverified=True)
            report.append(f"{team}: kept {v['tag']}, account uses no clear slogan tag ({tops})")
    print("\n".join(report))
    if not dry:
        json.dump(data, open(TAG_FILE, "w"), indent=1, ensure_ascii=False)
        open(TAG_FILE, "a").write("\n")
    bad = [r for r in report if "couldn't" in r]
    if bad and os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning::couldn't read {len(bad)} team account(s): {'; '.join(bad)}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
