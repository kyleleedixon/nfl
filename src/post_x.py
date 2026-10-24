"""Post the weekly tweet to X (@TelegraphedNFL) with the card image (any further tweets in the file go as replies).

Usage: python src/post_x.py [--dry-run] [season week]   (default: newest weekly report not yet posted)
       python src/post_x.py games [--dry-run]            (next post: a due scheduled post, else the oldest game card;
                                                          one per run so they're spaced out)
       python src/post_x.py --check                      (verify credentials only; posts nothing)
Reads state/threads/{season}_wk{week:02d}.json (weekly_report.py), state/games/{game_id}.json (game_cards.py) or
state/queue/{key}.json (scheduled.py: prime-time previews, weekly surprises; each has post_at / expires).
Credentials: X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET from the environment (GitHub Actions secrets)
or ~/.config/telegraphed/x.env locally; never in the repo.
Safe to re-run: every posted tweet is recorded in state/x_posted.json (committed), so a retry finishes a partial thread
instead of starting a new one. A file state/x_paused stops all posting. Won't auto-post a report older than 2 days.
"""
import glob
import json
import os
import sys
import time

ST = "state"
ENV = os.path.expanduser("~/.config/telegraphed/x.env")
LOG = f"{ST}/x_posted.json"
KEYS = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"]
API = "https://api.x.com/2"



def pd_ts(day):
    """'2026-10-22' -> epoch seconds (no pandas needed in the posting job)."""
    import calendar
    return calendar.timegm(time.strptime(day, "%Y-%m-%d"))


def fail(msg):
    """Exit with an error; in GitHub Actions also emit it as an annotation (readable on the run page)."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{msg}")
    sys.exit(msg)


def session():
    from requests_oauthlib import OAuth1Session
    if os.environ.get("GITHUB_ACTIONS"):
        missing = [k for k in KEYS if not os.environ.get(k)]
        if missing:
            fail(f"missing GitHub secrets: {', '.join(missing)} (Settings > Secrets and variables > Actions > Repository secrets)")
    if all(os.environ.get(k) for k in KEYS):
        cred = {k: os.environ[k].strip() for k in KEYS}
    else:
        cred = dict(l.strip().split("=", 1) for l in open(ENV) if "=" in l and not l.lstrip().startswith("#"))
    return OAuth1Session(cred["X_API_KEY"], cred["X_API_SECRET"], cred["X_ACCESS_TOKEN"], cred["X_ACCESS_SECRET"])


if "--check" in sys.argv:
    r = session().get(f"{API}/users/me")
    if r.status_code != 200 or r.headers.get("x-access-level") != "read-write":
        fail(f"X credentials rejected: HTTP {r.status_code}, access={r.headers.get('x-access-level')}: {' '.join(r.text.split())[:200]}")
    sys.exit(print(f"credentials ok: @{r.json()['data']['username']}, {r.headers['x-access-level']}") or 0)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dry = "--dry-run" in sys.argv
posted = json.load(open(LOG)) if os.path.exists(LOG) else {}
games = bool(args) and args[0] == "games"
DIR = f"{ST}/games" if games else f"{ST}/threads"

if games:
    done = {k for k, v in posted.items() if v.get("complete")}
    now = time.time()
    # scheduled posts (previews before kickoff, surprises) go first when due; a game card never delays them
    sq = [json.load(open(p)) for p in glob.glob(f"{ST}/queue/*.json")]
    due = sorted((x for x in sq if x["key"] not in done and x["post_at"] <= now < x["expires"]), key=lambda x: x["post_at"])
    q = [json.load(open(p)) for p in glob.glob(f"{DIR}/*.json")]
    fresh = lambda g: now - pd_ts(g["kickoff"]) < 6 * 86400   # a game card a week late isn't worth posting
    q = sorted((g for g in q if g["game_id"] not in done and fresh(g)), key=lambda g: (g["kickoff"], g["game_id"]))
    if due:
        DIR, key = f"{ST}/queue", due[0]["key"]
    elif q:
        key = q[0]["game_id"]
    else:
        sys.exit(print("no game cards queued") or 0)
elif args:
    key = f"{int(args[0])}_wk{int(args[1]):02d}"
else:
    done = {k for k, v in posted.items() if v.get("complete")}
    todo = sorted(os.path.basename(p)[:-len(".json")] for p in glob.glob(f"{ST}/threads/*.json"))
    todo = [k for k in todo if k not in done]
    if not todo:
        sys.exit(print("nothing new to post") or 0)
    key = todo[-1]
T = json.load(open(f"{DIR}/{key}.json"))
if not args and not dry and time.time() - T.get("created", 0) > 2 * 86400:
    fail(f"newest report {key} is over 2 days old (did the refresh fail?), not posting a stale thread")
rec = posted.setdefault(key, {"ids": []})
if rec.get("complete"):
    sys.exit(print(f"{key} already posted: https://x.com/TelegraphedNFL/status/{rec['ids'][0]}") or 0)
if os.path.exists(f"{ST}/x_paused") and not dry:
    sys.exit(print(f"{ST}/x_paused exists, not posting") or 0)
for i, t in enumerate(T["tweets"]):
    assert len(t) <= 280, f"tweet {i + 1} is {len(t)} chars"

if dry:
    print(f"[dry run] {key}: {len(T['tweets'])} tweets, image {T['card']}")
    for i, t in enumerate(T["tweets"], 1):
        print(f"\n--- {i}/{len(T['tweets'])}{' (+ image)' if i == 1 else ''}{' (already posted)' if i <= len(rec['ids']) else ''}\n{t}")
    sys.exit(0)

x = session()


def ok(r):
    if r.status_code >= 300:
        fail(f"X API {r.request.method} {r.url} -> {r.status_code}: {' '.join(r.text.split())[:400]}")
    return r.json()


def save():
    json.dump(posted, open(LOG, "w"), indent=1)


media = None
if not rec["ids"]:
    with open(T["card"], "rb") as f:
        m = ok(x.post(f"{API}/media/upload", files={"media": f}, data={"media_category": "tweet_image"}))
    media = (m.get("data") or {}).get("id") or m.get("media_id_string")
for i, text in enumerate(T["tweets"]):
    if i < len(rec["ids"]):
        continue
    body = {"text": text}
    if i == 0:
        body["media"] = {"media_ids": [media]}
    else:
        body["reply"] = {"in_reply_to_tweet_id": rec["ids"][-1]}
    rec["ids"].append(ok(x.post(f"{API}/tweets", json=body))["data"]["id"])
    save()
    print(f"posted {i + 1}/{len(T['tweets'])}")
rec["complete"] = True
save()
print(f"live: https://x.com/TelegraphedNFL/status/{rec['ids'][0]}")
if T.get("tag_reply"):
    print("\nNow reply to it from the X app with (manual tags are allowed, automated ones aren't):\n\n" + T["tag_reply"])
