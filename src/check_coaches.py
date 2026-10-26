"""Keep src/headcoaches.csv and src/playcallers.csv current: who is each team's head coach, and who calls its offensive
plays, researched on the web by Claude (web search) with a source for every answer.

  audit    every team, this season (preseason: Aug 1 and Sep 15, preseason-check.yml); rewrites this season's rows
  changes  head-coach / play-caller changes announced in the last 10 days (weekly in season, coach-check.yml); adds rows

Two calls per batch: a research call with web search (free-form, cited), then a no-tools call that turns the research
into JSON against a schema (structured outputs can't be combined with web search citations). Answers without a source,
or marked low confidence, are reported and left out. Needs ANTHROPIC_API_KEY (GitHub secret); without it, exits 0 with
a warning. Roughly $1-3 per audit and well under $1 per weekly check (web searches + tokens).

usage: check_coaches.py audit|changes [--dry-run]
"""
import csv
import datetime as dt
import io
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brand import NICK
from season import current

SRC = os.path.dirname(os.path.abspath(__file__))
CALLERS, HCS = f"{SRC}/playcallers.csv", f"{SRC}/headcoaches.csv"
REPORT = "state/coach_check.json"
MODEL = "claude-opus-5-5"
TEAMS = sorted(NICK)
FULL = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
        "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
        "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers", "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
        "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks",
        "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}

SYSTEM = ("You verify NFL coaching staffs for a data pipeline. Use web search. Prefer team announcements, ESPN, NFL.com, "
          "the AP, Pro Football Talk and team beat reporters; prefer the most recent reporting. 'Calls the offensive plays' "
          "means who holds the play sheet and sends in the calls on game day, which is often not the head coach. Give a "
          "source URL for every answer. If reporting is unclear or conflicting, say so plainly rather than guessing.")

PERSON = {"type": "object", "additionalProperties": False, "required": ["from_week", "name"],
          "properties": {"from_week": {"type": "integer"}, "name": {"type": "string"}}}
AUDIT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["teams"],
    "properties": {"teams": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["team", "head_coaches", "play_callers", "confidence", "sources"],
        "properties": {
            "team": {"type": "string", "enum": TEAMS},
            "head_coaches": {"type": "array", "items": PERSON},
            "play_callers": {"type": "array", "items": PERSON},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "sources": {"type": "array", "items": {"type": "string"}}}}}}}
CHANGE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["changes"],
    "properties": {"changes": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["team", "kind", "name", "from_week", "confidence", "source"],
        "properties": {
            "team": {"type": "string", "enum": TEAMS},
            "kind": {"type": "string", "enum": ["head_coach", "play_caller"]},
            "name": {"type": "string"},
            "from_week": {"type": "integer"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "source": {"type": "string"}}}}}}


def client():
    import anthropic
    return anthropic.Anthropic()


def research(c, prompt, max_searches):
    """One research turn with web search; resumes server-side pause_turn. Returns the final text."""
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_searches},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_searches}]
    messages = [{"role": "user", "content": prompt}]
    for _ in range(6):
        with c.beta.messages.stream(model=MODEL, max_tokens=32000, system=SYSTEM, messages=messages, tools=tools,
                                    output_config={"effort": "high"},
                                    betas=["server-side-fallback-2026-07-01"], fallbacks="default") as s:
            r = s.get_final_message()
        if r.stop_reason == "refusal":
            raise RuntimeError(f"research declined: {r.stop_details}")
        if r.stop_reason != "pause_turn":
            return "\n".join(b.text for b in r.content if b.type == "text")
        messages = [messages[0], {"role": "assistant", "content": r.content}]
    raise RuntimeError("research didn't finish after 6 continuations")


def extract(c, research_text, schema, instruction):
    r = c.messages.create(model=MODEL, max_tokens=16000, output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                          messages=[{"role": "user", "content": f"{instruction}\n\nResearch notes:\n{research_text}"}])
    if r.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"extraction stopped: {r.stop_reason}")
    return json.loads(next(b.text for b in r.content if b.type == "text"))


def read(path, fields):
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path)))


def write(path, rows, fields):
    rows = sorted(rows, key=lambda r: (r["team"], int(r["season"]), int(r["from_week"])))
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    open(path, "w").write(out.getvalue())


CF = ["season", "team", "head_coach", "from_week", "caller", "confidence"]
HF = ["season", "team", "from_week", "head_coach", "source"]


def apply_audit(S, teams):
    """Replace this season's rows for every team answered with a source and at least medium confidence."""
    callers, hcs = read(CALLERS, CF), read(HCS, HF)
    done, skipped = [], []
    for t in teams:
        if t["confidence"] == "low" or not t["sources"] or not t["play_callers"] or not t["head_coaches"]:
            skipped.append(t["team"])
            continue
        callers = [r for r in callers if not (r["season"] == str(S) and r["team"] == t["team"])]
        hcs = [r for r in hcs if not (r["season"] == str(S) and r["team"] == t["team"])]
        for p in sorted(t["play_callers"], key=lambda p: p["from_week"]):
            callers.append(dict(season=S, team=t["team"], head_coach="", from_week=max(1, p["from_week"]), caller=p["name"].strip(),
                                confidence=t["confidence"]))
        for p in sorted(t["head_coaches"], key=lambda p: p["from_week"]):
            hcs.append(dict(season=S, team=t["team"], from_week=max(1, p["from_week"]), head_coach=p["name"].strip(), source=t["sources"][0]))
        done.append(t["team"])
    return callers, hcs, done, skipped


def apply_changes(S, changes):
    callers, hcs = read(CALLERS, CF), read(HCS, HF)
    done, skipped = [], []
    for ch in changes:
        if ch["confidence"] == "low" or not ch["source"].startswith("http"):
            skipped.append(f"{ch['team']} {ch['kind']} {ch['name']}")
            continue
        wk = max(1, ch["from_week"])
        if ch["kind"] == "play_caller":
            rows = [r for r in callers if r["season"] == str(S) and r["team"] == ch["team"] and int(r["from_week"]) <= wk]
            if rows and sorted(rows, key=lambda r: int(r["from_week"]))[-1]["caller"] == ch["name"]:
                continue   # already known
            callers = [r for r in callers if not (r["season"] == str(S) and r["team"] == ch["team"] and r["from_week"] == str(wk))]
            callers.append(dict(season=S, team=ch["team"], head_coach="", from_week=wk, caller=ch["name"].strip(), confidence=ch["confidence"]))
        else:
            rows = [r for r in hcs if r["season"] == str(S) and r["team"] == ch["team"] and int(r["from_week"]) <= wk]
            if rows and sorted(rows, key=lambda r: int(r["from_week"]))[-1]["head_coach"] == ch["name"]:
                continue
            hcs = [r for r in hcs if not (r["season"] == str(S) and r["team"] == ch["team"] and r["from_week"] == str(wk))]
            hcs.append(dict(season=S, team=ch["team"], from_week=wk, head_coach=ch["name"].strip(), source=ch["source"]))
        done.append(f"{ch['team']} {ch['kind']}: {ch['name']} from week {wk}")
    return callers, hcs, done, skipped


def current_table(S):
    """Who we think holds each job now (for the changes prompt)."""
    callers, hcs = read(CALLERS, CF), read(HCS, HF)
    last = lambda rows, t, f: (sorted([r for r in rows if r["season"] == str(S) and r["team"] == t], key=lambda r: int(r["from_week"])) or [{f: "?"}])[-1][f]
    return "\n".join(f"{t} ({FULL[t]}): head coach {last(hcs, t, 'head_coach')}, play-caller {last(callers, t, 'caller')}" for t in TEAMS)


def main(mode, dry):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        msg = "ANTHROPIC_API_KEY not set: coach check skipped (add it under GitHub > Settings > Secrets > Actions)"
        print(f"::warning::{msg}" if os.environ.get("GITHUB_ACTIONS") else msg)
        return
    S, today = current(), dt.date.today().isoformat()
    c = client()
    if mode == "audit":
        teams = []
        for i in range(0, 32, 8):
            batch = TEAMS[i:i + 8]
            notes = research(c, f"Today is {today}; the {S} NFL season. For each team below: who is the head coach now, and who "
                                f"calls the offensive plays? If either changed during the {S} regular season, give the week the "
                                f"new person took over (the first game they ran), and who held it before. Before the regular "
                                f"season starts, everything is week 1.\n" + "\n".join(f"- {FULL[t]} ({t})" for t in batch),
                             max_searches=24)
            got = extract(c, notes, AUDIT_SCHEMA, f"Turn these notes into one entry per team ({', '.join(batch)}). head_coaches and "
                          "play_callers list each person with the regular-season week they started (1 for the season opener), in "
                          "order. Use the team codes given. confidence: low if the notes say reporting is unclear or unsourced.")
            teams += [t for t in got["teams"] if t["team"] in batch]
        callers, hcs, done, skipped = apply_audit(S, teams)
        report = dict(mode=mode, season=S, date=today, updated=done, skipped=skipped, teams=teams)
    else:
        notes = research(c, f"Today is {today}; the {S} NFL season. Our current table:\n{current_table(S)}\n\nSearch for any NFL "
                            f"head-coach changes (firings, interim coaches) or changes in who calls a team's offensive plays that were "
                            f"announced in the last 10 days, or anything in the table that is wrong. For each, give the team, the new "
                            f"person, the regular-season week they take over (their first game), and a source URL. If nothing "
                            f"changed, say so.", max_searches=15)
        got = extract(c, notes, CHANGE_SCHEMA, "List each change in these notes (empty list if none). kind is head_coach or "
                      "play_caller; from_week is the regular-season week the new person's first game is in.")
        callers, hcs, done, skipped = apply_changes(S, got["changes"])
        report = dict(mode=mode, season=S, date=today, updated=done, skipped=skipped, changes=got["changes"])
    print(f"{mode}: updated {len(done)}; skipped (low confidence / no source): {skipped or 'none'}")
    for d in done:
        print("  ", d)
    if not dry:
        write(CALLERS, callers, CF)
        write(HCS, hcs, HF)
        os.makedirs("state", exist_ok=True)
        json.dump(report, open(REPORT, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], "--dry-run" in sys.argv)
