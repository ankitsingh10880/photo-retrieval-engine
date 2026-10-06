"""
Second-pass coding for items labelled 0_channel_avoidance ("didn't search").
Adds `stage0_reason` to data/baseline_labelled.jsonl. Items where the user actually
searched and search failed are moved to 2_understanding (original kept in `stage_v1`).

Usage (from the engine folder):
  export ANTHROPIC_API_KEY=...
  python3 subcode_stage0.py          # asks for confirmation before spending
  python3 subcode_stage0.py --yes    # skip the confirmation prompt
Safe to rerun: only items without a valid stage0_reason are sent.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd

import pipeline as pl

PATH = Path(__file__).parent / "data" / "baseline_labelled.jsonl"
COST_PER_ITEM = 0.001  # USD, measured on the first full run
BATCH = 15
REASONS = {
    "browse_friction": "relies on scrolling, folders, albums or the timeline, and that browsing is hard (layout, order, scattered folders, hidden photos)",
    "search_entry_missing": "wanted to search but the search option, people search or filter is missing, moved or changed after an update",
    "distrust_search": "says search is useless/inaccurate in general and so does not use it, without describing a specific failed search",
    "search_tried_failed": "actually ran a search for a specific photo and search returned the wrong thing or nothing",
    "memory_not_searchable": "does not know what to search for, or thinks what they remember cannot be searched",
    "other": "none of the above",
}
PROMPT = ("You classify Google Photos user feedback. Each item was already judged to be about "
          "failing to find a photo without (successfully) using search. Pick the ONE reason that best fits:\n"
          + "\n".join(f"- {k}: {v}" for k, v in REASONS.items())
          + "\n\nReturn ONLY a JSON array of objects {\"id\": ..., \"stage0_reason\": ...}, one per item, same ids.")


def code_batch(client, batch):
    items = "\n\n".join(f"[{r.id}]\n{r.text[:1500]}" for r in batch.itertuples())
    for attempt in range(3):
        try:
            msg = client.messages.create(model=pl.LABEL_MODEL, max_tokens=1500, system=PROMPT,
                                         messages=[{"role": "user", "content": items}])
            out = pl._parse_json_array(pl.response_text(msg))
            return {str(o.get("id", "")).strip("[]"): o.get("stage0_reason") for o in out}
        except Exception:
            if attempt == 2:
                return {}
            import time
            time.sleep(3 * 2 ** attempt)


def run(client, df):
    batches = [df.iloc[i:i + BATCH] for i in range(0, len(df), BATCH)]
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, part in enumerate(ex.map(lambda b: code_batch(client, b), batches), 1):
            res.update(part)
            print(f"\r{i / len(batches):.0%}", end="", flush=True)
    print()
    return res


def main():
    if not PATH.exists():
        raise SystemExit(f"{PATH} not found - run build_baseline.py first.")
    d = pd.read_json(PATH, lines=True, convert_dates=False, dtype={"date": str})
    if "stage0_reason" not in d:
        d["stage0_reason"] = ""
    if "stage_v1" not in d:
        d["stage_v1"] = d["failure_stage"]
    d["stage0_reason"] = d["stage0_reason"].fillna("")
    target = d["is_retrieval"].astype(bool) & d["stage_v1"].eq("0_channel_avoidance")
    todo = d[target & ~d["stage0_reason"].isin(REASONS)]
    print(f"Stage-0 items: {int(target.sum())} | still to code: {len(todo)}")
    if todo.empty:
        return report(d)

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key.startswith("sk-ant-"):
        raise SystemExit("ANTHROPIC_API_KEY is missing or malformed (must start with 'sk-ant-').")
    client = anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(model=pl.LABEL_MODEL, max_tokens=5,
                               messages=[{"role": "user", "content": "ping"}])
    except anthropic.AuthenticationError:
        raise SystemExit("The API key was rejected.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"API check failed ({e.status_code}): {e.message}")

    print(f"Estimated cost: ${len(todo) * COST_PER_ITEM:.2f}")
    if "--yes" not in sys.argv and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Stopped before spending anything.")

    pilot = todo.head(BATCH)
    got = run(client, pilot)
    valid = sum(1 for i in pilot["id"] if got.get(i) in REASONS)
    if valid < 0.8 * len(pilot):
        raise SystemExit(f"Pilot batch failed validation ({valid}/{len(pilot)} valid) - nothing else spent.")
    got.update(run(client, todo.iloc[BATCH:]) if len(todo) > BATCH else {})

    for i, reason in got.items():
        if reason in REASONS:
            d.loc[d["id"].eq(i), "stage0_reason"] = reason
    moved = target & d["stage0_reason"].eq("search_tried_failed")
    d.loc[moved, "failure_stage"] = "2_understanding"
    d.loc[target & ~moved, "failure_stage"] = "0_channel_avoidance"
    d.to_json(PATH, orient="records", lines=True)
    missing = int((target & ~d["stage0_reason"].isin(REASONS)).sum())
    if missing:
        print(f"{missing} items not coded - rerun to retry only those.")
    report(d)


def report(d):
    t = d[d["is_retrieval"].astype(bool) & d["stage_v1"].eq("0_channel_avoidance")]
    print(t["stage0_reason"].value_counts().to_string())
    print("\nFailure stages after recoding:")
    print(pl.retrieval_only(d)["failure_stage"].value_counts().to_string())


if __name__ == "__main__":
    main()
