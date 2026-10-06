"""
Sonnet re-check of the Haiku labels (Task 13A).

Why: Haiku over-includes general complaints as "about finding a photo", so the retrieval count
and stage split are inflated. Sonnet re-labels is_retrieval and failure_stage with a stricter
R.T.C.F. prompt and few-shot examples (written for this script, NOT taken from the 40 gold rows),
at temperature 0. Ankit's 40 hand labels (data/gold_sample.csv) are the answer key.

Steps (run from engine/):
  export ANTHROPIC_API_KEY=...
  python3 recheck_relevance.py --pilot    # 40 gold rows (~$0.15): agreement table + go/no-go
  python3 recheck_relevance.py            # all Haiku "retrieval" items (~$3): only after a "go"
  python3 recheck_relevance.py --apply    # backs up the Haiku file, then makes the re-checked
                                          # labels the ones the app and validate.py read
Outputs (data/):
  recheck_pilot.json        pilot agreement table and verdict
  recheck_cache.jsonl       every Sonnet answer, by id (reruns skip ids already done)
  baseline_rechecked.jsonl  full labelled corpus with Sonnet's labels on the re-checked items
                            (haiku_is_retrieval / haiku_failure_stage / sonnet_reason kept)
Never modifies baseline_labelled.jsonl except with --apply, which first saves
baseline_labelled_haiku.jsonl.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

import pipeline as pl

D = Path(__file__).parent / "data"
BASE, GOLD = D / "baseline_labelled.jsonl", D / "gold_sample.csv"
CACHE, PILOT_OUT, OUT = D / "recheck_cache.jsonl", D / "recheck_pilot.json", D / "baseline_rechecked.jsonl"
BACKUP = D / "baseline_labelled_haiku.jsonl"

MODEL = pl.SYNTH_MODEL  # the Sonnet model the engine already uses
BATCH = 10
COST_PER_ITEM = 0.004   # USD, deliberately generous; the run prints real token usage
ERRORS: list[str] = []  # first errors seen, printed if a run fails
USE_TEMPERATURE = [False]  # claude-sonnet-5-5 rejects temperature ("deprecated for this model"), so it is never sent
PASS_RETRIEVAL = 80.0   # % is_retrieval agreement for a clean "go"
HAIKU_FLOOR = 67.5      # Haiku's is_retrieval agreement; at or below this -> stop

STAGES = [s for s in pl.STAGES if s != "unclear"]

PROMPT = f"""ROLE: You are a senior UX researcher coding Google Photos user feedback for a study on how
people retrieve photos they remember only vaguely. You are strict and literal.

TASK: For EACH numbered item decide (1) is_retrieval and (2) failure_stage.

CONTEXT (definitions):
is_retrieval = true ONLY when the text is about LOCATING photos that exist in the person's own
library: trying to find a specific photo, or finding/searching/scrolling/browsing being slow,
inaccurate or failing (including general complaints that search doesn't work or takes forever).
is_retrieval = false for: praise with no difficulty; photos missing, deleted, not synced or not
backed up; storage, backup, pricing, subscription, sharing, editing, collages, Memories/resurfacing,
printing, UI or performance complaints; face grouping, albums or organising UNLESS the text says
it made finding a photo hard; feature requests not about finding a photo.
failure_stage = the EARLIEST stage where retrieval broke (only when is_retrieval is true):
  0_channel_avoidance: didn't search (scrolled, folders, gave up) or says search is useless
  1_expression: knew things about the photo but couldn't turn them into a query
  2_understanding: gave a reasonable clue; search ignored or misread it
  3_evaluation: results came back but too many/too similar to spot the target
  4_refinement: first try failed and they couldn't narrow or adjust
If is_retrieval is false, failure_stage = "not_retrieval". Code only what the text states or
directly implies. If unsure whether it is retrieval, choose false.

EXAMPLES (illustrative, not from the data):
- "Love that I can find any picture just by typing what's in it!" -> false, not_retrieval
- "After the update half my photos are gone and I can't find them anywhere" -> false, not_retrieval (missing photos)
- "Memories keeps showing my ex, let me turn it off" -> false, not_retrieval
- "Backup is stuck at 98% and I'm out of storage" -> false, not_retrieval
- "Needed my licence photo at the counter and scrolled for ten minutes" -> true, 0_channel_avoidance
- "Search is useless so I just scroll through years of photos" -> true, 0_channel_avoidance
- "I know it was a red car at a wedding but I have no idea what to type" -> true, 1_expression
- "Searched 'passport' and it showed random documents, not mine" -> true, 2_understanding
- "Typing 'beach' gives me 2,000 photos and I can't spot the one I want" -> true, 3_evaluation
- "First search didn't find it and there's no way to add a year to narrow it, so I gave up" -> true, 4_refinement

FORMAT: Return ONLY a JSON array, one object per item, same ids, no prose:
[{{"id": "<id>", "is_retrieval": true|false, "failure_stage": one of {STAGES}, "reason": "<max 12 words>"}}]"""


# ---------------------------------------------------------------- model calls
def _batch(client, rows: list[dict], usage: dict, retries=3) -> list[dict]:
    items = "\n\n".join(f"[{r['id']}]\n{str(r['text'])[:1500]}" for r in rows)
    for attempt in range(retries):
        raw = ""
        try:
            kw = {"temperature": 0} if USE_TEMPERATURE[0] else {}
            msg = client.messages.create(model=MODEL, max_tokens=4000, system=PROMPT,
                                         messages=[{"role": "user", "content": items}], **kw)
            raw = pl.response_text(msg)
            u = getattr(msg, "usage", None)
            if u is not None:
                usage["in"] += getattr(u, "input_tokens", 0) or 0
                usage["out"] += getattr(u, "output_tokens", 0) or 0
            got = {str(o.get("id", "")).strip("[] "): o for o in pl._parse_json_array(raw)}
            out = []
            for r in rows:
                o = got.get(r["id"])
                if o is None:
                    raise ValueError(f"id {r['id']} missing from the answer")
                isr = o.get("is_retrieval")
                isr = isr is True or (isinstance(isr, str) and isr.strip().lower() == "true")
                st = o.get("failure_stage") if o.get("failure_stage") in STAGES else "unclear"
                if not isr:
                    st = "not_retrieval"
                elif st == "not_retrieval":
                    st = "unclear"
                out.append({"id": r["id"], "is_retrieval": isr, "failure_stage": st,
                            "reason": str(o.get("reason", ""))[:120]})
            return out
        except Exception as e:
            text = f"{type(e).__name__}: {e}"
            if "temperature" in text.lower() and USE_TEMPERATURE[0]:
                USE_TEMPERATURE[0] = False  # this model doesn't accept it; retry without
                continue
            if len(ERRORS) < 3 and "temperature" not in text.lower():
                ERRORS.append(text[:400] + (f" | answer began: {raw[:300]!r}" if raw else ""))
            if attempt == retries - 1:
                return [{"id": r["id"], "error": True} for r in rows]
            time.sleep(3 * 2 ** attempt)


def run(client, rows: list[dict], workers=4, progress=True) -> tuple[dict, dict]:
    """Label rows not already in the cache; returns ({id: answer}, usage)."""
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            o = json.loads(line)
            if not o.get("error"):
                cache[o["id"]] = o
    todo = [r for r in rows if r["id"] not in cache]
    usage = {"in": 0, "out": 0}
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    with CACHE.open("a") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(lambda b: _batch(client, b, usage), batches), 1):
            for o in res:
                f.write(json.dumps(o) + "\n")
                if not o.get("error"):
                    cache[o["id"]] = o
            if progress:
                print(f"\r{k}/{len(batches)} batches", end="", flush=True)
    if progress and batches:
        print()
    if ERRORS:
        print("First errors seen:\n  " + "\n  ".join(ERRORS))
    pass
    return cache, usage


def client_or_exit(n_items: int):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key.startswith("sk-ant-"):
        raise SystemExit("ANTHROPIC_API_KEY missing or malformed. Run: export ANTHROPIC_API_KEY=\"sk-ant-...\"")
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    try:
        client.messages.create(model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "ping"}])
    except anthropic.AuthenticationError:
        raise SystemExit("The API key was rejected.")
    except anthropic.APIStatusError as e:
        raise SystemExit(f"API check failed ({e.status_code}): {e.message}")
    print(f"Model: {MODEL} | items to label: {n_items} | estimated cost: ${n_items * COST_PER_ITEM:.2f}")
    if n_items and "--yes" not in sys.argv and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Stopped before spending anything.")
    return client


def tokens_line(usage):
    # list price assumption for a Sonnet-class model: $3 / M input, $15 / M output
    est = usage["in"] * 3e-6 + usage["out"] * 15e-6
    return f"Tokens used: {usage['in']:,} in / {usage['out']:,} out (≈ ${est:.2f} at $3/$15 per M; check the console balance)"


def load_base() -> pd.DataFrame:
    if not BASE.exists():
        raise SystemExit("data/baseline_labelled.jsonl not found - run from the engine/ folder.")
    return pd.read_json(BASE, lines=True, convert_dates=False, dtype={"date": str})


def norm_bool(v) -> str:
    return "true" if (v is True or str(v).strip().lower() == "true") else "false"


# ---------------------------------------------------------------- pilot
def pilot():
    gold = pd.read_csv(GOLD, dtype=str).fillna("")
    gold = gold[gold["is_retrieval"].str.strip().ne("")]
    if gold.empty:
        raise SystemExit("gold_sample.csv has no hand labels.")
    base = load_base().set_index("id")
    missing = [i for i in gold["id"] if i not in base.index]
    if missing:
        raise SystemExit(f"{len(missing)} gold ids are not in baseline_labelled.jsonl (e.g. {missing[0]}).")

    client = client_or_exit(sum(1 for i in gold["id"] if not _cached(i)))
    ans, usage = run(client, gold[["id", "text"]].to_dict("records"))
    failed = [i for i in gold["id"] if i not in ans]
    if failed:
        raise SystemExit(f"{len(failed)} items failed - rerun to retry only those.")

    g_ret = gold["is_retrieval"].map(norm_bool)
    g_st = gold["failure_stage"].str.strip().str.lower()
    h_ret = gold["id"].map(lambda i: norm_bool(base.at[i, "is_retrieval"]))
    h_st = gold["id"].map(lambda i: str(base.at[i, "failure_stage"]).lower())
    s_ret = gold["id"].map(lambda i: norm_bool(ans[i]["is_retrieval"]))
    s_st = gold["id"].map(lambda i: ans[i]["failure_stage"])
    # the full run only re-checks Haiku's positives, so the pipeline result is:
    # Haiku said not-retrieval -> keep Haiku; Haiku said retrieval -> Sonnet's answer
    p_ret = pd.Series([s if h == "true" else h for h, s in zip(h_ret, s_ret)], index=gold.index)
    p_st = pd.Series([s if h == "true" else hs for h, s, hs in zip(h_ret, s_st, h_st)], index=gold.index)
    st_mask = g_st.ne("")

    def row(name, r, s):
        return {"labeller": name,
                "is_retrieval_agreement_pct": round(100 * (r == g_ret).mean(), 1),
                "failure_stage_agreement_pct": round(100 * (s[st_mask] == g_st[st_mask]).mean(), 1),
                "over_includes": int(((r == "true") & (g_ret == "false")).sum()),
                "under_includes": int(((r == "false") & (g_ret == "true")).sum())}

    table = [row("Haiku (current)", h_ret, h_st), row("Sonnet alone", s_ret, s_st),
             row("Pipeline (Haiku, then Sonnet re-check)", p_ret, p_st)]
    pipe, haiku = table[2], table[0]
    if pipe["is_retrieval_agreement_pct"] >= PASS_RETRIEVAL and pipe["over_includes"] < haiku["over_includes"]:
        verdict = "GO: validated (>= 80% and fewer over-includes than Haiku)"
    elif pipe["is_retrieval_agreement_pct"] > HAIKU_FLOOR:
        verdict = "GO, but report as an improvement over Haiku, not as validated"
    else:
        verdict = "STOP: no better than Haiku. Do not run the full re-check; tell Claude."
    disagreements = [{"id": i, "gold": f"{a}/{b}", "pipeline": f"{c}/{d}", "sonnet_reason": ans[i]["reason"]}
                     for i, a, b, c, d in zip(gold["id"], g_ret, g_st, p_ret, p_st) if a != c or (b and b != d)]
    PILOT_OUT.write_text(json.dumps({"n": len(gold), "table": table, "verdict": verdict,
                                     "disagreements": disagreements, "model": MODEL}, indent=2))
    print(pd.DataFrame(table).to_string(index=False))
    print(f"\nVerdict: {verdict}")
    print(f"{len(disagreements)} rows where the pipeline still differs from your labels (see {PILOT_OUT.name}).")
    print(tokens_line(usage))


def _cached(i) -> bool:
    if not CACHE.exists():
        return False
    return any(json.loads(l).get("id") == i and not json.loads(l).get("error") for l in CACHE.read_text().splitlines())


# ---------------------------------------------------------------- full run
def full():
    if not PILOT_OUT.exists():
        raise SystemExit("Run the pilot first: python3 recheck_relevance.py --pilot")
    verdict = json.loads(PILOT_OUT.read_text())["verdict"]
    if verdict.startswith("STOP") and "--force" not in sys.argv:
        raise SystemExit(f"Pilot verdict was: {verdict}\nNot running the full re-check.")
    base = load_base()
    pos = base[base["is_retrieval"].map(lambda v: norm_bool(v) == "true")].copy()  # same scope as the pilot: everything Haiku marked retrieval
    cached = set()
    if CACHE.exists():
        cached = {json.loads(l)["id"] for l in CACHE.read_text().splitlines() if not json.loads(l).get("error")}
    todo = [r for r in pos[["id", "text"]].to_dict("records") if r["id"] not in cached]
    print(f"Haiku 'retrieval' items: {len(pos)} | already re-checked: {len(pos) - len(todo)}")
    client = client_or_exit(len(todo))
    ans, usage = run(client, pos[["id", "text"]].to_dict("records"), workers=6)
    failed = [i for i in pos["id"] if i not in ans]

    out = base.copy()
    out["haiku_is_retrieval"] = out["is_retrieval"]
    out["haiku_failure_stage"] = out["failure_stage"]
    out["sonnet_reason"] = ""
    out["rechecked"] = False
    for k in out.index[out["id"].isin(ans.keys()) & out["id"].isin(pos["id"])]:
        a = ans[out.at[k, "id"]]
        out.at[k, "is_retrieval"] = bool(a["is_retrieval"])
        out.at[k, "failure_stage"] = a["failure_stage"]
        out.at[k, "sonnet_reason"] = a["reason"]
        out.at[k, "rechecked"] = True
    out["is_retrieval"] = out["is_retrieval"].map(lambda v: norm_bool(v) == "true")
    out.to_json(OUT, orient="records", lines=True)

    old = pl.retrieval_only(base.assign(is_retrieval=base["is_retrieval"].map(lambda v: norm_bool(v) == "true")))
    new = pl.retrieval_only(out)
    split = pd.DataFrame({"haiku": old["failure_stage"].value_counts(),
                          "after_recheck": new["failure_stage"].value_counts()}).fillna(0).astype(int)
    split["after_%"] = (100 * split["after_recheck"] / max(len(new), 1)).round(1)
    print(f"\nAbout finding a photo: Haiku {len(old)} -> after re-check {len(new)} "
          f"({len(old) - len(new)} removed as not retrieval)")
    print(split.to_string())
    if failed:
        print(f"\n{len(failed)} items failed - rerun this command to retry only those (nothing is re-billed).")
    print(f"\nSaved {OUT.name}. Your original labels are untouched. Next: python3 recheck_relevance.py --apply")
    print(tokens_line(usage))


# ---------------------------------------------------------------- apply
def apply():
    if not OUT.exists():
        raise SystemExit("Nothing to apply - run the full re-check first.")
    if not BACKUP.exists():
        shutil.copy(BASE, BACKUP)
        print(f"Backed up the Haiku labels to {BACKUP.name}")
    shutil.copy(OUT, BASE)
    print("baseline_labelled.jsonl now holds the re-checked labels. Run: python3 validate.py")
    print(f"To undo: cp data/{BACKUP.name} data/{BASE.name}")


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        pilot()
    elif "--apply" in sys.argv:
        apply()
    else:
        full()
