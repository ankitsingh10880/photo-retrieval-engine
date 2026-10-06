"""Compare hand labels in data/gold_sample.csv with LLM labels -> data/validation.json"""
import json
from pathlib import Path
import pandas as pd

D = Path(__file__).parent / "data"
gold = pd.read_csv(D / "gold_sample.csv", dtype=str).fillna("")
llm = pd.read_json(D / "baseline_labelled.jsonl", lines=True, convert_dates=False, dtype={"date": str}).set_index("id")
fields = ["is_retrieval", "photo_type", "cue_anchor", "channel", "failure_stage"]
rows = []
for f in fields:
    g = gold[gold[f].ne("")]
    if g.empty:
        continue
    pred = llm.loc[g["id"], f].astype(str).str.lower().values
    agree = (g[f].str.strip().str.lower().values == pred).mean()
    rows.append({"field": f, "items_checked": len(g), "agreement_pct": round(agree * 100, 1)})
out = {"fields": rows, "note": "Agreement between the final labels (Claude Haiku, then a Claude Sonnet re-check of every item Haiku marked retrieval) and 40 items hand-labelled by one person, sampled across Haiku's stages (25 it marked retrieval, 15 other)."}
(D / "validation.json").write_text(json.dumps(out, indent=2))
print(pd.DataFrame(rows))
