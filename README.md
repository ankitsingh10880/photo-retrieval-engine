# Photo Retrieval Discovery Engine

An AI discovery engine over public Google Photos user feedback. It identifies **where** retrieval of
vaguely remembered photos breaks, instead of summarising sentiment.

**Live app:** one URL runs everything — How it works · Run the engine (live) · Findings · Ask the evidence · Explorer & method.

## Pipeline
1. **Collect** – the current dataset is 4,667 global English Google Play reviews. The pipeline also accepts uploaded
   CSVs and pasted text (Reddit, forums), but no Reddit data is included because Reddit blocks automated collection.
2. **Filter** – keyword pass, then the LLM keeps only items about finding an existing photo.
3. **Label** – Claude Haiku codes each item: photo type, cues remembered, cues forgotten, channel used,
   failure stage, verbatim query, workaround, severity.
4. **Re-check** – every item Haiku marked as retrieval (923) is re-labelled by Claude Sonnet with a stricter prompt and
   10 few-shot examples written outside the gold set (`recheck_relevance.py`) → 458 retrieval items.
   Then `subcode_stage0.py` codes why 'didn't search' items didn't search.
5. **Second pass** – items where the user didn't search are re-coded for *why*.
6. **Validate** – a stratified human-labelled sample checks LLM agreement per field.
7. **Score** – cue × failure-stage matrix; opportunity = share × severity × coverage gap.
8. **Ask** – questions answered only from retrieved items, with item ids cited (Claude Sonnet).

## Files
- `app.py` – Streamlit app
- `pipeline.py` – collect / filter / label / score / ask functions
- `data/baseline_labelled.jsonl` – the labelled dataset the app loads (no API calls needed to view it)

## Cost controls
Viewing findings makes no API calls. Live runs label at most 50 items per run, 3 runs per visitor,
30 runs per day across all visitors; Ask: 10 per visitor, 60 per day. The API account is prepaid.

## Run locally
```
pip install -r requirements.txt
mkdir -p .streamlit && echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml
streamlit run app.py
```
