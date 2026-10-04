# Photo Retrieval Discovery Engine

An AI discovery engine over public Google Photos user feedback. It identifies **where** retrieval of
vaguely remembered photos breaks, instead of summarising sentiment.

**Live app:** one URL runs everything — How it works · Run the engine (live) · Findings · Ask the evidence · Explorer & method.

## Pipeline
1. **Collect** – Google Play reviews (global English) plus forum threads.
2. **Filter** – keyword pass, then the LLM keeps only items about finding an existing photo.
3. **Label** – Claude Haiku codes each item: photo type, cues remembered, cues forgotten, channel used,
   failure stage, verbatim query, workaround, severity.
4. **Second pass** – items where the user didn't search are re-coded for *why*.
5. **Validate** – a stratified human-labelled sample checks LLM agreement per field.
6. **Score** – cue × failure-stage matrix; opportunity = share × severity × coverage gap.
7. **Ask** – questions answered only from retrieved items, with item ids cited (Claude Sonnet).

## Files
- `app.py` – Streamlit app
- `pipeline.py` – collect / filter / label / score / ask functions
- `data/baseline_labelled.jsonl` – the labelled dataset the app loads (no API calls needed to view it)

## Cost controls
Viewing findings makes no API calls. Live runs are capped at 100 items and 2 runs per session;
questions at 10 per session. The API account is prepaid.

## Run locally
```
pip install -r requirements.txt
mkdir -p .streamlit && echo 'ANTHROPIC_API_KEY = "sk-ant-..."' > .streamlit/secrets.toml
streamlit run app.py
```
