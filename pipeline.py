"""
Project Nebula — Discovery Engine pipeline.

collect -> filter -> label (LLM) -> aggregate/score -> ask (retrieval + LLM)

Every stage is a plain function so the Streamlit app, the baseline builder
and tests all call the same code.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

LABEL_MODEL = "claude-haiku-4-5-20251001"
SYNTH_MODEL = "claude-sonnet-5-5"
BATCH_SIZE = 15

# ---------------------------------------------------------------- schema
PHOTO_TYPES = ["document_or_slide", "receipt_or_bill", "medical", "screenshot",
               "whatsapp_forward", "travel_or_place", "people_or_family",
               "event_or_occasion", "pet", "other", "unclear"]
ANCHORS = ["content_anchored", "event_anchored", "mixed", "unclear"]
CUES_RETAINED = ["event", "place", "people", "rough_time", "object_or_content",
                 "text_in_image", "sender_or_source", "visual_attribute",
                 "emotion_or_situation", "none_stated"]
CUES_FORGOTTEN = ["exact_date", "exact_place", "album", "keyword", "file_name",
                  "who_took_it", "none_stated"]
CHANNELS = ["search", "scroll", "folder_or_album", "map_or_explore", "ask_someone",
            "external_app", "gave_up_without_trying", "unclear"]
STAGES = ["0_channel_avoidance", "1_expression", "2_understanding",
          "3_evaluation", "4_refinement", "not_retrieval", "unclear"]
STAGE_LABELS = {
    "0_channel_avoidance": "0 · Didn't search",
    "1_expression": "1 · Couldn't express",
    "2_understanding": "2 · Not understood",
    "3_evaluation": "3 · Too many results",
    "4_refinement": "4 · Couldn't refine",
}
# How well Google Photos already serves each retained cue (1 = not at all).
COVERAGE_GAP = {
    "sender_or_source": 1.0, "emotion_or_situation": 1.0, "event": 0.5,
    "object_or_content": 0.5, "text_in_image": 0.5, "visual_attribute": 0.5,
    "rough_time": 0.5, "people": 0.2, "place": 0.2,
}

KEYWORDS = re.compile(
    r"\b(?:search|find|found|finding|locate|look(?:ing)? for|can'?t find|cannot find|"
    r"scroll|scrolling|remember|forgot|old photo|years? ago|screenshot|album|"
    r"face|people|location|map|date|timeline|memories|ask photos)\b", re.I)

LABEL_PROMPT = f"""You label user feedback about Google Photos for a product study on
retrieving photos that users remember only vaguely.

For EACH numbered item return one JSON object with keys:
id, is_retrieval, photo_type, cue_anchor, cues_retained, cues_forgotten,
channel, failure_stage, query_text, workaround, severity, evidence.

Rules:
- is_retrieval is true ONLY when the person is trying to locate a photo that
  exists in their library. Missing/deleted/unsynced photos, storage, backup,
  editing, UI and pricing complaints are false: set failure_stage
  "not_retrieval", lists to [], strings to "", severity 0.
- Code only what the text states or directly implies. Never invent cues.
- failure_stage = the EARLIEST stage where retrieval broke:
  0_channel_avoidance: didn't search (scrolled, folders, gave up) or says search is useless
  1_expression: knew things about the photo but couldn't turn them into a query
  2_understanding: gave a reasonable clue; search ignored or misread it
  3_evaluation: results came back but too many/too similar to spot the target
  4_refinement: first try failed and they couldn't narrow or adjust
- query_text: copied verbatim from the text, else "".
- workaround: short phrase, else "".
- severity: 1 mild, 2 real cost of time/effort, 3 urgent or high-stakes.
- evidence: ONE sentence copied from the text.

Allowed values:
photo_type: {PHOTO_TYPES}
cue_anchor: {ANCHORS}
cues_retained (list): {CUES_RETAINED}
cues_forgotten (list): {CUES_FORGOTTEN}
channel: {CHANNELS}
failure_stage: {STAGES}

Return ONLY a JSON array, one object per item, same ids. No prose."""


# ---------------------------------------------------------------- collect
def collect_play(app_id="com.google.android.apps.photos", country="in",
                 n=200, lang="en") -> pd.DataFrame:
    from google_play_scraper import Sort, reviews
    rows, token = [], None
    while len(rows) < n:
        batch, token = reviews(app_id, lang=lang, country=country,
                               sort=Sort.NEWEST, count=min(200, n - len(rows)),
                               continuation_token=token)
        if not batch:
            break
        rows.extend(batch)
        if token is None:
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id", "source", "text", "date", "url"])
    return pd.DataFrame({
        "id": "play_" + df["reviewId"].astype(str),
        "source": "play_store",
        "text": df["content"].fillna(""),
        "date": pd.to_datetime(df["at"]).dt.date.astype(str),
        "url": "",
    })


def collect_csv(df: pd.DataFrame, text_col: str, source="uploaded_csv") -> pd.DataFrame:
    return pd.DataFrame({
        "id": [f"{source}_{i}" for i in range(len(df))],
        "source": source,
        "text": df[text_col].fillna("").astype(str),
        "date": "", "url": "",
    })


def collect_pasted(blob: str, source="pasted") -> pd.DataFrame:
    """Split pasted text into items on blank lines or '---' separators."""
    parts = [p.strip() for p in re.split(r"\n\s*(?:---+)?\s*\n", blob) if len(p.strip()) > 30]
    return pd.DataFrame({
        "id": [f"{source}_{i}" for i in range(len(parts))],
        "source": source, "text": parts, "date": "", "url": "",
    })


# ---------------------------------------------------------------- filter
def keyword_filter(df: pd.DataFrame, min_len=40) -> pd.DataFrame:
    m = df["text"].str.contains(KEYWORDS) & (df["text"].str.len() >= min_len)
    return df[m].reset_index(drop=True)


# ---------------------------------------------------------------- label
def response_text(msg) -> str:
    """Join the text blocks of a response; some models put a non-text block (e.g. thinking) first."""
    return "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "text") == "text").strip()


def _parse_json_array(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("["), raw.rfind("]")
    return json.loads(raw[start:end + 1])


def _label_batch(client, batch: pd.DataFrame, retries=3) -> list[dict]:
    items = "\n\n".join(f"[{r.id}]\n{r.text[:1500]}" for r in batch.itertuples())
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=LABEL_MODEL, max_tokens=4000, system=LABEL_PROMPT,
                messages=[{"role": "user", "content": items}])
            return _parse_json_array(response_text(msg))
        except Exception:
            if attempt == retries - 1:
                return [{"id": i, "failure_stage": "unclear", "is_retrieval": None,
                         "label_error": True} for i in batch["id"]]
            time.sleep(2 ** attempt * 3)


def label(df: pd.DataFrame, client, workers=4, progress=None) -> pd.DataFrame:
    batches = [df.iloc[i:i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
    out, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda b: _label_batch(client, b), batches):
            out.extend(res)
            done += 1
            if progress:
                progress(done / len(batches))
    lab = pd.DataFrame(out)
    for col in ["is_retrieval", "photo_type", "cue_anchor", "cues_retained", "cues_forgotten",
                "channel", "failure_stage", "query_text", "workaround", "severity",
                "evidence", "label_error"]:
        if col not in lab:
            lab[col] = None
    lab["id"] = lab["id"].astype(str).str.strip("[]")
    merged = df.merge(lab, on="id", how="left")
    for col in ["cues_retained", "cues_forgotten"]:
        merged[col] = merged[col].apply(lambda v: v if isinstance(v, list) else [])
    for col in ["query_text", "workaround", "evidence", "photo_type",
                "cue_anchor", "channel", "failure_stage"]:
        merged[col] = merged[col].fillna("")
    # model output can drift from the schema: lists where strings belong, off-list values
    for col, allowed in [("photo_type", PHOTO_TYPES), ("cue_anchor", ANCHORS),
                         ("channel", CHANNELS), ("failure_stage", STAGES)]:
        merged[col] = merged[col].apply(lambda v: "" if isinstance(v, list) else v)
        merged[col] = merged[col].where(merged[col].isin(allowed + [""]), "unclear")
    for col in ["query_text", "workaround", "evidence"]:
        merged[col] = merged[col].apply(lambda v: "" if isinstance(v, list) else v)
    for col, allowed in [("cues_retained", CUES_RETAINED), ("cues_forgotten", CUES_FORGOTTEN)]:
        merged[col] = merged[col].apply(lambda v, a=allowed: [x for x in v if isinstance(x, str) and x in a])
    merged["severity"] = pd.to_numeric(merged["severity"], errors="coerce").fillna(0)
    # explicit conversion (no pandas downcasting); a string "true" from the model still counts as True
    merged["is_retrieval"] = merged["is_retrieval"].map(
        lambda v: v is True or (isinstance(v, str) and v.strip().lower() == "true"))
    if "label_error" not in merged:
        merged["label_error"] = False
    # ids the model skipped also count as errors so they get retried
    # failed batches are flagged True; items the model silently skipped have no failure_stage
    merged["label_error"] = merged["label_error"].map(lambda v: v is True) | merged["failure_stage"].eq("")
    return merged


# ---------------------------------------------------------------- aggregate
def retrieval_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_retrieval"] & df["failure_stage"].isin(STAGE_LABELS)].copy()


def cue_stage_matrix(df: pd.DataFrame) -> pd.DataFrame:
    r = retrieval_only(df).explode("cues_retained").reset_index(drop=True)
    r = r[r["cues_retained"].isin(COVERAGE_GAP)]
    if r.empty:
        return pd.DataFrame()
    m = pd.crosstab(r["cues_retained"], r["failure_stage"])
    return m.reindex(columns=[s for s in STAGE_LABELS if s in m.columns])


def opportunity_table(df: pd.DataFrame) -> pd.DataFrame:
    r = retrieval_only(df).explode("cues_retained").reset_index(drop=True)
    r = r[r["cues_retained"].isin(COVERAGE_GAP)]
    if r.empty:
        return pd.DataFrame()
    total = retrieval_only(df).shape[0]
    g = (r.groupby(["cues_retained", "failure_stage"])
          .agg(items=("id", "nunique"), mean_severity=("severity", "mean"))
          .reset_index())
    g["share"] = g["items"] / total
    g["coverage_gap"] = g["cues_retained"].map(COVERAGE_GAP)
    g["opportunity_score"] = (g["share"] * g["mean_severity"] * g["coverage_gap"] * 100).round(2)
    g["stage"] = g["failure_stage"].map(STAGE_LABELS)
    return g.sort_values("opportunity_score", ascending=False)


def example_quotes(df: pd.DataFrame, cue: str, stage: str, k=3) -> list[str]:
    r = retrieval_only(df)
    r = r[r["failure_stage"].eq(stage) & r["cues_retained"].apply(lambda c: cue in c)]
    r = r.sort_values("severity", ascending=False)
    return [f"{e} — ({i})" for e, i in zip(r["evidence"].head(k), r["id"].head(k)) if e]


# ---------------------------------------------------------------- ask
def ask(df: pd.DataFrame, question: str, client, k=25) -> str:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    r = retrieval_only(df).reset_index(drop=True)
    if r.empty:
        return "No labelled retrieval evidence loaded yet."
    docs = (r["text"] + " " + r["photo_type"] + " " + r["failure_stage"] + " "
            + r["cues_retained"].apply(" ".join) + " " + r["workaround"])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    sims = cosine_similarity(vec.transform([question]), X).ravel()
    top = r.iloc[sims.argsort()[::-1][:k]]

    counts = {
        "n_retrieval_items": len(r),
        "failure_stage_counts": r["failure_stage"].value_counts().to_dict(),
        "channel_counts": r["channel"].value_counts().to_dict(),
        "photo_type_counts": r["photo_type"].value_counts().to_dict(),
    }
    evidence = "\n".join(
        f"[{t.id}] ({t.source}; type={t.photo_type}; stage={t.failure_stage}; "
        f"retained={','.join(t.cues_retained)}) {t.text[:600]}"
        for t in top.itertuples())
    msg = client.messages.create(
        model=SYNTH_MODEL, max_tokens=2000,
        system=("You answer product-research questions using ONLY the evidence "
                "given. Lead with the answer, quantify with the corpus counts "
                "where relevant, cite item ids in square brackets after each "
                "claim, and say plainly when the evidence is thin. Under 250 words."),
        messages=[{"role": "user", "content":
                   f"Corpus counts: {json.dumps(counts)}\n\nRetrieved evidence:\n"
                   f"{evidence}\n\nQuestion: {question}"}])
    answer = response_text(msg)
    return answer or "The model returned no text for this question. Please try rephrasing it."
