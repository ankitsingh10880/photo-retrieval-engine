"""
Project Nebula — AI-Powered Discovery Engine (single-URL gateway).

Tabs: How it works · Run the engine (live) · Findings · Ask the evidence · Explorer
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as pl

st.set_page_config(page_title="Photo Retrieval Discovery Engine", page_icon="🔎", layout="wide")

DATA = Path(__file__).parent / "data" / "baseline_labelled.jsonl"
VALIDATION = Path(__file__).parent / "data" / "validation.json"
LIVE_CAP = 100
MAX_LIVE_RUNS = 2
MAX_QUESTIONS = 10
SEQ = ["#EEF3FB", "#C6D7F0", "#8FB0E0", "#4F7FC6", "#24519A", "#0F2E63"]  # one hue, light→dark
BAR = "#24519A"


@st.cache_data
def load_baseline() -> pd.DataFrame:
    if not DATA.exists():
        return pd.DataFrame()
    return pd.read_json(DATA, lines=True, convert_dates=False, dtype={"date": str})


def get_client():
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        key = None
    if not key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=key)


def corpus() -> pd.DataFrame:
    base = load_baseline()
    live = st.session_state.get("live")
    if live is not None and not live.empty:
        return pd.concat([base, live], ignore_index=True).drop_duplicates("id", keep="last")
    return base


client = get_client()
st.session_state.setdefault("live_runs", 0)
st.session_state.setdefault("questions", 0)
runs_left = MAX_LIVE_RUNS - st.session_state["live_runs"]
qs_left = MAX_QUESTIONS - st.session_state["questions"]
df = corpus()
ret = pl.retrieval_only(df) if not df.empty else pd.DataFrame()

st.title("Why can't people find photos they remember?")
st.caption("An AI discovery engine over public user feedback about Google Photos — "
           "built to compare *where* retrieval of vaguely remembered photos breaks, not just summarise sentiment.")

if not df.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items analysed", f"{len(df):,}")
    c2.metric("About finding a photo", f"{len(ret):,}")
    c3.metric("Sources", df["source"].str.replace(r"_(in|us)$", "", regex=True).nunique())
    top = ret["failure_stage"].value_counts().idxmax() if not ret.empty else "—"
    c4.metric("Top break point", pl.STAGE_LABELS.get(top, top).split(" · ")[-1])

tabs = st.tabs(["How it works", "Run the engine", "Findings", "Ask the evidence", "Explorer & method"])

# ------------------------------------------------------------- 1. how it works
with tabs[0]:
    st.subheader("Pipeline")
    st.markdown("""
| Step | What happens | Tooling |
|---|---|---|
| **1 · Collect** | Play Store reviews (India + US), 4,667 in this dataset. The pipeline can also take Reddit / Help Community threads, uploads or pasted text, but the current dataset is Play Store only. | `google-play-scraper`, manual thread capture |
| **2 · Filter** | Keyword pass, then the LLM keeps only items about *finding* an existing photo (drops backup, deletion, storage, pricing noise) | regex + Claude Haiku |
| **3 · Label** | Each item coded against a fixed schema: photo type, cues the user still remembers, cues forgotten, channel used, **failure stage**, verbatim query, workaround, severity | Claude Haiku, batched JSON |
| **4 · Re-check** | Every item the first labeller (Claude Haiku) marked as retrieval (923) is re-labelled by Claude Sonnet with a stricter prompt and 10 few-shot examples written outside the gold set; retrieval items fall from 923 to 458 | LLM re-check |
| **5 · Validate** | 40 items sampled across the first labeller's stages (25 it marked retrieval, 15 other) and hand-labelled by one person. Agreement after the re-check: is-retrieval 77.5% (was 67.5%), failure stage 75.0% (was 57.5%); over-inclusions 13 → 6. Improved, not validated (reference bar 85%); single labeller | human gold set |
| **6 · Score** | Cue × failure-stage matrix; opportunity = share × severity × coverage gap | pandas |
| **7 · Ask** | Questions answered only from retrieved items, with item ids cited | TF-IDF retrieval + Claude Sonnet |
""")
    st.subheader("The failure stages the engine codes against")
    st.markdown("""
Retrieval success = **channel** (did they even search?) × **expression** (could they turn memory into a query?)
× **understanding** (did the system read the clue?) × **evaluation** (could they spot it in the results?)
× **refinement** (could they recover from a miss?).
Each item is assigned the *earliest* stage where retrieval broke.
""")
    st.subheader("Cost controls on this public demo")
    st.markdown(f"""
- **Findings, Explorer and this page make no API calls**: they read a dataset labelled once, offline.
- **Live runs** are capped at **{LIVE_CAP} items per run** and **{MAX_LIVE_RUNS} runs per session**.
- **Ask the evidence** is capped at **{MAX_QUESTIONS} questions per session**.
- The offline batch labelling run validates the API key with a 1-token call, shows an estimated cost for confirmation, and checks a 15-item pilot batch against the schema before labelling the rest.
- The API account is prepaid, so total spend has a hard ceiling.
""")
    st.info("Start with **Findings** for results on the full dataset, or **Run the engine** to watch the "
            "pipeline process fresh data live.")

# ------------------------------------------------------------- 2. run live
with tabs[1]:
    st.subheader("Run the full pipeline on fresh data")
    runs_slot = st.empty()
    if client is None:
        st.warning("Live labelling is unavailable: no API key configured on this deployment.")
    src = st.radio("Source", ["Play Store (live pull)", "Upload CSV", "Paste text (Reddit, forums, comments)"],
                   horizontal=True)
    raw = None
    if src.startswith("Play"):
        a, b, c = st.columns(3)
        app_id = a.text_input("App id", "com.google.android.apps.photos")
        country = b.selectbox("Country", ["in", "us", "gb", "au", "ca"])
        n_pull = c.slider("Reviews to pull", 200, 2000, 600, 100)
        if st.button("Collect, filter and label", type="primary", disabled=client is None or runs_left <= 0):
            with st.status("Running pipeline…", expanded=True) as s:
                st.write("Collecting reviews…")
                raw = pl.collect_play(app_id, country, n_pull)
                st.write(f"Collected {len(raw)}.")
    elif src.startswith("Upload"):
        f = st.file_uploader("CSV file", type="csv")
        if f is not None:
            up = pd.read_csv(f)
            col = st.selectbox("Text column", up.columns)
            if st.button("Filter and label", type="primary", disabled=client is None or runs_left <= 0):
                raw = pl.collect_csv(up, col)
    else:
        blob = st.text_area("Paste posts. Separate items with a blank line or ---", height=220)
        if st.button("Filter and label", type="primary", disabled=client is None or runs_left <= 0) and blob.strip():
            raw = pl.collect_pasted(blob)

    if raw is not None:
        st.session_state["live_runs"] += 1
        filt = pl.keyword_filter(raw) if src.startswith(("Play", "Upload")) else raw
        filt = filt.head(LIVE_CAP)
        st.write(f"Keyword filter kept **{len(filt)}** of {len(raw)} items. Labelling…")
        bar = st.progress(0.0)
        out = pl.label(filt, client, progress=bar.progress)
        st.session_state["live"] = pd.concat([st.session_state.get("live", pd.DataFrame()), out],
                                             ignore_index=True)
        kept = pl.retrieval_only(out)
        st.success(f"Done. {len(kept)} of {len(out)} labelled items are about finding a photo. "
                   "Findings and Ask now include them.")
        if not kept.empty:
            st.dataframe(kept[["id", "photo_type", "cues_retained", "channel", "failure_stage", "evidence"]],
                         width="stretch", hide_index=True)

    used = st.session_state["live_runs"]
    runs_slot.caption(f"Live runs are capped at {LIVE_CAP} items. Results merge into Findings and Ask for this session. "
                      f"Runs left this session: {max(MAX_LIVE_RUNS - used, 0)} of {MAX_LIVE_RUNS}."
                      + (" Demo limit reached for this session to keep API costs bounded." if used >= MAX_LIVE_RUNS else ""))

# ------------------------------------------------------------- 3. findings
with tabs[2]:
    if ret.empty:
        st.info("No labelled data loaded yet.")
    else:
        anchor = st.radio("Memory type", ["All", "content_anchored", "event_anchored"], horizontal=True,
                          format_func=lambda x: {"All": "All", "content_anchored": "Content-anchored (what was in it)",
                                                 "event_anchored": "Event-anchored (trip / occasion)"}[x])
        period = st.radio("Period", ["All years", "2024 onwards (AI search era)"], horizontal=True,
                          help="Play reviews carry dates. The current dataset is Play Store only; undated items such as pasted Reddit / Help Community threads would appear in both views.")
        view = df if anchor == "All" else df[df["cue_anchor"].eq(anchor)]
        if period != "All years":
            yr = pd.to_datetime(view["date"], errors="coerce").dt.year
            view = view[yr.isna() | (yr >= 2024)]
        vret = pl.retrieval_only(view)
        st.caption(f"{len(vret):,} retrieval items in this view.")

        st.subheader("Where retrieval breaks")
        stage_ct = (vret["failure_stage"].map(pl.STAGE_LABELS).value_counts()
                    .reindex(list(pl.STAGE_LABELS.values())).fillna(0).reset_index())
        stage_ct.columns = ["Stage", "Items"]
        stage_ct["Share"] = (stage_ct["Items"] / max(len(vret), 1) * 100).round(1)
        fig = px.bar(stage_ct, x="Items", y="Stage", orientation="h", text="Share",
                     color_discrete_sequence=[BAR])
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          hovertemplate="%{y}: %{x} items (%{text}%)<extra></extra>")
        fig.update_layout(height=300, yaxis=dict(autorange="reversed", title=None),
                          xaxis=dict(showgrid=False), margin=dict(l=0, r=40, t=10, b=10))
        st.plotly_chart(fig)

        if "stage0_reason" in vret:
            stage = vret["failure_stage"]
            reason = vret["stage0_reason"].fillna("").astype(str)
            s0 = vret[stage.eq("0_channel_avoidance") & reason.ne("")]
            if not s0.empty:
                st.subheader("Why people didn't search")
                st.caption("Second-pass coding of items labelled 'didn't search' after the Sonnet re-check. "
                           "'Searched and failed' items were moved to stage 2." + f" n = {len(s0)} items.")
                r_ct = s0["stage0_reason"].str.replace("_", " ").value_counts().reset_index()
                r_ct.columns = ["Reason", "Items"]
                r_ct["Share"] = (r_ct["Items"] / r_ct["Items"].sum() * 100).round(1)
                rfig = px.bar(r_ct, x="Items", y="Reason", orientation="h", text="Share",
                              color_discrete_sequence=[BAR])
                rfig.update_traces(texttemplate="%{text}%", textposition="outside",
                                   hovertemplate="%{y}: %{x} items (%{text}%)<extra></extra>")
                rfig.update_layout(height=260, yaxis=dict(autorange="reversed", title=None),
                                   xaxis=dict(showgrid=False), margin=dict(l=0, r=40, t=10, b=10))
                st.plotly_chart(rfig)

        st.subheader("What users remember × where it breaks")
        mat = pl.cue_stage_matrix(view)
        if not mat.empty:
            mat = mat.rename(columns=pl.STAGE_LABELS)
            hm = px.imshow(mat, text_auto=True, aspect="auto", color_continuous_scale=SEQ,
                           labels=dict(x="Failure stage", y="Cue the user still remembers", color="Items"))
            hm.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(hm)

        st.subheader("Ranked opportunities")
        st.caption("Score = share of retrieval items × mean severity × coverage gap (how poorly the cue is served today).")
        opp = pl.opportunity_table(view).head(8)
        if not opp.empty:
            for row in opp.itertuples():
                with st.expander(f"{row.opportunity_score:.1f} · remembers **{row.cues_retained}** → "
                                 f"breaks at **{row.stage}** ({row.items} items, severity {row.mean_severity:.1f})"):
                    for q in pl.example_quotes(view, row.cues_retained, row.failure_stage):
                        st.markdown(f"> {q}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("How people try to find it")
            ch = vret["channel"].value_counts().reset_index()
            ch.columns = ["Channel", "Items"]
            st.dataframe(ch, hide_index=True, width="stretch")
        with c2:
            st.subheader("Workarounds")
            wk = vret.loc[vret["workaround"].ne(""), "workaround"].str.lower().value_counts().head(10).reset_index()
            wk.columns = ["Workaround", "Mentions"]
            st.dataframe(wk, hide_index=True, width="stretch")

        st.subheader("Queries users actually typed")
        qs = vret.loc[vret["query_text"].ne(""), ["query_text", "photo_type", "failure_stage"]]
        st.dataframe(qs, hide_index=True, width="stretch")

# ------------------------------------------------------------- 4. ask
with tabs[3]:
    st.subheader("Ask the evidence")
    st.caption("Answers use only the labelled items, and cite them by id. Check any id in Explorer.")
    examples = ["Why do people scroll instead of searching?",
                "What do people remember about a photo, and what have they forgotten?",
                "What clues do users give that search fails to understand?",
                "How do users phrase searches when their memory is incomplete?",
                "What workarounds do people use when they can't find a photo?"]
    pick = st.selectbox("Example questions", ["—"] + examples)
    q = st.text_input("Your question", value="" if pick == "—" else pick)
    qs_slot = st.empty()
    if st.button("Ask", type="primary", disabled=client is None or not q or qs_left <= 0):
        st.session_state["questions"] += 1
        with st.spinner("Retrieving and reading evidence…"):
            st.markdown(pl.ask(df, q, client))

    asked = st.session_state["questions"]
    qs_slot.caption(f"Questions left this session: {max(MAX_QUESTIONS - asked, 0)} of {MAX_QUESTIONS}."
                    + (" Demo limit reached for this session to keep API costs bounded." if asked >= MAX_QUESTIONS else ""))

# ------------------------------------------------------------- 5. explorer
with tabs[4]:
    st.subheader("Labelled data")
    if not df.empty:
        f1, f2, f3 = st.columns(3)
        s_src = f1.multiselect("Source", sorted(df["source"].unique()))
        s_stage = f2.multiselect("Failure stage", list(pl.STAGE_LABELS))
        only_ret = f3.checkbox("Only items about finding a photo", True)
        e = df.copy()
        if only_ret:
            e = pl.retrieval_only(e)
        if s_src:
            e = e[e["source"].isin(s_src)]
        if s_stage:
            e = e[e["failure_stage"].isin(s_stage)]
        st.dataframe(e, hide_index=True, width="stretch")
        st.download_button("Download CSV", e.to_csv(index=False), "labelled_items.csv")
    st.subheader("Validation")
    if VALIDATION.exists():
        v = json.loads(VALIDATION.read_text())
        st.dataframe(pd.DataFrame(v["fields"]), hide_index=True)
        st.caption(v.get("note", ""))
    else:
        st.caption("Gold-set agreement will appear here once the hand-labelled check is complete.")
    st.subheader("Limitations")
    st.markdown("""
- The current dataset is Play Store reviews only. The pipeline can also take Reddit / Help Community threads (pasted or uploaded), but Reddit blocks automated collection, so live Reddit pulls are not possible.
- Reviews skew to complaints; the engine measures *where* retrieval fails, not *how often* it fails across all users.
- English-language feedback only.
""")
