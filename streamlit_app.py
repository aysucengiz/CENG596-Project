from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import faiss
import pandas as pd
import streamlit as st

import clinical_trials_local_pipeline as backend


APP_OUTPUT_DIR = RUNTIME_ROOT
EMBEDDING_CACHE_DIR = RUNTIME_ROOT / "embeddings"
PREPARED_CACHE_DIR = RUNTIME_ROOT / "prepared_cache"


def render_active_stage(active_stage_box: Any, stage_name: str, detail: str | None = None) -> None:
    detail_html = f"<div class='stage-detail'>{detail}</div>" if detail else ""
    active_stage_box.markdown(
        f"""
        <style>
        .live-stage {{
            padding: 0.85rem 1rem;
            border-radius: 0.75rem;
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.28);
            color: rgb(8, 51, 97);
            margin-bottom: 0.5rem;
        }}
        .live-stage-title {{
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.15rem;
        }}
        .live-stage-title .dot {{
            animation: pulseDot 1.2s infinite;
            display: inline-block;
            width: 0.35rem;
            text-align: center;
        }}
        .live-stage-title .dot:nth-child(2) {{
            animation-delay: 0.2s;
        }}
        .live-stage-title .dot:nth-child(3) {{
            animation-delay: 0.4s;
        }}
        .stage-detail {{
            margin-top: 0.35rem;
            font-size: 0.92rem;
            color: rgba(8, 51, 97, 0.85);
        }}
        @keyframes pulseDot {{
            0%, 20% {{ opacity: 0.2; }}
            50% {{ opacity: 1; }}
            100% {{ opacity: 0.2; }}
        }}
        </style>
        <div class="live-stage">
            <div class="live-stage-title">
                <span>{stage_name}</span>
                <span class="dot">.</span>
                <span class="dot">.</span>
                <span class="dot">.</span>
            </div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_active_stage(active_stage_box: Any) -> None:
    active_stage_box.empty()


def render_status(status_box: Any, title: str, detail: str | None = None) -> None:
    detail_text = detail or ""
    status_box.markdown(
        f"**{title}**\n\n{detail_text}"
    )


def make_progress_callback(progress_bar: Any, status_box: Any, details_box: Any, active_stage_box: Any) -> Any:
    def _callback(percent: int, message: str, detail: str | None = None) -> None:
        progress_bar.progress(percent)
        render_active_stage(active_stage_box, message, detail)
        render_status(status_box, message, detail)
        if detail:
            details_box.caption(detail)
        else:
            details_box.empty()

    return _callback


def render_stage_log(stage_log: Any, stage_messages: list[str]) -> None:
    if stage_messages:
        stage_log.markdown("\n".join(f"- {message}" for message in stage_messages))


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {secs:.1f}s"


def build_search_stage_list(retrieval_mode: str, tune_rrf: bool, selected_qid: str | None) -> list[str]:
    stages = [
        "Load BM25 resources",
        "Run BM25 retrieval",
    ]
    if retrieval_mode in {"Dense", "Fused", "Reranked"}:
        stages.extend(
            [
                "Load dense resources",
                "Run dense retrieval",
            ]
        )
    if retrieval_mode in {"Fused", "Reranked"}:
        stages.append("Tune weighted RRF" if tune_rrf and selected_qid is not None else "Fuse results")
    if retrieval_mode == "Reranked":
        stages.extend(
            [
                "Load reranker",
                "Run reranking",
            ]
        )
    stages.append("Finalize results")
    return stages


def build_evaluation_stage_list(retrieval_mode: str, tune_rrf: bool) -> list[str]:
    stages = ["Load BM25 resources"]
    if retrieval_mode in {"Dense", "Fused", "Reranked"}:
        stages.append("Load dense resources")
    if retrieval_mode == "Reranked":
        stages.append("Load reranker")
    if retrieval_mode in {"Fused", "Reranked"} and tune_rrf:
        stages.append("Tune weighted RRF")
    stages.append("Run evaluation")
    stages.append("Finalize metrics")
    return stages


def enrich_results(results_df: pd.DataFrame, raw_docs: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in results_df.itertuples(index=False):
        raw = raw_docs.get(row.docno, {})
        rows.append(
            {
                "rank": int(row.rank),
                "docno": row.docno,
                "score": float(row.score),
                "title": raw.get("title", ""),
                "condition": raw.get("condition", ""),
                "summary_snippet": (raw.get("summary", "") or "")[:300],
                "eligibility_snippet": (raw.get("eligibility", "") or "")[:300],
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_bm25_resources(max_docs: int | None, rebuild_index: bool) -> dict[str, Any]:
    backend.ensure_pyterrier()
    docs_df, raw_docs = backend.get_docs_df(
        max_docs,
        cache_dir=PREPARED_CACHE_DIR,
        rebuild_prepared=False,
    )
    queries_df, qrels = backend.load_queries_and_qrels()
    index_dir = APP_OUTPUT_DIR / "bm25_index"
    index_ref = backend.build_bm25_index(docs_df, index_dir, rebuild=rebuild_index)
    return {
        "docs_df": docs_df,
        "raw_docs": raw_docs,
        "index_ref": index_ref,
        "queries_df": queries_df,
        "qrels": qrels,
    }


@st.cache_resource(show_spinner=False)
def load_query_resources() -> dict[str, Any]:
    queries_df, qrels = backend.load_queries_and_qrels()
    return {
        "queries_df": queries_df,
        "qrels": qrels,
    }


@st.cache_resource(show_spinner=False)
def load_dense_resources(
    max_docs: int | None,
    dense_text_mode: str,
    dense_device: str,
    dense_batch_size: int,
    raw_docs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    dense_doc_df = backend.get_dense_docs_df(
        max_docs,
        mode=dense_text_mode,
        cache_dir=PREPARED_CACHE_DIR,
        rebuild_prepared=False,
        raw_docs=raw_docs,
    )
    dense_model = backend.load_dense_model("NeuML/pubmedbert-base-embeddings", dense_device)
    doc_embeddings, dense_docnos = backend.get_dense_embeddings(
        dense_model,
        dense_doc_df,
        EMBEDDING_CACHE_DIR / "dense_doc_embeddings.npy",
        EMBEDDING_CACHE_DIR / "dense_docnos.npy",
        batch_size=dense_batch_size,
        reuse_cache=True,
    )
    faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings)
    return {
        "dense_doc_df": dense_doc_df,
        "dense_model": dense_model,
        "dense_docnos": dense_docnos,
        "faiss_index": faiss_index,
    }


@st.cache_resource(show_spinner=False)
def load_reranker(rerank_device: str) -> Any:
    return backend.load_reranker("NeuML/biomedbert-base-reranker", rerank_device)


def search_bm25(index_ref: Any, query_text: str, top_k: int) -> pd.DataFrame:
    query_df = pd.DataFrame([{"qid": "user_query", "query": query_text}])
    return backend.run_bm25(index_ref, query_df, top_k)


def search_dense(
    dense_model: Any,
    faiss_index: faiss.IndexFlatIP,
    dense_docnos: list[str],
    query_text: str,
    top_k: int,
) -> pd.DataFrame:
    normalized = backend.normalize_query_text(query_text)
    query_embedding = dense_model.encode(
        [normalized],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, indices = faiss_index.search(query_embedding, top_k)
    rows = []
    for rank, (doc_idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        rows.append(
            {
                "qid": "user_query",
                "docno": dense_docnos[doc_idx],
                "score": float(score),
                "rank": rank,
            }
        )
    return pd.DataFrame(rows)


def search_dense_all_queries(
    dense_model: Any,
    faiss_index: faiss.IndexFlatIP,
    dense_docnos: list[str],
    queries_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    query_df = queries_df[["qid", "query"]].copy()
    query_df["query"] = query_df["query"].apply(backend.normalize_query_text)
    query_embeddings = dense_model.encode(
        query_df["query"].tolist(),
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    scores, indices = faiss_index.search(query_embeddings, top_k)
    rows = []
    for q_idx, qid in enumerate(query_df["qid"].tolist()):
        for rank, (doc_idx, score) in enumerate(zip(indices[q_idx], scores[q_idx]), start=1):
            rows.append(
                {
                    "qid": qid,
                    "docno": dense_docnos[doc_idx],
                    "score": float(score),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def search_fused(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    top_k: int,
    rrf_k: int,
    alpha_bm25: float,
) -> pd.DataFrame:
    alpha_dense = 1.0 - alpha_bm25
    return backend.fuse_weighted_rrf(
        bm25_df,
        dense_df,
        rrf_k=rrf_k,
        top_k=top_k,
        alpha_bm25=alpha_bm25,
        alpha_dense=alpha_dense,
    )


def search_reranked(
    candidate_df: pd.DataFrame,
    reranker: Any,
    docs_df: pd.DataFrame,
    query_text: str,
    top_k: int,
) -> pd.DataFrame:
    query_df = pd.DataFrame([{"qid": "user_query", "query": query_text}])
    reranked = backend.rerank_candidates(
        reranker,
        candidate_df,
        query_df,
        docs_df,
        batch_size=8,
    )
    return reranked[reranked["rank"] <= top_k].copy()


def run_full_evaluation(
    retrieval_mode: str,
    top_k: int,
    bm25_resources: dict[str, Any],
    dense_resources: dict[str, Any] | None,
    rrf_k: int,
    alpha_bm25: float,
    tune_rrf: bool,
    reranker: Any | None,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    queries_df = bm25_resources["queries_df"]
    qrels = bm25_resources["qrels"]
    thresholded_qrels = backend.build_thresholded_qrels(qrels)
    metrics_by_name: dict[str, dict[str, dict[str, float]]] = {}

    def evaluate_named_run(name: str, results_df: pd.DataFrame) -> None:
        metrics_by_name[name] = {
            "rel>=1": backend.evaluate_run(results_df, thresholded_qrels["rel>=1"], top_k),
            "rel>=2": backend.evaluate_run(results_df, thresholded_qrels["rel>=2"], top_k),
        }

    bm25_results = backend.run_bm25(bm25_resources["index_ref"], queries_df, top_k)
    evaluate_named_run("BM25", bm25_results)
    final_results = bm25_results

    if retrieval_mode in {"Dense", "Fused", "Reranked"} and dense_resources is not None:
        dense_results = search_dense_all_queries(
            dense_resources["dense_model"],
            dense_resources["faiss_index"],
            dense_resources["dense_docnos"],
            queries_df,
            top_k,
        )
        evaluate_named_run("Dense", dense_results)
        final_results = dense_results
    else:
        dense_results = None

    if retrieval_mode in {"Fused", "Reranked"} and dense_results is not None:
        if tune_rrf:
            final_results, _, _ = backend.tune_weighted_rrf(
                bm25_results,
                dense_results,
                thresholded_qrels["rel>=1"],
                top_k,
            )
        else:
            final_results = search_fused(
                bm25_results,
                dense_results,
                top_k=top_k,
                rrf_k=rrf_k,
                alpha_bm25=alpha_bm25,
            )
        evaluate_named_run("Fused", final_results)

    if retrieval_mode == "Reranked" and reranker is not None:
        final_results = backend.rerank_candidates(
            reranker,
            final_results,
            queries_df,
            bm25_resources["docs_df"],
            batch_size=8,
        )
        final_results = final_results[final_results["rank"] <= top_k].copy()
        evaluate_named_run("Reranked", final_results)

    return final_results, metrics_by_name


def metrics_to_table(metrics_by_name: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for system_name, threshold_metrics in metrics_by_name.items():
        for threshold, metrics in threshold_metrics.items():
            p_key = next(key for key in metrics if key.startswith("P@"))
            r_key = next(key for key in metrics if key.startswith("R@"))
            ndcg_key = next(key for key in metrics if key.startswith("nDCG@"))
            rows.append(
                {
                    "system": system_name,
                    "threshold": threshold,
                    p_key: metrics[p_key],
                    r_key: metrics[r_key],
                    "MAP": metrics["MAP"],
                    "MRR": metrics["MRR"],
                    ndcg_key: metrics[ndcg_key],
                }
            )
    return pd.DataFrame(rows)


def update_progress(progress_bar: Any, status_box: Any, value: int, message: str) -> None:
    progress_bar.progress(value)
    render_status(status_box, message)


st.set_page_config(page_title="Clinical Trial Search", layout="wide")
st.title("Clinical Trial Search")
st.caption("BM25, dense retrieval, fusion, and reranking over the TREC Clinical Trials collection.")

with st.sidebar:
    st.header("Settings")
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        ["BM25", "Dense", "Fused", "Reranked"],
        index=2,
    )
    top_k = st.select_slider("Top-k", options=[10, 20, 50, 100, 250, 500, 1000], value=10)
    max_docs_choice = st.selectbox(
        "Corpus scope",
        ["Full corpus", "10,000 docs", "50,000 docs"],
        index=0,
    )
    max_docs = None if max_docs_choice == "Full corpus" else 10000 if max_docs_choice == "10,000 docs" else 50000
    rebuild_index = st.checkbox("Rebuild BM25 index", value=False)
    dense_text_mode = st.selectbox("Dense text mode", ["short", "full"], index=0)
    dense_device = st.text_input("Dense device", value="cpu")
    rerank_device = st.text_input("Reranker device", value="cpu")
    rrf_k = st.number_input("RRF k", min_value=1, max_value=500, value=60, step=1)
    alpha_bm25 = st.slider("BM25 fusion weight", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    tune_rrf = st.checkbox("Tune RRF (existing TREC query only)", value=False)

with st.spinner("Loading query list..."):
    query_loader_resources = load_query_resources()

query_source = st.radio("Query source", ["Custom query", "Existing TREC query"], horizontal=True)

selected_qid = None
selected_query_text = ""

if query_source == "Custom query":
    query_text = st.text_area(
        "Patient case description",
        height=180,
        placeholder="Enter a patient-style free-text query...",
    )
else:
    queries_df = query_loader_resources["queries_df"].copy()
    query_options = queries_df["qid"].tolist()
    selected_qid = st.selectbox("Choose a TREC query id", query_options)
    selected_query_text = queries_df.loc[queries_df["qid"] == selected_qid, "query"].iloc[0]
    st.text_area(
        "Selected TREC query text",
        value=selected_query_text,
        height=180,
        disabled=True,
    )
    query_text = selected_query_text

run_search = st.button("Run Search", type="primary", use_container_width=True)
run_evaluation = st.button("Evaluate All TREC Queries", use_container_width=True)

if run_search:
    if not query_text.strip():
        st.warning("Enter a query first.")
        st.stop()

    progress_bar = st.progress(0)
    status_box = st.empty()
    details_box = st.empty()
    active_stage_box = st.empty()
    stage_log = st.empty()
    stage_messages: list[str] = []
    stage_plan = build_search_stage_list(retrieval_mode, tune_rrf, selected_qid)
    completed_stages = [0]
    stage_started_at = [time.perf_counter()]
    workflow_started_at = time.perf_counter()

    def advance_stage(message: str, detail: str | None = None) -> None:
        elapsed = time.perf_counter() - stage_started_at[0]
        completed_stages[0] += 1
        percent = int((completed_stages[0] / len(stage_plan)) * 100)
        update_progress(progress_bar, status_box, percent, f"{completed_stages[0]}/{len(stage_plan)} - {message}")
        summary = f"{message} ({format_elapsed(elapsed)})"
        if detail:
            details_box.caption(detail)
            stage_messages.append(f"{summary}: {detail}")
        else:
            stage_messages.append(summary)
        render_stage_log(stage_log, stage_messages)
        stage_started_at[0] = time.perf_counter()

    progress_callback = make_progress_callback(progress_bar, status_box, details_box, active_stage_box)
    backend.set_progress_callback(progress_callback)
    try:
        update_progress(progress_bar, status_box, 1, f"0/{len(stage_plan)} - Starting search workflow...")
        details_box.caption(f"Mode: {retrieval_mode}")
        render_active_stage(active_stage_box, "Loading BM25 resources")

        bm25_resources = load_bm25_resources(max_docs=max_docs, rebuild_index=rebuild_index)
        advance_stage("Load BM25 resources", f"{len(bm25_resources['docs_df'])} documents ready")

        render_active_stage(active_stage_box, "Running BM25 retrieval", f"top_k={top_k}")
        bm25_results = search_bm25(bm25_resources["index_ref"], query_text, top_k)
        final_results = bm25_results
        dense_results = None
        fused_results = None
        advance_stage("Run BM25 retrieval", f"{len(bm25_results)} rows returned")

        if retrieval_mode in {"Dense", "Fused", "Reranked"}:
            render_active_stage(active_stage_box, "Loading dense resources", f"text mode={dense_text_mode}")
            dense_resources = load_dense_resources(
                max_docs=max_docs,
                dense_text_mode=dense_text_mode,
                dense_device=dense_device,
                dense_batch_size=32,
                raw_docs=bm25_resources["raw_docs"],
            )
            advance_stage("Load dense resources", f"{len(dense_resources['dense_docnos'])} vectors ready")

            render_active_stage(active_stage_box, "Dense retrieval", f"top_k={top_k}")
            dense_results = search_dense(
                dense_resources["dense_model"],
                dense_resources["faiss_index"],
                dense_resources["dense_docnos"],
                query_text,
                top_k,
            )
            final_results = dense_results
            advance_stage("Run dense retrieval", f"{len(dense_results)} rows returned")

        if retrieval_mode in {"Fused", "Reranked"}:
            if tune_rrf and selected_qid is not None:
                render_active_stage(active_stage_box, "Tuning weighted RRF", f"query={selected_qid}")
                bm25_for_tuning = bm25_results.copy()
                dense_for_tuning = dense_results.copy()
                bm25_for_tuning["qid"] = selected_qid
                dense_for_tuning["qid"] = selected_qid
                qrels_for_tuning = {selected_qid: bm25_resources["qrels"].get(selected_qid, {})}
                fused_results, best_trial, tuning_trials_df = backend.tune_weighted_rrf(
                    bm25_for_tuning,
                    dense_for_tuning,
                    backend.qrels_at_threshold(qrels_for_tuning, 1),
                    top_k,
                )
                advance_stage(
                    "Tune weighted RRF",
                    f"rrf_k={int(best_trial['rrf_k'])}, alpha_bm25={best_trial['alpha_bm25']:.1f}",
                )
                st.caption(
                    "Tuned weighted RRF: "
                    f"rrf_k={int(best_trial['rrf_k'])}, "
                    f"alpha_bm25={best_trial['alpha_bm25']:.1f}, "
                    f"alpha_dense={best_trial['alpha_dense']:.1f}"
                )
                with st.expander("Show RRF tuning trials"):
                    st.dataframe(tuning_trials_df, use_container_width=True, hide_index=True)
            else:
                if tune_rrf and selected_qid is None:
                    st.info("RRF tuning requires an existing TREC query because qrels are needed.")
                render_active_stage(active_stage_box, "Fusing results", f"rrf_k={rrf_k}, alpha_bm25={alpha_bm25:.1f}")
                fused_results = search_fused(
                    bm25_results,
                    dense_results,
                    top_k=top_k,
                    rrf_k=rrf_k,
                    alpha_bm25=alpha_bm25,
                )
                advance_stage("Fuse results", f"{len(fused_results)} rows returned")
            final_results = fused_results

        if retrieval_mode == "Reranked":
            render_active_stage(active_stage_box, "Loading reranker", rerank_device)
            reranker = load_reranker(rerank_device)
            advance_stage("Load reranker", "model ready")
            render_active_stage(active_stage_box, "Reranking", f"top_k={top_k}")
            final_results = search_reranked(
                fused_results,
                reranker,
                bm25_resources["docs_df"],
                query_text,
                top_k=top_k,
            )
            advance_stage("Run reranking", f"{len(final_results)} rows returned")

        render_active_stage(active_stage_box, "Finalizing results", retrieval_mode)
        advance_stage("Finalize results", f"displaying {len(final_results)} results for {retrieval_mode}")
        clear_active_stage(active_stage_box)
    finally:
        backend.set_progress_callback(None)
        clear_active_stage(active_stage_box)

    details_box.caption(f"Total search workflow time: {format_elapsed(time.perf_counter() - workflow_started_at)}")

    display_df = enrich_results(final_results, bm25_resources["raw_docs"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Ranked Results")
    with col2:
        st.metric("Returned documents", len(display_df))

    if selected_qid is not None:
        st.caption(f"Showing results for TREC query `{selected_qid}`")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    with st.expander("Show raw result table"):
        st.dataframe(final_results, use_container_width=True, hide_index=True)

if run_evaluation:
    progress_bar = st.progress(0)
    status_box = st.empty()
    details_box = st.empty()
    active_stage_box = st.empty()
    progress_callback = make_progress_callback(progress_bar, status_box, details_box, active_stage_box)
    stage_log = st.empty()
    stage_messages: list[str] = []
    stage_plan = build_evaluation_stage_list(retrieval_mode, tune_rrf)
    completed_stages = [0]
    stage_started_at = [time.perf_counter()]
    workflow_started_at = time.perf_counter()

    def advance_stage(message: str, detail: str | None = None) -> None:
        elapsed = time.perf_counter() - stage_started_at[0]
        completed_stages[0] += 1
        percent = int((completed_stages[0] / len(stage_plan)) * 100)
        update_progress(progress_bar, status_box, percent, f"{completed_stages[0]}/{len(stage_plan)} - {message}")
        summary = f"{message} ({format_elapsed(elapsed)})"
        if detail:
            details_box.caption(detail)
            stage_messages.append(f"{summary}: {detail}")
        else:
            stage_messages.append(summary)
        render_stage_log(stage_log, stage_messages)
        stage_started_at[0] = time.perf_counter()

    backend.set_progress_callback(progress_callback)
    try:
        update_progress(progress_bar, status_box, 1, f"0/{len(stage_plan)} - Starting evaluation workflow...")
        render_active_stage(active_stage_box, "Loading BM25 resources")
        bm25_resources = load_bm25_resources(max_docs=max_docs, rebuild_index=rebuild_index)
        advance_stage("Load BM25 resources", f"{len(bm25_resources['docs_df'])} documents ready")

        dense_resources = None
        if retrieval_mode in {"Dense", "Fused", "Reranked"}:
            render_active_stage(active_stage_box, "Loading dense resources", f"text mode={dense_text_mode}")
            dense_resources = load_dense_resources(
                max_docs=max_docs,
                dense_text_mode=dense_text_mode,
                dense_device=dense_device,
                dense_batch_size=32,
                raw_docs=bm25_resources["raw_docs"],
            )
            advance_stage("Load dense resources", f"{len(dense_resources['dense_docnos'])} vectors ready")

        reranker = None
        if retrieval_mode == "Reranked":
            render_active_stage(active_stage_box, "Loading reranker", rerank_device)
            reranker = load_reranker(rerank_device)
            advance_stage("Load reranker", "model ready")

        if retrieval_mode in {"Fused", "Reranked"} and tune_rrf:
            render_active_stage(active_stage_box, "Preparing weighted RRF tuning", "using rel>=1 qrels during evaluation")
            advance_stage("Tune weighted RRF", "tuning runs inside evaluation")

        update_progress(progress_bar, status_box, min(95, int(((completed_stages[0] + 0.5) / len(stage_plan)) * 100)), f"{completed_stages[0]}/{len(stage_plan)} - Running evaluation")
        details_box.caption(f"Evaluating mode: {retrieval_mode}")
        render_active_stage(active_stage_box, "Running evaluation", f"mode={retrieval_mode}, top_k={top_k}")
        _, eval_metrics = run_full_evaluation(
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            bm25_resources=bm25_resources,
            dense_resources=dense_resources,
            rrf_k=rrf_k,
            alpha_bm25=alpha_bm25,
            tune_rrf=tune_rrf,
            reranker=reranker,
        )
        advance_stage("Run evaluation", f"computed rel>=1 and rel>=2 metrics for {retrieval_mode}")
        render_active_stage(active_stage_box, "Finalizing metrics", retrieval_mode)
        advance_stage("Finalize metrics", f"finished evaluation for mode: {retrieval_mode}")
        clear_active_stage(active_stage_box)
    finally:
        backend.set_progress_callback(None)
        clear_active_stage(active_stage_box)

    details_box.caption(f"Total evaluation workflow time: {format_elapsed(time.perf_counter() - workflow_started_at)}")

    st.subheader("Evaluation Comparison")
    eval_table = metrics_to_table(eval_metrics)
    available_systems = ", ".join(eval_table["system"].drop_duplicates().tolist())
    st.caption(f"Included systems: {available_systems}")
    st.caption(
        f"Parameters used: top_k={top_k}, rrf_k={rrf_k}, alpha_bm25={alpha_bm25:.1f}, "
        f"alpha_dense={1.0 - alpha_bm25:.1f}, tune_rrf={tune_rrf}"
    )
    st.dataframe(
        eval_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAP": st.column_config.NumberColumn(format="%.4f"),
            "MRR": st.column_config.NumberColumn(format="%.4f"),
        },
    )
