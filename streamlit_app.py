from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import faiss
import pandas as pd
import streamlit as st

import clinical_trials_local_pipeline as backend


RERANKER_MODEL = "ncbi/MedCPT-Cross-Encoder"
APP_OUTPUT_DIR = backend.RUNTIME_ROOT
EMBEDDING_CACHE_DIR = backend.RUNTIME_ROOT / "embeddings"
PREPARED_CACHE_DIR = backend.RUNTIME_ROOT / "prepared_cache"
MODEL_CACHE_DIR = backend.DEFAULT_HF_CACHE_DIR
RERANK_CACHE_PATH = (
    APP_OUTPUT_DIR
    / f"rerank_scores_{backend.safe_cache_name(RERANKER_MODEL)}_{backend.RERANK_CACHE_VERSION}.parquet"
)


def render_active_stage(active_stage_box: Any, stage_name: str, detail: str | None = None) -> None:
    detail_html = f"<div class='stage-detail'>{detail}</div>" if detail else ""
    active_stage_box.markdown(
        f"""
        <style>
        :root {{
            --stage-bg-a: rgba(236, 253, 245, 0.98);
            --stage-bg-b: rgba(240, 249, 255, 0.98);
            --stage-orb: rgba(20, 184, 166, 0.18);
            --stage-border: rgba(20, 184, 166, 0.42);
            --stage-accent: #0f766e;
            --stage-title: #134e4a;
            --stage-text: #334155;
            --stage-shadow: rgba(15, 23, 42, 0.1);
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --stage-bg-a: rgba(15, 23, 42, 0.92);
                --stage-bg-b: rgba(19, 78, 74, 0.72);
                --stage-orb: rgba(45, 212, 191, 0.22);
                --stage-border: rgba(94, 234, 212, 0.38);
                --stage-accent: #2dd4bf;
                --stage-title: #ccfbf1;
                --stage-text: #cbd5e1;
                --stage-shadow: rgba(0, 0, 0, 0.35);
            }}
        }}
        .live-stage {{
            padding: 0.95rem 1rem;
            border-radius: 0.9rem;
            background:
                radial-gradient(circle at top left, var(--stage-orb), transparent 44%),
                linear-gradient(135deg, var(--stage-bg-a), var(--stage-bg-b));
            border: 1px solid var(--stage-border);
            border-left: 5px solid var(--stage-accent);
            color: var(--stage-text);
            margin-bottom: 0.5rem;
            box-shadow: 0 12px 30px var(--stage-shadow);
        }}
        .live-stage-title {{
            font-weight: 800;
            display: flex;
            align-items: center;
            gap: 0.15rem;
            color: var(--stage-title);
            letter-spacing: 0.01em;
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
            color: var(--stage-text);
            line-height: 1.4;
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
    detail_html = f"<div class='status-detail'>{detail}</div>" if detail else ""
    status_box.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">{title}</div>
            {detail_html}
        </div>
        <style>
        :root {{
            --status-bg-a: #fff7ed;
            --status-bg-b: #fefce8;
            --status-border: #fed7aa;
            --status-accent: #f97316;
            --status-title: #7c2d12;
            --status-text: #475569;
            --status-shadow: rgba(124, 45, 18, 0.09);
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --status-bg-a: rgba(30, 41, 59, 0.94);
                --status-bg-b: rgba(67, 56, 202, 0.34);
                --status-border: rgba(129, 140, 248, 0.42);
                --status-accent: #818cf8;
                --status-title: #e0e7ff;
                --status-text: #cbd5e1;
                --status-shadow: rgba(0, 0, 0, 0.32);
            }}
        }}
        .status-card {{
            padding: 0.75rem 0.9rem;
            border-radius: 0.8rem;
            background: linear-gradient(135deg, var(--status-bg-a), var(--status-bg-b));
            border: 1px solid var(--status-border);
            border-left: 4px solid var(--status-accent);
            color: var(--status-text);
            margin: 0.35rem 0;
            box-shadow: 0 8px 20px var(--status-shadow);
        }}
        .status-title {{
            font-weight: 800;
            color: var(--status-title);
        }}
        .status-detail {{
            margin-top: 0.25rem;
            color: var(--status-text);
            font-size: 0.92rem;
            line-height: 1.4;
        }}
        </style>
        """,
        unsafe_allow_html=True,
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
        "BM25 preparation",
        "BM25 ranking",
    ]
    if retrieval_mode in {"Dense", "Fused", "Reranked"}:
        stages.extend(
            [
                "Dense preparation",
                "Dense ranking",
            ]
        )
    if retrieval_mode in {"Fused", "Reranked"}:
        stages.append("Fusing")
    if retrieval_mode == "Reranked":
        stages.extend(
            [
                "Reranking preparation",
                "Reranking",
            ]
        )
    stages.append("Finalize results")
    return stages


def build_evaluation_stage_list(retrieval_mode: str, tune_rrf: bool) -> list[str]:
    stages = ["BM25 preparation", "BM25 ranking"]
    if retrieval_mode in {"Dense", "Fused", "Reranked"}:
        stages.extend(["Dense preparation", "Dense ranking"])
    if retrieval_mode in {"Fused", "Reranked"}:
        stages.append("Fusing")
    if retrieval_mode == "Reranked":
        stages.extend(["Reranking preparation", "Reranking"])
    stages.append("Evaluations")
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


def enrich_results_with_relevance(
    results_df: pd.DataFrame,
    raw_docs: dict[str, dict[str, str]],
    relevance_by_docno: dict[str, int],
    top_n: int = 10,
) -> pd.DataFrame:
    enriched = enrich_results(results_df.sort_values("rank").head(top_n), raw_docs)
    if enriched.empty:
        return enriched
    enriched["relevance"] = enriched["docno"].map(relevance_by_docno).fillna(0).astype(int)
    enriched["relevance_label"] = enriched["relevance"].map(
        {
            0: "0 - not judged/not relevant",
            1: "1 - relevant",
            2: "2 - highly relevant",
        }
    ).fillna(enriched["relevance"].astype(str))
    return enriched[
        [
            "rank",
            "docno",
            "relevance",
            "relevance_label",
            "score",
            "title",
            "condition",
            "summary_snippet",
        ]
    ]


def render_step_rankings(
    step_results: dict[str, pd.DataFrame],
    raw_docs: dict[str, dict[str, str]],
    relevance_by_docno: dict[str, int],
) -> None:
    if not step_results:
        return

    st.subheader("Top 10 Rankings by Step")
    st.caption("Relevance is taken from the TREC qrels for the selected query.")
    tabs = st.tabs(list(step_results.keys()))
    for tab, (step_name, results_df) in zip(tabs, step_results.items()):
        with tab:
            step_df = enrich_results_with_relevance(results_df, raw_docs, relevance_by_docno, top_n=10)
            relevant_count = int((step_df["relevance"] > 0).sum()) if not step_df.empty else 0
            highly_relevant_count = int((step_df["relevance"] >= 2).sum()) if not step_df.empty else 0
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Shown", len(step_df))
            col_b.metric("Relevant", relevant_count)
            col_c.metric("Highly relevant", highly_relevant_count)
            st.dataframe(
                step_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "score": st.column_config.NumberColumn(format="%.4f"),
                    "relevance": st.column_config.NumberColumn("qrel"),
                    "relevance_label": st.column_config.TextColumn("relevance meaning"),
                    "summary_snippet": st.column_config.TextColumn("summary"),
                },
            )


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
def load_bm25_retriever(_index_ref: Any, top_k: int, corpus_key: str) -> Any:
    return backend.pt.BatchRetrieve(
        _index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 1.5, "bm25.b": 0.75},
        num_results=top_k,
    )


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
    rebuild_index: bool,
    dense_text_mode: str,
    dense_device: str,
    dense_batch_size: int,
    model_offline: bool,
) -> dict[str, Any]:
    bm25_resources = load_bm25_resources(max_docs, rebuild_index)
    dense_doc_df = backend.get_dense_docs_df(
        max_docs,
        mode=dense_text_mode,
        cache_dir=PREPARED_CACHE_DIR,
        rebuild_prepared=False,
        raw_docs=bm25_resources["raw_docs"],
    )
    dense_model = backend.load_dense_model(
        "NeuML/pubmedbert-base-embeddings",
        dense_device,
        cache_dir=MODEL_CACHE_DIR,
        offline=model_offline,
    )
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
def load_reranker(rerank_device: str, model_offline: bool) -> Any:
    return backend.load_reranker(
        RERANKER_MODEL,
        rerank_device,
        cache_dir=MODEL_CACHE_DIR,
        offline=model_offline,
    )


def search_bm25(bm25_retriever: Any, query_text: str) -> pd.DataFrame:
    query_df = pd.DataFrame([{"qid": "user_query", "query": query_text}])
    query_df["query"] = query_df["query"].apply(backend.normalize_query_text)
    results_df = bm25_retriever.transform(query_df)
    return results_df.sort_values(["qid", "rank"]).reset_index(drop=True)


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
    raw_docs: dict[str, dict[str, str]],
    query_text: str,
    top_k: int,
    query_id: str,
) -> pd.DataFrame:
    query_df = pd.DataFrame([{"qid": query_id, "query": query_text}])
    candidate_df = candidate_df.copy()
    candidate_df["qid"] = query_id
    reranked = backend.rerank_candidates(
        reranker,
        candidate_df,
        query_df,
        docs_df,
        raw_docs,
        RERANK_CACHE_PATH,
        batch_size=8,
        rerank_alpha=None,
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
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]], dict[str, Any]]:
    queries_df = bm25_resources["queries_df"]
    qrels = bm25_resources["qrels"]
    thresholded_qrels = backend.build_thresholded_qrels(qrels)
    metrics_by_name: dict[str, dict[str, dict[str, float]]] = {}
    run_info: dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "tune_rrf": tune_rrf,
        "rrf_k": rrf_k,
        "alpha_bm25": alpha_bm25,
        "alpha_dense": 1.0 - alpha_bm25,
        "rrf_source": "manual",
    }

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
            final_results, best_trial, _ = backend.tune_weighted_rrf(
                bm25_results,
                dense_results,
                thresholded_qrels["rel>=1"],
                top_k,
            )
            run_info.update(
                {
                    "rrf_k": int(best_trial["rrf_k"]),
                    "alpha_bm25": float(best_trial["alpha_bm25"]),
                    "alpha_dense": float(best_trial["alpha_dense"]),
                    "rrf_source": "tuned rel>=1 MRR",
                    "rrf_tuned_mrr": float(best_trial["MRR"]),
                    "rrf_tuned_map": float(best_trial["MAP"]),
                }
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
            bm25_resources["raw_docs"],
            RERANK_CACHE_PATH,
            batch_size=8,
            rerank_alpha=None,
        )
        final_results = final_results[final_results["rank"] <= top_k].copy()
        evaluate_named_run("Reranked", final_results)

    return final_results, metrics_by_name, run_info


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


def log_app_init(message: str, status_box: Any | None = None) -> None:
    console_message = f"[app init] {message}"
    print(console_message, flush=True)
    if status_box is not None:
        status_box.write(message)


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
    model_offline = st.checkbox("Use cached models only", value=False)
    rrf_k = st.number_input("RRF k", min_value=1, max_value=500, value=60, step=1)
    alpha_bm25 = st.slider("BM25 fusion weight", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    tune_rrf = st.checkbox("Tune RRF (existing TREC query only)", value=False)

with st.status("Initializing backend resources...", expanded=True) as init_status:
    log_app_init("Loading BM25 documents, qrels, and BM25 index.", init_status)
    preloaded_bm25_resources = load_bm25_resources(max_docs=max_docs, rebuild_index=rebuild_index)
    log_app_init(
        f"BM25 resources ready: {len(preloaded_bm25_resources['docs_df'])} documents loaded.",
        init_status,
    )

    log_app_init(f"Preparing cached BM25 retriever for top_k={top_k}.", init_status)
    preloaded_bm25_retriever = load_bm25_retriever(
        preloaded_bm25_resources["index_ref"],
        top_k,
        corpus_key=f"{max_docs_choice}|rebuild={rebuild_index}",
    )
    log_app_init("BM25 retriever ready.", init_status)

    preloaded_dense_resources = None
    if retrieval_mode in {"Dense", "Fused", "Reranked"}:
        log_app_init(
            "Loading dense model, dense document embeddings, and FAISS index.",
            init_status,
        )
        preloaded_dense_resources = load_dense_resources(
            max_docs=max_docs,
            rebuild_index=rebuild_index,
            dense_text_mode=dense_text_mode,
            dense_device=dense_device,
            dense_batch_size=32,
            model_offline=model_offline,
        )
        log_app_init(
            f"Dense resources ready: {len(preloaded_dense_resources['dense_docnos'])} vectors loaded.",
            init_status,
        )

    preloaded_reranker = None
    if retrieval_mode == "Reranked":
        log_app_init("Loading cross-encoder reranker model.", init_status)
        preloaded_reranker = load_reranker(rerank_device, model_offline)
        log_app_init("Reranker ready.", init_status)

    log_app_init("Backend initialization complete.", init_status)
    init_status.update(label="Backend resources ready.", state="complete", expanded=False)

query_loader_resources = preloaded_bm25_resources

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
        update_progress(progress_bar, status_box, 1, f"0/{len(stage_plan)} - Starting search...")
        details_box.caption(f"Mode: {retrieval_mode}")
        render_active_stage(
            active_stage_box,
            "Loading BM25 documents, qrels, and index",
            "rebuilding index if the checkbox is enabled",
        )

        bm25_resources = preloaded_bm25_resources
        bm25_retriever = preloaded_bm25_retriever
        advance_stage("BM25 preparation", f"{len(bm25_resources['docs_df'])} documents and BM25 index ready")

        render_active_stage(active_stage_box, "Running BM25 retrieval", f"top_k={top_k}")
        bm25_results = search_bm25(bm25_retriever, query_text)
        final_results = bm25_results
        dense_results = None
        fused_results = None
        step_results: dict[str, pd.DataFrame] = {"BM25": bm25_results}
        advance_stage("BM25 ranking", f"{len(bm25_results)} rows returned")

        if retrieval_mode in {"Dense", "Fused", "Reranked"}:
            render_active_stage(
                active_stage_box,
                "Loading dense model, embeddings, and FAISS index",
                f"text mode={dense_text_mode}; uses cached embeddings when available",
            )
            dense_resources = preloaded_dense_resources
            advance_stage("Dense preparation", f"{len(dense_resources['dense_docnos'])} dense vectors ready")

            render_active_stage(active_stage_box, "Dense retrieval", f"top_k={top_k}")
            dense_results = search_dense(
                dense_resources["dense_model"],
                dense_resources["faiss_index"],
                dense_resources["dense_docnos"],
                query_text,
                top_k,
            )
            final_results = dense_results
            step_results["Dense"] = dense_results
            advance_stage("Dense ranking", f"{len(dense_results)} rows returned")

        if retrieval_mode in {"Fused", "Reranked"}:
            if tune_rrf and selected_qid is not None:
                render_active_stage(active_stage_box, "Tuning weighted RRF for this TREC query", f"query={selected_qid}")
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
                    "Fusing",
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
                render_active_stage(active_stage_box, "Fusing BM25 and dense results", f"rrf_k={rrf_k}, alpha_bm25={alpha_bm25:.1f}")
                fused_results = search_fused(
                    bm25_results,
                    dense_results,
                    top_k=top_k,
                    rrf_k=rrf_k,
                    alpha_bm25=alpha_bm25,
                )
                advance_stage("Fusing", f"{len(fused_results)} rows returned")
            final_results = fused_results
            step_results["Fused"] = fused_results

        if retrieval_mode == "Reranked":
            render_active_stage(active_stage_box, "Loading cross-encoder reranker", rerank_device)
            reranker = preloaded_reranker
            advance_stage("Reranking preparation", "cross-encoder model ready")
            render_active_stage(active_stage_box, "Reranking fused candidates", f"top_k={top_k}")
            rerank_query_id = selected_qid or (
                "custom_" + hashlib.sha1(backend.normalize_query_text(query_text).encode("utf-8")).hexdigest()[:12]
            )
            final_results = search_reranked(
                fused_results,
                reranker,
                bm25_resources["docs_df"],
                bm25_resources["raw_docs"],
                query_text,
                top_k=top_k,
                query_id=rerank_query_id,
            )
            step_results["Reranked"] = final_results
            advance_stage("Reranking", f"{len(final_results)} rows returned")

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
        relevance_by_docno = bm25_resources["qrels"].get(selected_qid, {})
        render_step_rankings(step_results, bm25_resources["raw_docs"], relevance_by_docno)

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
        update_progress(progress_bar, status_box, 1, f"0/{len(stage_plan)} - Starting evaluation...")
        render_active_stage(
            active_stage_box,
            "Loading BM25 documents, qrels, and index",
            "rebuilding index if the checkbox is enabled",
        )
        bm25_resources = preloaded_bm25_resources
        advance_stage("BM25 preparation", f"{len(bm25_resources['docs_df'])} documents and BM25 index ready")

        queries_df = bm25_resources["queries_df"]
        thresholded_qrels = backend.build_thresholded_qrels(bm25_resources["qrels"])
        result_sets: dict[str, pd.DataFrame] = {}
        eval_run_info: dict[str, Any] = {
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "tune_rrf": tune_rrf,
            "rrf_k": rrf_k,
            "alpha_bm25": alpha_bm25,
            "alpha_dense": 1.0 - alpha_bm25,
            "rrf_source": "manual",
        }

        render_active_stage(active_stage_box, "BM25 ranking", f"{len(queries_df)} queries, top_k={top_k}")
        bm25_queries_df = queries_df.copy()
        bm25_queries_df["query"] = bm25_queries_df["query"].apply(backend.normalize_query_text)
        bm25_results = preloaded_bm25_retriever.transform(bm25_queries_df)
        bm25_results = bm25_results.sort_values(["qid", "rank"]).reset_index(drop=True)
        result_sets["BM25"] = bm25_results
        final_results = bm25_results
        advance_stage("BM25 ranking", f"{len(bm25_results)} rows returned")

        dense_resources = None
        dense_results = None
        if retrieval_mode in {"Dense", "Fused", "Reranked"}:
            render_active_stage(
                active_stage_box,
                "Loading dense model, embeddings, and FAISS index",
                f"text mode={dense_text_mode}; uses cached embeddings when available",
            )
            dense_resources = preloaded_dense_resources
            advance_stage("Dense preparation", f"{len(dense_resources['dense_docnos'])} dense vectors ready")

            render_active_stage(active_stage_box, "Dense ranking", f"{len(queries_df)} queries, top_k={top_k}")
            dense_results = search_dense_all_queries(
                dense_resources["dense_model"],
                dense_resources["faiss_index"],
                dense_resources["dense_docnos"],
                queries_df,
                top_k,
            )
            result_sets["Dense"] = dense_results
            final_results = dense_results
            advance_stage("Dense ranking", f"{len(dense_results)} rows returned")

        if retrieval_mode in {"Fused", "Reranked"} and dense_results is not None:
            fusion_detail = "tuning weighted RRF on rel>=1 qrels" if tune_rrf else f"rrf_k={rrf_k}, alpha_bm25={alpha_bm25:.2f}"
            render_active_stage(active_stage_box, "Fusing BM25 and dense rankings", fusion_detail)
            if tune_rrf:
                final_results, best_trial, _ = backend.tune_weighted_rrf(
                    bm25_results,
                    dense_results,
                    thresholded_qrels["rel>=1"],
                    top_k,
                )
                eval_run_info.update(
                    {
                        "rrf_k": int(best_trial["rrf_k"]),
                        "alpha_bm25": float(best_trial["alpha_bm25"]),
                        "alpha_dense": float(best_trial["alpha_dense"]),
                        "rrf_source": "tuned rel>=1 MRR",
                    }
                )
            else:
                final_results = search_fused(
                    bm25_results,
                    dense_results,
                    top_k=top_k,
                    rrf_k=rrf_k,
                    alpha_bm25=alpha_bm25,
                )
            result_sets["Fused"] = final_results
            advance_stage("Fusing", f"{len(final_results)} rows returned")

        if retrieval_mode == "Reranked":
            render_active_stage(active_stage_box, "Loading cross-encoder reranker", rerank_device)
            reranker = preloaded_reranker
            advance_stage("Reranking preparation", "cross-encoder model ready")

            render_active_stage(active_stage_box, "Reranking", f"{len(final_results)} candidates")
            final_results = backend.rerank_candidates(
                reranker,
                final_results,
                queries_df,
                bm25_resources["docs_df"],
                bm25_resources["raw_docs"],
                RERANK_CACHE_PATH,
                batch_size=8,
                rerank_alpha=None,
            )
            final_results = final_results[final_results["rank"] <= top_k].copy()
            result_sets["Reranked"] = final_results
            advance_stage("Reranking", f"{len(final_results)} rows returned")

        render_active_stage(active_stage_box, "Evaluations", "computing rel>=1 and rel>=2 metrics")
        eval_metrics = {}
        for system_name, results_df in result_sets.items():
            eval_metrics[system_name] = {
                "rel>=1": backend.evaluate_run(results_df, thresholded_qrels["rel>=1"], top_k),
                "rel>=2": backend.evaluate_run(results_df, thresholded_qrels["rel>=2"], top_k),
            }
        advance_stage("Evaluations", f"computed metrics for {', '.join(result_sets)}")
        clear_active_stage(active_stage_box)
    finally:
        backend.set_progress_callback(None)
        clear_active_stage(active_stage_box)

    details_box.caption(f"Total evaluation workflow time: {format_elapsed(time.perf_counter() - workflow_started_at)}")

    st.subheader("Evaluation Comparison")
    eval_table = metrics_to_table(eval_metrics)
    available_systems = ", ".join(eval_table["system"].drop_duplicates().tolist())
    st.caption(f"Included systems: {available_systems}")

    st.markdown("**Run settings**")
    settings_cols = st.columns(3)
    settings_cols[0].metric("Mode", eval_run_info["retrieval_mode"])
    settings_cols[1].metric("Top-k", eval_run_info["top_k"])
    if eval_run_info["retrieval_mode"] in {"Fused", "Reranked"}:
        settings_cols[2].metric("RRF k", eval_run_info["rrf_k"])
        st.caption(f"alpha_bm25={eval_run_info['alpha_bm25']:.2f}")
        st.caption(f"alpha_dense={eval_run_info['alpha_dense']:.2f}")
    else:
        settings_cols[2].metric("Fusion", "Not used")

    st.dataframe(
        eval_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAP": st.column_config.NumberColumn(format="%.4f"),
            "MRR": st.column_config.NumberColumn(format="%.4f"),
        },
    )
