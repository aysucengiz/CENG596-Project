from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = REPO_ROOT / "runtime"

import faiss
import ir_datasets
import numpy as np
import pandas as pd
import pyterrier as pt
import torch
from tqdm.auto import tqdm


DOC_DATASET_ID = "clinicaltrials/2021"
QUERY_DATASET_ID = "clinicaltrials/2021/trec-ct-2021"
PROGRESS_CALLBACK = None
BM25_BATCH_SIZE = 25
RERANK_CACHE_VERSION = "structured_summary_v2"


def log_step(message: str) -> None:
    tqdm.write(f"\n=== {message} ===")


def log_info(message: str) -> None:
    tqdm.write(f"  -> {message}")


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {secs:.1f}s"


def set_progress_callback(callback: Any | None) -> None:
    global PROGRESS_CALLBACK
    PROGRESS_CALLBACK = callback


def emit_progress(percent: int, message: str, detail: str | None = None) -> None:
    bounded_percent = max(0, min(100, int(percent)))
    progress_line = f"[{bounded_percent:>3}%] {message}"
    if detail:
        progress_line = f"{progress_line} - {detail}"
    tqdm.write(progress_line)
    if PROGRESS_CALLBACK is not None:
        PROGRESS_CALLBACK(bounded_percent, message, detail)


class PipelineProgress:
    def __init__(self, stages: list[str]) -> None:
        self.stages = stages
        self.total = max(1, len(stages))
        self.current = 0

    def _stage_label(self, stage_name: str) -> str:
        return f"Stage {self.current + 1}/{self.total}: {stage_name}"

    def start(self, stage_name: str, detail: str | None = None) -> None:
        start_percent = int((self.current / self.total) * 100)
        emit_progress(start_percent, self._stage_label(stage_name), detail or "starting")

    def finish(self, stage_name: str, detail: str | None = None) -> None:
        completed_index = self.current + 1
        end_percent = int((completed_index / self.total) * 100)
        emit_progress(end_percent, f"Stage {completed_index}/{self.total}: {stage_name}", detail or "done")
        self.current = completed_index

    @contextmanager
    def stage(self, stage_name: str, detail: str | None = None):
        self.start(stage_name, detail)
        try:
            yield
        finally:
            self.finish(stage_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local BM25 + dense retrieval + fusion + reranking pipeline "
        "for the TREC Clinical Trials dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUNTIME_ROOT,
        help="Directory for indexes, runs, caches, and metrics.",
    )
    parser.add_argument(
        "--ir-datasets-home",
        type=Path,
        default=RUNTIME_ROOT / "ir_datasets_cache",
        help="Local cache directory for ir_datasets.",
    )
    parser.add_argument(
        "--prepared-cache-dir",
        type=Path,
        default=RUNTIME_ROOT / "prepared_cache",
        help="Directory for cached prepared dataframes and metadata.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents for faster local tests.",
    )
    parser.add_argument("--top-k", type=int, default=1000, help="Top-k candidates per retriever.")
    parser.add_argument("--rrf-k", type=int, default=60, help="Reciprocal Rank Fusion k value.")
    parser.add_argument(
        "--tune-rrf",
        action="store_true",
        help="Tune weighted RRF over a small grid and keep the best fused result.",
    )
    parser.add_argument(
        "--bm25-index-dir",
        type=Path,
        default=None,
        help="Optional explicit BM25 index directory.",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=RUNTIME_ROOT / "embeddings",
        help="Directory for persistent dense embedding cache files.",
    )
    parser.add_argument(
        "--dense-model",
        type=str,
        default="NeuML/pubmedbert-base-embeddings",
        help="SentenceTransformer model for dense retrieval.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="ncbi/MedCPT-Cross-Encoder",
        help="CrossEncoder model for reranking.",
    )
    parser.add_argument("--dense-batch-size", type=int, default=32)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument(
        "--rerank-alpha",
        type=float,
        default=0.8,
        help="Fuse reranker score with retrieval score; retrieval score gets 1-alpha.",
    )
    parser.add_argument(
        "--dense-device",
        type=str,
        default=None,
        help="Override device for dense retrieval model, e.g. cpu, cuda, xpu.",
    )
    parser.add_argument(
        "--rerank-device",
        type=str,
        default=None,
        help="Override device for reranker model, e.g. cpu, cuda, xpu.",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense retrieval and only run BM25.",
    )
    parser.add_argument(
        "--skip-rerank",
        action="store_true",
        help="Skip cross-encoder reranking.",
    )
    parser.add_argument(
        "--reuse-dense-cache",
        action="store_true",
        help="Reuse cached dense document embeddings if present.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuilding the BM25 index even if it already exists on disk.",
    )
    parser.add_argument(
        "--rebuild-prepared",
        action="store_true",
        help="Force rebuilding prepared document caches instead of loading them from disk.",
    )
    parser.add_argument(
        "--dense-text-mode",
        choices=["short", "full"],
        default="short",
        help="Dense text uses title+condition+summary+eligibility in short mode, "
        "or the full BM25 text in full mode.",
    )
    return parser.parse_args()


def select_device() -> tuple[Any, str]:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", "torch-xpu"

    try:
        import torch_directml  # type: ignore

        return torch_directml.device(), "directml"
    except Exception:
        pass

    if torch.cuda.is_available():
        return "cuda", "cuda"

    return "cpu", "cpu"


def resolve_device(override: str | None, default_device: Any) -> Any:
    if override is None:
        return default_device
    return override


def ensure_pyterrier() -> None:
    log_step("Initializing PyTerrier/Java")
    if not pt.java.started():
        pt.java.init()
        log_info("PyTerrier Java backend started")
    else:
        log_info("PyTerrier Java backend already running")


def normalize_query_text(text: str) -> str:
    return " ".join(text.lower().split())


def build_bm25_document_text(doc: Any) -> tuple[str, dict[str, str]]:
    title = getattr(doc, "title", "") or ""
    condition = getattr(doc, "condition", "") or ""
    summary = getattr(doc, "summary", "") or ""
    detailed_description = getattr(doc, "detailed_description", "") or ""
    eligibility = getattr(doc, "eligibility", "") or ""

    joined = " ".join(
        [title, condition, summary, detailed_description, eligibility]
    ).strip()

    raw_doc = {
        "title": title,
        "condition": condition,
        "summary": summary,
        "detailed_description": detailed_description,
        "eligibility": eligibility,
    }
    return joined, raw_doc


def build_dense_document_text(doc: Any, mode: str) -> str:
    if mode == "full":
        return build_bm25_document_text(doc)[0]

    title = getattr(doc, "title", "") or ""
    condition = getattr(doc, "condition", "") or ""
    summary = getattr(doc, "summary", "") or ""
    eligibility = getattr(doc, "eligibility", "") or ""
    return " ".join([title, condition, summary, eligibility]).strip()


def build_dense_document_text_from_raw(raw_doc: dict[str, str], mode: str) -> str:
    title = raw_doc.get("title", "") or ""
    condition = raw_doc.get("condition", "") or ""
    summary = raw_doc.get("summary", "") or ""
    detailed_description = raw_doc.get("detailed_description", "") or ""
    eligibility = raw_doc.get("eligibility", "") or ""

    if mode == "full":
        return " ".join([title, condition, summary, detailed_description, eligibility]).strip()

    return " ".join([title, condition, summary, eligibility]).strip()


def shorten(text: str | None, max_chars: int = 1200) -> str:
    return text[:max_chars] if text else ""


def extract_structured_eligibility(eligibility_text: Any, max_items: int = 4, max_chars: int = 1200) -> str:
    if eligibility_text is None or pd.isna(eligibility_text):
        return ""

    eligibility_text = str(eligibility_text)
    if not eligibility_text:
        return ""

    text_lower = eligibility_text.lower()
    inc_idx = text_lower.find("inclusion criteria")
    exc_idx = text_lower.find("exclusion criteria")

    inclusion_raw = ""
    exclusion_raw = ""

    if inc_idx != -1 and exc_idx != -1:
        if inc_idx < exc_idx:
            inclusion_raw = eligibility_text[inc_idx:exc_idx]
            exclusion_raw = eligibility_text[exc_idx:]
        else:
            exclusion_raw = eligibility_text[exc_idx:inc_idx]
            inclusion_raw = eligibility_text[inc_idx:]
    elif inc_idx != -1:
        inclusion_raw = eligibility_text[inc_idx:]
    elif exc_idx != -1:
        exclusion_raw = eligibility_text[exc_idx:]
    else:
        return str(eligibility_text)[:max_chars]

    def get_top_rules(text_block: str, block_label: str, items_count: int) -> str:
        if not text_block:
            return ""
        lines = [line.strip() for line in re.split(r"\n|-|\u2022|\*", text_block) if len(line.strip()) > 10]
        extracted_lines = lines[: items_count + 1]
        return f"[{block_label}] " + " | ".join(extracted_lines)

    inc_summary = get_top_rules(inclusion_raw, "INCLUSION", max_items)
    exc_summary = get_top_rules(exclusion_raw, "EXCLUSION", max_items)

    final_text = f"{inc_summary}  {exc_summary}".strip()
    return final_text[:max_chars]


def build_rerank_doc_text(raw_doc: dict[str, str]) -> str:
    parts = []

    if raw_doc.get("title"):
        parts.append("Title: " + str(raw_doc["title"])[:200])

    if raw_doc.get("condition"):
        parts.append("Condition: " + str(raw_doc["condition"])[:200])

    if raw_doc.get("summary"):
        parts.append("Summary: " + shorten(str(raw_doc["summary"]), 700))

    if raw_doc.get("eligibility"):
        structured_eligibility = extract_structured_eligibility(raw_doc["eligibility"])
        parts.append("Eligibility: " + structured_eligibility)

    return " | ".join(parts).strip()


def cache_suffix(max_docs: int | None, mode: str | None = None) -> str:
    base = "all" if max_docs is None else str(max_docs)
    return f"{base}_{mode}" if mode else base


def safe_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def minmax_normalize_by_query(df: pd.DataFrame, column: str) -> pd.Series:
    min_score = df.groupby("qid")[column].transform("min")
    max_score = df.groupby("qid")[column].transform("max")
    denom = (max_score - min_score).replace(0, 1)
    return ((df[column] - min_score) / denom).fillna(0.0)


def get_docs_df(
    max_docs: int | None,
    cache_dir: Path,
    rebuild_prepared: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    suffix = cache_suffix(max_docs)
    docs_cache_path = cache_dir / f"docs_df_{suffix}.parquet"
    raw_docs_cache_path = cache_dir / f"raw_docs_{suffix}.pkl"

    if docs_cache_path.exists() and raw_docs_cache_path.exists() and not rebuild_prepared:
        log_step(f"Loading cached BM25 documents from {cache_dir}")
        docs_df = pd.read_parquet(docs_cache_path)
        with raw_docs_cache_path.open("rb") as handle:
            raw_docs = pickle.load(handle)
        log_info(f"Loaded cached BM25 documents: {len(docs_df)}")
        return docs_df, raw_docs

    log_step(f"Loading BM25 documents from {DOC_DATASET_ID}")
    dataset = ir_datasets.load(DOC_DATASET_ID)
    total_docs = max_docs if max_docs is not None else dataset.docs_count()
    log_info(f"Target BM25 document count: {total_docs}")

    rows: list[dict[str, str]] = []
    raw_docs: dict[str, dict[str, str]] = {}

    for i, doc in enumerate(tqdm(dataset.docs_iter(), total=total_docs, desc="Preparing BM25 docs")):
        if max_docs is not None and i >= max_docs:
            break

        docno = doc.doc_id
        text, raw_doc = build_bm25_document_text(doc)
        if not text:
            continue

        rows.append({"docno": docno, "text": text})
        raw_docs[docno] = raw_doc

    docs_df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs_df.to_parquet(docs_cache_path, index=False)
    with raw_docs_cache_path.open("wb") as handle:
        pickle.dump(raw_docs, handle)
    log_info(f"Finished BM25 document preparation with {len(rows)} usable documents")
    log_info(f"Saved BM25 document cache to {docs_cache_path}")
    return docs_df, raw_docs


def get_dense_docs_df(
    max_docs: int | None,
    mode: str,
    cache_dir: Path,
    rebuild_prepared: bool,
    raw_docs: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    suffix = cache_suffix(max_docs, mode=mode)
    dense_cache_path = cache_dir / f"dense_docs_df_{suffix}.parquet"

    if dense_cache_path.exists() and not rebuild_prepared:
        log_step(f"Loading cached dense documents from {dense_cache_path}")
        dense_df = pd.read_parquet(dense_cache_path)
        log_info(f"Loaded cached dense documents: {len(dense_df)}")
        return dense_df

    if raw_docs is not None:
        log_step(f"Building dense documents from cached BM25 metadata using '{mode}' text mode")
        rows: list[dict[str, str]] = []
        for docno, raw_doc in tqdm(raw_docs.items(), total=len(raw_docs), desc="Preparing dense docs from cache"):
            text = build_dense_document_text_from_raw(raw_doc, mode=mode)
            if not text:
                continue
            rows.append({"docno": docno, "text": text})

        dense_df = pd.DataFrame(rows)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dense_df.to_parquet(dense_cache_path, index=False)
        log_info(f"Finished dense document preparation with {len(rows)} usable documents")
        log_info(f"Saved dense document cache to {dense_cache_path}")
        return dense_df

    log_step(f"Loading dense documents from {DOC_DATASET_ID} using '{mode}' text mode")
    dataset = ir_datasets.load(DOC_DATASET_ID)
    total_docs = max_docs if max_docs is not None else dataset.docs_count()
    log_info(f"Target dense document count: {total_docs}")

    rows: list[dict[str, str]] = []
    for i, doc in enumerate(tqdm(dataset.docs_iter(), total=total_docs, desc="Preparing dense docs")):
        if max_docs is not None and i >= max_docs:
            break

        docno = doc.doc_id
        text = build_dense_document_text(doc, mode=mode)
        if not text:
            continue

        rows.append({"docno": docno, "text": text})

    dense_df = pd.DataFrame(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dense_df.to_parquet(dense_cache_path, index=False)
    log_info(f"Finished dense document preparation with {len(rows)} usable documents")
    log_info(f"Saved dense document cache to {dense_cache_path}")
    return dense_df


def load_queries_and_qrels() -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    log_step(f"Loading queries and qrels from {QUERY_DATASET_ID}")
    query_dataset = ir_datasets.load(QUERY_DATASET_ID)
    queries_df = pd.DataFrame(
        {"qid": q.query_id, "query": q.text} for q in query_dataset.queries_iter()
    )

    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for qr in query_dataset.qrels_iter():
        qrels[qr.query_id][qr.doc_id] = qr.relevance

    log_info(f"Loaded {len(queries_df)} queries and qrels for {len(qrels)} query ids")
    return queries_df, qrels


def write_run(df: pd.DataFrame, output_path: Path, tag: str) -> None:
    log_step(f"Writing run file '{tag}' to {output_path}")
    with output_path.open("w", encoding="utf-8") as handle:
        for row in df.itertuples(index=False):
            handle.write(f"{row.qid} Q0 {row.docno} {int(row.rank)} {row.score} {tag}\n")
    log_info(f"Wrote {len(df)} ranked rows")


def qrels_at_threshold(qrels: dict[str, dict[str, int]], min_rel: int) -> dict[str, dict[str, int]]:
    thresholded: dict[str, dict[str, int]] = {}
    for qid, doc_map in qrels.items():
        thresholded[qid] = {docno: int(rel >= min_rel) for docno, rel in doc_map.items()}
    return thresholded


def build_thresholded_qrels(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, dict[str, int]]]:
    return {
        "rel>=1": qrels_at_threshold(qrels, 1),
        "rel>=2": qrels_at_threshold(qrels, 2),
    }


def evaluate_run(
    results_df: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    k: int,
) -> dict[str, float]:
    grouped_results = {
        qid: group.sort_values("rank")[["docno", "score"]].to_dict("records")
        for qid, group in results_df.groupby("qid")
    }

    precisions: list[float] = []
    recalls: list[float] = []
    maps: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for qid, rel_docs in qrels.items():
        retrieved = grouped_results.get(qid, [])[:k]
        relevant_set = {docno for docno, rel in rel_docs.items() if rel > 0}
        total_relevant = len(relevant_set)

        hits = 0
        ap_sum = 0.0
        dcg = 0.0
        rr = 0.0

        for idx, row in enumerate(retrieved, start=1):
            gain = 1 if row["docno"] in relevant_set else 0
            if gain:
                hits += 1
                ap_sum += hits / idx
                if rr == 0.0:
                    rr = 1.0 / idx
            dcg += gain / math.log2(idx + 1)

        ideal_hits = min(total_relevant, k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(2, ideal_hits + 2))

        precisions.append(hits / k if k else 0.0)
        recalls.append(hits / total_relevant if total_relevant else 0.0)
        maps.append(ap_sum / total_relevant if total_relevant else 0.0)
        reciprocal_ranks.append(rr)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        f"P@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"R@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "MAP": float(np.mean(maps)) if maps else 0.0,
        "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        f"nDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    tqdm.write(label)
    for key, value in metrics.items():
        tqdm.write(f"     {key:<10} {value:.4f}")


def print_metrics_summary(metrics_by_name: dict[str, dict[str, dict[str, float]]]) -> None:
    preferred_order = ["bm25", "dense", "fused_rrf", "reranked"]
    row_order = [name for name in preferred_order if name in metrics_by_name]
    row_order.extend(name for name in metrics_by_name if name not in row_order)

    for threshold in ["rel>=1", "rel>=2"]:
        available_rows = [name for name in row_order if threshold in metrics_by_name.get(name, {})]
        if not available_rows:
            continue

        tqdm.write(f"\n=== Final Summary ({threshold}) ===")
        headers = ["System", "P", "R", "MAP", "MRR", "nDCG"]
        widths = [12, 8, 8, 8, 8, 8]
        header_line = " ".join(f"{header:<{width}}" for header, width in zip(headers, widths))
        tqdm.write(header_line)
        tqdm.write("-" * len(header_line))

        for name in available_rows:
            metrics = metrics_by_name[name][threshold]
            row = [
                name,
                f"{metrics.get(next(k for k in metrics if k.startswith('P@')), 0.0):.4f}",
                f"{metrics.get(next(k for k in metrics if k.startswith('R@')), 0.0):.4f}",
                f"{metrics.get('MAP', 0.0):.4f}",
                f"{metrics.get('MRR', 0.0):.4f}",
                f"{metrics.get(next(k for k in metrics if k.startswith('nDCG@')), 0.0):.4f}",
            ]
            tqdm.write(" ".join(f"{value:<{width}}" for value, width in zip(row, widths)))


def get_index_doc_count(index_ref: Any) -> int:
    if not hasattr(index_ref, "getCollectionStatistics"):
        index_ref = pt.IndexFactory.of(index_ref)
    return int(index_ref.getCollectionStatistics().getNumberOfDocuments())


def iter_bm25_index_records(docs_df: pd.DataFrame):
    total_docs = len(docs_df)
    progress_interval = max(1, total_docs // 100)
    last_percent = -1
    rows = docs_df[["docno", "text"]].itertuples(index=False)

    with tqdm(total=total_docs, desc="Indexing BM25 docs", unit="doc") as progress_bar:
        for idx, row in enumerate(rows, start=1):
            progress_bar.update(1)
            if PROGRESS_CALLBACK is not None and (idx == 1 or idx % progress_interval == 0 or idx == total_docs):
                percent = int((idx / total_docs) * 100)
                if percent != last_percent:
                    PROGRESS_CALLBACK(
                        percent,
                        "Indexing BM25 documents",
                        f"{idx:,}/{total_docs:,} documents passed to Terrier",
                    )
                    last_percent = percent
            yield {"docno": str(row.docno), "text": str(row.text or "")}


def build_bm25_index(docs_df: pd.DataFrame, index_dir: Path, rebuild: bool) -> Any:
    data_properties = index_dir / "data.properties"
    expected_doc_count = len(docs_df)
    if data_properties.exists() and not rebuild:
        log_step(f"Reusing existing BM25 index at {index_dir}")
        index_ref = pt.IndexFactory.of(str(index_dir))
        actual_doc_count = get_index_doc_count(index_ref)
        if actual_doc_count != expected_doc_count:
            raise RuntimeError(
                "Existing BM25 index document count does not match the prepared corpus. "
                f"Index has {actual_doc_count} documents, but prepared corpus has {expected_doc_count}. "
                "Run again with --rebuild-index."
            )
        log_info(f"Loaded existing BM25 index from disk with {actual_doc_count} documents")
        return index_ref

    log_step(f"Building BM25 index at {index_dir}")
    if rebuild and index_dir.exists():
        log_info("Removing existing BM25 index directory before rebuild")
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    log_info(f"Starting Terrier indexing for {expected_doc_count:,} documents")
    indexer = pt.terrier.IterDictIndexer(str(index_dir), overwrite=True, meta={"docno": 32})
    index_ref = indexer.index(iter_bm25_index_records(docs_df))
    index_ref = pt.IndexFactory.of(index_ref)
    actual_doc_count = get_index_doc_count(index_ref)
    if actual_doc_count != expected_doc_count:
        raise RuntimeError(
            "BM25 index construction finished with the wrong document count. "
            f"Index has {actual_doc_count} documents, but expected {expected_doc_count}."
        )
    log_info(f"BM25 index construction completed with {actual_doc_count} documents")
    return index_ref


def run_bm25(index_ref: Any, queries_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    log_step(f"Running BM25 retrieval with top_k={top_k}")
    overall_start = time.perf_counter()
    bm25 = pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        controls={"bm25.k_1": 1.5, "bm25.b": 0.75},
        num_results=top_k,
    )
    bm25_queries_df = queries_df.copy()
    bm25_queries_df["query"] = bm25_queries_df["query"].apply(normalize_query_text)
    query_count = len(bm25_queries_df)
    batch_size = BM25_BATCH_SIZE if query_count > BM25_BATCH_SIZE else max(1, query_count)
    batch_total = max(1, math.ceil(query_count / batch_size))
    log_info(
        f"Submitting {query_count} BM25 queries to PyTerrier in {batch_total} batch(es) "
        f"of up to {batch_size}"
    )

    if batch_total == 1:
        log_info(f"BM25 single batch started for {query_count} queries")
        batch_start = time.perf_counter()
        results_df = bm25.transform(bm25_queries_df)
        log_info(
            f"BM25 single batch finished in {format_elapsed(time.perf_counter() - batch_start)} "
            f"with {len(results_df)} rows"
        )
    else:
        result_batches: list[pd.DataFrame] = []
        for batch_number, start_idx in enumerate(range(0, query_count, batch_size), start=1):
            end_idx = min(start_idx + batch_size, query_count)
            batch_df = bm25_queries_df.iloc[start_idx:end_idx].copy()
            batch_start = time.perf_counter()
            batch_qids = batch_df["qid"].tolist()
            log_info(
                f"BM25 batch {batch_number}/{batch_total} started "
                f"for queries {batch_qids[0]} -> {batch_qids[-1]} ({len(batch_df)} queries)"
            )
            batch_results = bm25.transform(batch_df)
            batch_elapsed = time.perf_counter() - batch_start
            log_info(
                f"BM25 batch {batch_number}/{batch_total} finished in {format_elapsed(batch_elapsed)} "
                f"with {len(batch_results)} rows"
            )
            result_batches.append(batch_results)

        results_df = pd.concat(result_batches, ignore_index=True) if result_batches else pd.DataFrame()
    results_df = results_df.sort_values(["qid", "rank"]).reset_index(drop=True)
    total_elapsed = time.perf_counter() - overall_start
    log_info(f"BM25 retrieval produced {len(results_df)} rows in {format_elapsed(total_elapsed)}")
    return results_df


def load_dense_model(model_name: str, device: Any) -> Any:
    from sentence_transformers import SentenceTransformer

    log_step(f"Loading dense model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to(device)
    log_info("Dense model loaded")
    return model


def load_reranker(model_name: str, device: Any) -> Any:
    from sentence_transformers import CrossEncoder

    log_step(f"Loading reranker model: {model_name}")
    try:
        reranker = CrossEncoder(model_name, device=device, max_length=512)
    except TypeError:
        reranker = CrossEncoder(model_name, max_length=512)
        reranker.model.to(device)
    log_info("Reranker model loaded")
    return reranker


def get_dense_embeddings(
    dense_model: Any,
    dense_doc_df: pd.DataFrame,
    emb_path: Path,
    docno_path: Path,
    batch_size: int,
    reuse_cache: bool,
) -> tuple[np.ndarray, list[str]]:
    if reuse_cache and emb_path.exists() and docno_path.exists():
        log_step(f"Loading cached dense embeddings from {emb_path}")
        cache_start = time.perf_counter()
        embeddings = np.load(emb_path)
        docnos = np.load(docno_path, allow_pickle=True).tolist()
        log_info(
            f"Loaded {len(docnos)} cached dense document ids in "
            f"{format_elapsed(time.perf_counter() - cache_start)}"
        )
        return embeddings, docnos

    log_step("Encoding dense document embeddings")
    encode_start = time.perf_counter()
    log_info(f"Encoding {len(dense_doc_df)} dense documents with batch_size={batch_size}")
    embeddings = dense_model.encode(
        dense_doc_df["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    docnos = dense_doc_df["docno"].tolist()

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    np.save(docno_path, np.array(docnos, dtype=object))
    log_info(f"Finished dense document encoding in {format_elapsed(time.perf_counter() - encode_start)}")
    log_info(f"Saved dense embedding cache to {emb_path}")
    return embeddings, docnos


def run_dense_retrieval(
    dense_model: Any,
    dense_doc_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    emb_path: Path,
    docno_path: Path,
    top_k: int,
    batch_size: int,
    reuse_cache: bool,
) -> pd.DataFrame:
    log_step("Preparing dense retrieval embeddings and FAISS index")
    overall_start = time.perf_counter()
    doc_embeddings, dense_docnos = get_dense_embeddings(
        dense_model,
        dense_doc_df,
        emb_path,
        docno_path,
        batch_size=batch_size,
        reuse_cache=reuse_cache,
    )

    faiss_index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings)
    log_info(f"FAISS index now contains {faiss_index.ntotal} vectors")

    dense_queries_df = queries_df.copy()
    dense_queries_df["query"] = dense_queries_df["query"].apply(normalize_query_text)
    log_step("Encoding dense query embeddings")
    query_encode_start = time.perf_counter()
    log_info(f"Encoding {len(dense_queries_df)} dense queries with batch_size={batch_size}")
    query_embeddings = dense_model.encode(
        dense_queries_df["query"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log_info(f"Dense query encoding finished in {format_elapsed(time.perf_counter() - query_encode_start)}")

    log_step(f"Running dense retrieval with top_k={top_k}")
    search_start = time.perf_counter()
    scores, indices = faiss_index.search(query_embeddings, top_k)
    log_info(f"FAISS search finished in {format_elapsed(time.perf_counter() - search_start)}")
    rows: list[dict[str, Any]] = []
    for q_idx, qid in enumerate(dense_queries_df["qid"].tolist()):
        for rank_idx, (doc_idx, score) in enumerate(zip(indices[q_idx], scores[q_idx]), start=1):
            rows.append(
                {
                    "qid": qid,
                    "docno": dense_docnos[doc_idx],
                    "score": float(score),
                    "rank": rank_idx,
                }
            )

    dense_df = pd.DataFrame(rows).sort_values(["qid", "rank"]).reset_index(drop=True)
    log_info(f"Dense retrieval produced {len(dense_df)} rows in {format_elapsed(time.perf_counter() - overall_start)}")
    return dense_df


def fuse_rrf(bm25_df: pd.DataFrame, dense_df: pd.DataFrame, rrf_k: int, top_k: int) -> pd.DataFrame:
    log_step(f"Fusing BM25 and dense rankings with RRF (k={rrf_k}, top_k={top_k})")
    bm25_rrf = bm25_df[["qid", "docno", "rank"]].copy()
    bm25_rrf["bm25_rrf"] = 1.0 / (rrf_k + bm25_rrf["rank"])

    dense_rrf = dense_df[["qid", "docno", "rank"]].copy()
    dense_rrf["dense_rrf"] = 1.0 / (rrf_k + dense_rrf["rank"])

    fused_df = bm25_rrf.merge(dense_rrf, on=["qid", "docno"], how="outer")
    fused_df["bm25_rrf"] = fused_df["bm25_rrf"].fillna(0.0)
    fused_df["dense_rrf"] = fused_df["dense_rrf"].fillna(0.0)
    fused_df["score"] = fused_df["bm25_rrf"] + fused_df["dense_rrf"]
    fused_df = fused_df.sort_values(["qid", "score"], ascending=[True, False]).copy()
    fused_df["rank"] = fused_df.groupby("qid").cumcount() + 1
    fused_df = fused_df[fused_df["rank"] <= top_k].reset_index(drop=True)
    log_info(f"Fusion produced {len(fused_df)} rows")
    return fused_df


def fuse_weighted_rrf(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    rrf_k: int,
    top_k: int,
    alpha_bm25: float,
    alpha_dense: float,
) -> pd.DataFrame:
    bm25_rrf = bm25_df[["qid", "docno", "rank"]].copy()
    bm25_rrf["bm25_rrf"] = alpha_bm25 * (1.0 / (rrf_k + bm25_rrf["rank"]))

    dense_rrf = dense_df[["qid", "docno", "rank"]].copy()
    dense_rrf["dense_rrf"] = alpha_dense * (1.0 / (rrf_k + dense_rrf["rank"]))

    fused_df = bm25_rrf.merge(dense_rrf, on=["qid", "docno"], how="outer")
    fused_df["bm25_rrf"] = fused_df["bm25_rrf"].fillna(0.0)
    fused_df["dense_rrf"] = fused_df["dense_rrf"].fillna(0.0)
    fused_df["score"] = fused_df["bm25_rrf"] + fused_df["dense_rrf"]
    fused_df = fused_df.sort_values(["qid", "score"], ascending=[True, False]).copy()
    fused_df["rank"] = fused_df.groupby("qid").cumcount() + 1
    return fused_df[fused_df["rank"] <= top_k].reset_index(drop=True)


def build_union_candidates(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    fused_df: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    log_step("Building BM25+dense union candidate pool for reranking")
    bm25_pool = bm25_df[bm25_df["rank"] <= top_k][["qid", "docno", "score", "rank"]].copy()
    dense_pool = dense_df[dense_df["rank"] <= top_k][["qid", "docno", "score", "rank"]].copy()
    fused_pool = fused_df[fused_df["rank"] <= top_k][["qid", "docno", "score", "rank"]].copy()

    bm25_pool = bm25_pool.rename(columns={"score": "bm25_score", "rank": "bm25_rank"})
    dense_pool = dense_pool.rename(columns={"score": "dense_score", "rank": "dense_rank"})
    fused_pool = fused_pool.rename(columns={"score": "fused_score", "rank": "fused_rank"})

    candidate_pool = bm25_pool.merge(dense_pool, on=["qid", "docno"], how="outer")
    candidate_pool = candidate_pool.merge(fused_pool, on=["qid", "docno"], how="outer")
    candidate_pool["bm25_score_norm"] = minmax_normalize_by_query(candidate_pool, "bm25_score")
    candidate_pool["dense_score_norm"] = minmax_normalize_by_query(candidate_pool, "dense_score")
    candidate_pool["fused_score_norm"] = minmax_normalize_by_query(candidate_pool, "fused_score")
    candidate_pool["score"] = candidate_pool[
        ["fused_score_norm", "dense_score_norm", "bm25_score_norm"]
    ].max(axis=1)
    candidate_pool = candidate_pool.sort_values(["qid", "score"], ascending=[True, False]).copy()
    candidate_pool["rank"] = candidate_pool.groupby("qid").cumcount() + 1
    candidate_pool = candidate_pool[candidate_pool["rank"] <= top_k].reset_index(drop=True)
    log_info(f"Union candidate pool contains {len(candidate_pool)} rows")
    return candidate_pool[["qid", "docno", "score", "rank"]]


def tune_weighted_rrf(
    bm25_df: pd.DataFrame,
    dense_df: pd.DataFrame,
    qrels: dict[str, dict[str, int]],
    top_k: int,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    log_step("Tuning weighted RRF fusion")
    rrf_k_values = [10, 30, 60]
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    total_trials = len(rrf_k_values) * len(alpha_values)
    tune_start = time.perf_counter()
    log_info(f"Running {total_trials} weighted RRF trials")

    trials: list[dict[str, float]] = []
    best_trial: dict[str, float] | None = None
    best_fused_df: pd.DataFrame | None = None

    trial_number = 0
    for rrf_k in rrf_k_values:
        for alpha_bm25 in alpha_values:
            trial_number += 1
            alpha_dense = 1.0 - alpha_bm25
            log_info(
                f"RRF trial {trial_number}/{total_trials}: "
                f"rrf_k={rrf_k}, alpha_bm25={alpha_bm25:.1f}, alpha_dense={alpha_dense:.1f}"
            )
            fused_df = fuse_weighted_rrf(
                bm25_df,
                dense_df,
                rrf_k=rrf_k,
                top_k=top_k,
                alpha_bm25=alpha_bm25,
                alpha_dense=alpha_dense,
            )
            metrics = evaluate_run(fused_df, qrels, top_k)
            trial = {
                "rrf_k": float(rrf_k),
                "alpha_bm25": float(alpha_bm25),
                "alpha_dense": float(alpha_dense),
                "MAP": metrics["MAP"],
                "MRR": metrics["MRR"],
            }
            trials.append(trial)

            if best_trial is None or trial["MRR"] > best_trial["MRR"] or (
                trial["MRR"] == best_trial["MRR"] and trial["MAP"] > best_trial["MAP"]
            ):
                best_trial = trial
                best_fused_df = fused_df
                log_info(
                    f"New best RRF trial at {trial_number}/{total_trials} "
                    f"with MRR={trial['MRR']:.4f}, MAP={trial['MAP']:.4f}"
                )

    assert best_trial is not None
    assert best_fused_df is not None

    trials_df = pd.DataFrame(trials).sort_values(["MRR", "MAP"], ascending=[False, False]).reset_index(drop=True)
    log_info(
        "Best weighted RRF setting: "
        f"rrf_k={int(best_trial['rrf_k'])}, "
        f"alpha_bm25={best_trial['alpha_bm25']:.1f}, "
        f"alpha_dense={best_trial['alpha_dense']:.1f}, "
        f"MRR={best_trial['MRR']:.4f}, MAP={best_trial['MAP']:.4f}"
    )
    log_info(f"Weighted RRF tuning finished in {format_elapsed(time.perf_counter() - tune_start)}")
    return best_fused_df, best_trial, trials_df


def rerank_candidates(
    reranker: Any,
    candidate_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    docs_df: pd.DataFrame,
    raw_docs: dict[str, dict[str, str]],
    cache_path: Path,
    batch_size: int,
    rerank_alpha: float | None,
) -> pd.DataFrame:
    log_step("Preparing candidate pairs for reranking")
    rerank_start = time.perf_counter()
    query_map = dict(zip(queries_df["qid"], queries_df["query"]))
    fallback_doc_map = dict(zip(docs_df["docno"], docs_df["text"]))
    doc_map = {
        docno: build_rerank_doc_text(raw_doc) or fallback_doc_map.get(docno, "")
        for docno, raw_doc in raw_docs.items()
    }

    rerank_df = candidate_df.copy()
    rerank_df["retrieval_score"] = pd.to_numeric(rerank_df["score"], errors="coerce").fillna(0.0)
    rerank_df["query_text"] = rerank_df["qid"].map(query_map).map(normalize_query_text)
    rerank_df["doc_text"] = rerank_df["docno"].map(doc_map)
    rerank_df = rerank_df.dropna(subset=["query_text", "doc_text"]).copy()
    log_info(f"Reranking will score {len(rerank_df)} query-document pairs")

    cache_columns = ["qid", "docno", "rerank_score"]
    if cache_path.exists():
        log_step(f"Loading reranker score cache from {cache_path}")
        try:
            cache_df = pd.read_parquet(cache_path)
            if not set(cache_columns).issubset(cache_df.columns):
                log_info("Reranker cache is missing required columns; rebuilding cache entries")
                cache_df = pd.DataFrame(columns=cache_columns)
            else:
                cache_df = cache_df[cache_columns].drop_duplicates(["qid", "docno"], keep="last")
                log_info(f"Loaded {len(cache_df)} cached reranker scores")
        except Exception as exc:
            log_info(f"Could not load reranker cache ({exc}); rebuilding cache entries")
            cache_df = pd.DataFrame(columns=cache_columns)
    else:
        cache_df = pd.DataFrame(columns=cache_columns)

    rerank_df = rerank_df.merge(cache_df, on=["qid", "docno"], how="left")
    missing_mask = rerank_df["rerank_score"].isna()
    missing_count = int(missing_mask.sum())
    cached_count = len(rerank_df) - missing_count
    log_info(f"Reranker cache hits: {cached_count}; cache misses: {missing_count}")

    if missing_count:
        missing_df = rerank_df.loc[missing_mask].copy()
        pairs = list(zip(missing_df["query_text"], missing_df["doc_text"]))
        log_step("Running cross-encoder reranking for cache misses")
        log_info(f"Sending {len(pairs)} query-document pairs to the reranker with batch_size={batch_size}")
        rerank_scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True)
        rerank_df.loc[missing_mask, "rerank_score"] = rerank_scores

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        updated_cache_df = pd.concat(
            [cache_df, rerank_df[cache_columns]],
            ignore_index=True,
        ).drop_duplicates(["qid", "docno"], keep="last")
        updated_cache_df.to_parquet(cache_path, index=False)
        log_info(f"Saved {len(updated_cache_df)} reranker scores to cache")
    else:
        log_info("All reranker scores were loaded from cache")

    rerank_df["rerank_score"] = pd.to_numeric(rerank_df["rerank_score"], errors="coerce").fillna(0.0)
    if rerank_alpha is None:
        rerank_df["score"] = rerank_df["rerank_score"]
        log_info("Final rerank score uses reranker score only")
    else:
        rerank_alpha = max(0.0, min(1.0, rerank_alpha))
        retrieval_alpha = 1.0 - rerank_alpha
        rerank_df["retrieval_score_norm"] = minmax_normalize_by_query(rerank_df, "retrieval_score")
        rerank_df["rerank_score_norm"] = minmax_normalize_by_query(rerank_df, "rerank_score")
        rerank_df["score"] = (
            rerank_alpha * rerank_df["rerank_score_norm"]
            + retrieval_alpha * rerank_df["retrieval_score_norm"]
        )
        log_info(
            f"Final rerank score uses reranker weight={rerank_alpha:.2f} "
            f"and retrieval weight={retrieval_alpha:.2f}"
        )
    rerank_df = rerank_df.sort_values(["qid", "score"], ascending=[True, False]).copy()
    rerank_df["rank"] = rerank_df.groupby("qid").cumcount() + 1
    rerank_df = rerank_df.reset_index(drop=True)
    log_info(f"Reranking produced {len(rerank_df)} rows in {format_elapsed(time.perf_counter() - rerank_start)}")
    return rerank_df


def save_metrics_json(metrics_by_name: dict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    output_path.write_text(json.dumps(metrics_by_name, indent=2), encoding="utf-8")


def build_pipeline_stage_list(args: argparse.Namespace) -> list[str]:
    stages = [
        "Initialize runtime",
        "Prepare BM25 documents",
        "Load queries and qrels",
        "Build BM25 index",
        "Run BM25 retrieval",
        "Evaluate BM25",
    ]

    if not args.skip_dense:
        stages.extend(
            [
                "Prepare dense documents",
                "Load dense model",
                "Run dense retrieval",
                "Evaluate dense retrieval",
                "Fuse retrieval results",
                "Evaluate fused retrieval",
            ]
        )

    if not args.skip_rerank:
        stages.extend(
            [
                "Load reranker",
                "Run reranking",
                "Evaluate reranked results",
            ]
        )

    stages.extend(
        [
            "Save metrics",
            "Finalize outputs",
        ]
    )
    return stages


def main() -> None:
    args = parse_args()
    progress = PipelineProgress(build_pipeline_stage_list(args))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir = args.embedding_cache_dir.resolve()
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_cache_dir = args.prepared_cache_dir.resolve()
    prepared_cache_dir.mkdir(parents=True, exist_ok=True)
    log_step(f"Starting local pipeline with output directory {output_dir}")
    log_info(f"Dense embedding cache directory: {embedding_cache_dir}")
    log_info(f"Prepared data cache directory: {prepared_cache_dir}")

    with progress.stage("Initialize runtime", "setting cache directories and search backends"):
        os.environ["IR_DATASETS_HOME"] = str(args.ir_datasets_home.resolve())
        log_info(f"IR_DATASETS_HOME set to {os.environ['IR_DATASETS_HOME']}")
        ensure_pyterrier()

        device, device_label = select_device()
        log_info(f"Default device backend: {device_label} ({device})")
        dense_device = resolve_device(args.dense_device, device)
        rerank_device = resolve_device(args.rerank_device, device)
        log_info(f"Dense device: {dense_device}")
        log_info(f"Reranker device: {rerank_device}")

    with progress.stage("Prepare BM25 documents", "loading or building document text cache"):
        docs_df, raw_docs = get_docs_df(
            args.max_docs,
            cache_dir=prepared_cache_dir,
            rebuild_prepared=args.rebuild_prepared,
        )
        log_info(f"Prepared BM25 documents: {len(docs_df)}")

    with progress.stage("Load queries and qrels", "reading TREC query set"):
        queries_df, qrels = load_queries_and_qrels()
        thresholded_qrels = build_thresholded_qrels(qrels)
        log_info(f"Loaded queries: {len(queries_df)}")
        log_info(f"Loaded qrels: {len(qrels)} queries")

    bm25_index_dir = args.bm25_index_dir.resolve() if args.bm25_index_dir else output_dir / "bm25_index"
    with progress.stage("Build BM25 index", f"index directory: {bm25_index_dir}"):
        index_ref = build_bm25_index(docs_df, bm25_index_dir, rebuild=args.rebuild_index)

    with progress.stage("Run BM25 retrieval", f"top_k={args.top_k}"):
        bm25_results_df = run_bm25(index_ref, queries_df, args.top_k)
    write_run(bm25_results_df, output_dir / "bm25_run.txt", "bm25_pt")

    metrics_by_name: dict[str, dict[str, dict[str, float]]] = {}
    with progress.stage("Evaluate BM25", "computing rel>=1 and rel>=2 metrics"):
        bm25_rel1 = evaluate_run(bm25_results_df, thresholded_qrels["rel>=1"], args.top_k)
        bm25_rel2 = evaluate_run(bm25_results_df, thresholded_qrels["rel>=2"], args.top_k)
    metrics_by_name["bm25"] = {"rel>=1": bm25_rel1, "rel>=2": bm25_rel2}
    print_metrics("BM25 (rel>=1)", bm25_rel1)
    print_metrics("BM25 (rel>=2)", bm25_rel2)

    sample_qid = queries_df.iloc[0]["qid"]
    sample_results = bm25_results_df[bm25_results_df["qid"] == sample_qid].head(5)
    tqdm.write("\n=== Sample BM25 Results ===")
    for row in sample_results.itertuples(index=False):
        raw_doc = raw_docs.get(row.docno, {})
        tqdm.write(f"  {int(row.rank):>2}. {row.docno} | {row.score:.4f} | {raw_doc.get('title', '')}")

    candidate_df = bm25_results_df

    if not args.skip_dense:
        with progress.stage("Prepare dense documents", f"text mode: {args.dense_text_mode}"):
            dense_doc_df = get_dense_docs_df(
                args.max_docs,
                mode=args.dense_text_mode,
                cache_dir=prepared_cache_dir,
                rebuild_prepared=args.rebuild_prepared,
                raw_docs=raw_docs,
            )
            log_info(f"Prepared dense documents: {len(dense_doc_df)}")

        with progress.stage("Load dense model", args.dense_model):
            dense_model = load_dense_model(args.dense_model, device=dense_device)

        with progress.stage("Run dense retrieval", f"top_k={args.top_k}, batch_size={args.dense_batch_size}"):
            dense_results_df = run_dense_retrieval(
                dense_model,
                dense_doc_df,
                queries_df,
                emb_path=embedding_cache_dir / "dense_doc_embeddings.npy",
                docno_path=embedding_cache_dir / "dense_docnos.npy",
                top_k=args.top_k,
                batch_size=args.dense_batch_size,
                reuse_cache=True,
            )
        write_run(dense_results_df, output_dir / "dense_run.txt", "dense_only")

        with progress.stage("Evaluate dense retrieval", "computing dense metrics"):
            dense_rel1 = evaluate_run(dense_results_df, thresholded_qrels["rel>=1"], args.top_k)
            dense_rel2 = evaluate_run(dense_results_df, thresholded_qrels["rel>=2"], args.top_k)
        metrics_by_name["dense"] = {"rel>=1": dense_rel1, "rel>=2": dense_rel2}
        print_metrics("Dense (rel>=1)", dense_rel1)
        print_metrics("Dense (rel>=2)", dense_rel2)

        fusion_detail = "tuning weighted RRF" if args.tune_rrf else f"rrf_k={args.rrf_k}"
        with progress.stage("Fuse retrieval results", fusion_detail):
            if args.tune_rrf:
                fused_results_df, best_rrf_trial, tuning_trials_df = tune_weighted_rrf(
                    bm25_results_df,
                    dense_results_df,
                    thresholded_qrels["rel>=1"],
                    args.top_k,
                )
                tuning_trials_path = output_dir / "rrf_tuning_trials.csv"
                tuning_trials_df.to_csv(tuning_trials_path, index=False)
                log_info(f"Saved RRF tuning trials to {tuning_trials_path}")
            else:
                fused_results_df = fuse_rrf(bm25_results_df, dense_results_df, args.rrf_k, args.top_k)
        write_run(fused_results_df, output_dir / "fused_run.txt", "bm25_dense_rrf")

        with progress.stage("Evaluate fused retrieval", "computing fused metrics"):
            fused_rel1 = evaluate_run(fused_results_df, thresholded_qrels["rel>=1"], args.top_k)
            fused_rel2 = evaluate_run(fused_results_df, thresholded_qrels["rel>=2"], args.top_k)
        metrics_by_name["fused_rrf"] = {"rel>=1": fused_rel1, "rel>=2": fused_rel2}
        print_metrics("Fused RRF (rel>=1)", fused_rel1)
        print_metrics("Fused RRF (rel>=2)", fused_rel2)
        candidate_df = build_union_candidates(
            bm25_results_df,
            dense_results_df,
            fused_results_df,
            args.top_k,
        )

    if not args.skip_rerank:
        safe_reranker_name = safe_cache_name(args.reranker_model)
        rerank_cache_path = output_dir / f"rerank_scores_{safe_reranker_name}_{RERANK_CACHE_VERSION}.parquet"
        with progress.stage("Load reranker", args.reranker_model):
            reranker = load_reranker(args.reranker_model, device=rerank_device)
        with progress.stage("Run reranking", f"batch_size={args.rerank_batch_size}, cache={rerank_cache_path.name}"):
            reranked_df = rerank_candidates(
                reranker,
                candidate_df,
                queries_df,
                docs_df,
                raw_docs,
                rerank_cache_path,
                batch_size=args.rerank_batch_size,
                rerank_alpha=args.rerank_alpha,
            )
            reranked_df = reranked_df[reranked_df["rank"] <= args.top_k].copy()
        write_run(reranked_df, output_dir / "reranked_run.txt", "crossencoder_rerank")

        with progress.stage("Evaluate reranked results", "computing reranked metrics"):
            rerank_rel1 = evaluate_run(reranked_df, thresholded_qrels["rel>=1"], args.top_k)
            rerank_rel2 = evaluate_run(reranked_df, thresholded_qrels["rel>=2"], args.top_k)
        metrics_by_name["reranked"] = {"rel>=1": rerank_rel1, "rel>=2": rerank_rel2}
        print_metrics("Reranked (rel>=1)", rerank_rel1)
        print_metrics("Reranked (rel>=2)", rerank_rel2)

    with progress.stage("Save metrics", "writing metrics.json and summary table"):
        save_metrics_json(metrics_by_name, output_dir / "metrics.json")
        print_metrics_summary(metrics_by_name)

    with progress.stage("Finalize outputs", str(output_dir)):
        log_step(f"Pipeline finished. Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
