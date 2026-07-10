#!/usr/bin/env python3
"""Offline retrieval eval against the labeled fixture corpus (FakeEmbeddings)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma

from backend.api.deps import clear_vectorstore_cache
from backend.config import Settings
from backend.services.eval_quality import mean, precision_at_k, recall_at_k
from backend.services.faithfulness import is_faithful
from backend.services.ingestion import ingest_bytes
from backend.services.retrieval import retrieve_chunks

EVAL_K = 4
CORPUS_DIR = Path(__file__).parent / "corpus"
QA_PATH = Path(__file__).parent / "qa.json"
EMPTY_REFUSAL = (
    "I don't know. No relevant information found in the indexed documents. "
    "Upload documents or lower MIN_RELEVANCE_SCORE if needed."
)


def _candidate_answer(context: str, must_include: list[str]) -> str:
    """Build a grounded-looking answer from retrieved context for offline checks."""
    for phrase in must_include:
        if phrase and phrase.lower() in context.lower():
            return f"According to the documents, {phrase}."
    snippet = " ".join(context.split())[:180]
    return f"According to the documents, {snippet}"


def _run(write_results: bool = False) -> dict:
    clear_vectorstore_cache()
    cases = json.loads(QA_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        settings = Settings(
            openai_api_key="sk-eval",
            chroma_persist_dir=tmp_path / "chroma",
            upload_dir=tmp_path / "uploads",
            min_relevance_score=0.0,
            retrieval_k=EVAL_K,
            chunk_size=400,
            chunk_overlap=40,
        )
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

        emb = FakeEmbeddings(size=135)
        vs = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=emb,
            persist_directory=str(settings.chroma_persist_dir),
        )

        # Patch both call sites used by ingest + retrieve.
        import backend.services.ingestion as ingestion_mod
        import backend.services.retrieval as retrieval_mod

        ingestion_mod.get_vectorstore = lambda s=None: vs  # type: ignore[assignment]
        retrieval_mod.get_vectorstore = lambda s=None: vs  # type: ignore[assignment]

        for path in sorted(CORPUS_DIR.glob("*.txt")):
            ingest_bytes(path.read_bytes(), path.name, settings)

        per_case = []
        p_scores: list[float] = []
        r_scores: list[float] = []
        faith_flags: list[bool] = []

        # Empty-context refusal must stay faithful.
        empty_ok = is_faithful(EMPTY_REFUSAL, "", context_empty=True)
        faith_flags.append(empty_ok)

        for case in cases:
            pairs, context = retrieve_chunks(case["question"], settings, k=EVAL_K)
            retrieved_sources: list[str] = []
            for doc, _score in pairs:
                src = str(doc.metadata.get("source", "unknown"))
                if src not in retrieved_sources:
                    retrieved_sources.append(src)
            relevant = set(case["relevant_sources"])
            p = precision_at_k(retrieved_sources, relevant, EVAL_K)
            r = recall_at_k(retrieved_sources, relevant, EVAL_K)
            p_scores.append(p)
            r_scores.append(r)

            context_empty = not context.strip()
            answer = (
                EMPTY_REFUSAL
                if context_empty
                else _candidate_answer(context, case.get("must_include") or [])
            )
            faithful = is_faithful(answer, context, context_empty=context_empty)
            negative_ok = True
            if not context_empty:
                negative_ok = not is_faithful(
                    "Zephyr-quark photosynthesis yields violet plasma.",
                    context,
                    context_empty=False,
                )
            faith_flags.append(faithful and negative_ok)
            per_case.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "relevant_sources": sorted(relevant),
                    "retrieved_sources": retrieved_sources,
                    "precision_at_k": round(p, 4),
                    "recall_at_k": round(r, 4),
                    "context_empty": context_empty,
                    "faithful": faithful,
                }
            )

        faith_rate = mean([1.0 if f else 0.0 for f in faith_flags])
        summary = {
            "embedding": "FakeEmbeddings(size=135)",
            "k": EVAL_K,
            "n_questions": len(cases),
            "mean_precision_at_k": round(mean(p_scores), 4),
            "mean_recall_at_k": round(mean(r_scores), 4),
            "faithfulness_checks": len(faith_flags),
            "faithfulness_pass_rate": round(faith_rate, 4),
            "empty_context_refusal_ok": empty_ok,
            "cases": per_case,
            "command": "python evals/run_eval.py",
        }

    clear_vectorstore_cache()
    if write_results:
        out_json = Path(__file__).parent / "latest.json"
        out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Fixture retrieval eval results",
            "",
            "Offline harness using `FakeEmbeddings` (no live OpenAI calls).",
            "",
            f"- Command: `{summary['command']}`",
            f"- k = {summary['k']}",
            f"- Questions = {summary['n_questions']}",
            f"- Mean precision@{summary['k']} = **{summary['mean_precision_at_k']}**",
            f"- Mean recall@{summary['k']} = **{summary['mean_recall_at_k']}**",
            f"- Faithfulness pass rate = **{summary['faithfulness_pass_rate']}** "
            f"({summary['faithfulness_checks']} checks, including empty-context refusal)",
            "",
            "## Per question",
            "",
        ]
        for c in per_case:
            lines.append(
                f"- `{c['id']}`: P@{EVAL_K}={c['precision_at_k']}, "
                f"R@{EVAL_K}={c['recall_at_k']}, faithful={c['faithful']}; "
                f"retrieved={c['retrieved_sources']}"
            )
        lines.append("")
        (Path(__file__).parent / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    summary = _run(write_results=True)
    print(json.dumps({k: summary[k] for k in summary if k != "cases"}, indent=2))
    print(f"Wrote evals/RESULTS.md and evals/latest.json")


if __name__ == "__main__":
    main()
