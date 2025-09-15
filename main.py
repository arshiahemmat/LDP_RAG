"""
main.py
--------
Flexible runner for multiple RAG variants with optional adaptive chunking
and pluggable evaluation metrics.

Example
-------
python main.py \
    --rag_type normal \
    --metrics faithfulness answer_relevancy \
    --adaptive_chunking \
    --output_path ./runs/run_01.jsonl
"""
import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import json5  


# ---------------------------------------------------------------------------
# 0.  Registry of available Models
# ---------------------------------------------------------------------------

model_aliases = {
    'gemini': 'gemini-2.0-flash-lite',
    'openai': '4o-mini',
    'deepseek': 'deepseek-v3.1',
    'qwen': 'qwen3-235b-a22b-instruct-2507'
}


# ---------------------------------------------------------------------------
# 0.  Registry of available RAG classes 
# ---------------------------------------------------------------------------
RAG_CLASS_REGISTRY: Dict[str, str] = {
    "normal":       "models.normal_rag:NormalRAG",
    "privacy_gate": "models.privacy_gate_rag:PrivacyGateRAG",
    "ldp":          "models.ldp_rag:LDPRAG",
    "adaptive":     "models.adaptive_rag:AdaptiveRAG",
}


# ---------------------------------------------------------------------------
# 1.  Directory defaults that match your screenshot
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = "/kaggle/input/dataset-ldp/Data_LDP"
DEFAULT_DOCS_DIR  = os.path.join(DEFAULT_DATA_ROOT, "Docs")
DEFAULT_Q_DIR     = os.path.join(DEFAULT_DATA_ROOT, "Attack Question")


# ---------------------------------------------------------------------------
# 2.  Helper: dynamic import "package.module:ClassName"
# ---------------------------------------------------------------------------
def dynamic_import(path: str):
    module_path, _, cls_name = path.partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


# ---------------------------------------------------------------------------
# 3.  CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chosen RAG model with optional metrics.")
    parser.add_argument("--rag_type",
                        required=True,
                        choices=RAG_CLASS_REGISTRY.keys(),
                        help="Which RAG implementation to use.")
    parser.add_argument("--docs_dir",
                        default=DEFAULT_DOCS_DIR,
                        help="Folder with .docx knowledge‑base documents.")
    parser.add_argument("--questions_dir",
                        default=DEFAULT_Q_DIR,
                        help="Folder tree with question *.json files.")
    parser.add_argument("--metrics", nargs="*", default=[],
                        help="Metric function names defined in utils.metrics (space‑separated).")
    parser.add_argument("--adaptive_chunking", action="store_true",
                        help="Swap default splitter for utils.adaptive_chunking.AdaptiveChunkSplitter")
    parser.add_argument("--output_path", required=True,
                        help="Path for JSON‑Lines results file.")
    parser.add_argument("--index_name", default="faiss_index",
                        help="FAISS index filename (under RAG_Database/).")
    parser.add_argument("--top_k", type=int, default=4,
                        help="How many chunks to retrieve per question.")
    # LLM / API
    parser.add_argument("--llm_name", default="openai", choices=["openai", "gemini"])
    parser.add_argument("--model_name", default="gpt-4o-mini",
                        help="Chat model to use.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "API_KEY"),
                        help="LLM provider API key (or set OPENAI_API_KEY env var).")
    parser.add_argument("--base_url", default="https://api.avalai.ir/v1",
                        help="Base URL for an OpenAI‑compatible endpoint.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 4.  Build or load RAG
# ---------------------------------------------------------------------------
def build_or_load_rag(
    rag_cls,
    docs_dir: str,
    index_name: str,
    args: argparse.Namespace,
    use_adaptive_chunking: bool,
):
    rag = rag_cls(
        llm_name=args.llm_name,
        api_key=args.api_key,
        model_name=args.model_name,
        base_url=args.base_url,
    )

    # Optional adaptive chunk splitter swap‑in
    if use_adaptive_chunking:
        ac_mod = importlib.import_module("utils.adaptive_chunking")
        splitter = getattr(ac_mod, "AdaptiveChunkSplitter", None) \
                   or getattr(ac_mod, "get_splitter", None)
        if splitter is None:
            raise ImportError("utils.adaptive_chunking must expose AdaptiveChunkSplitter or get_splitter()")
        rag.text_splitter = splitter() if callable(splitter) else splitter
        print("• Using AdaptiveChunkSplitter")

    # Build or restore vector store
    idx_path = Path(rag.vector_store_path) / index_name
    if idx_path.exists():
        rag.load_vector_store(index_name=index_name)
    else:
        rag.load_documents(docs_dir)
        rag.save_vector_store(index_name=index_name)
    return rag


# ---------------------------------------------------------------------------
# 5.  Load metric callables
# ---------------------------------------------------------------------------
def load_metrics(metric_names: List[str]):
    if not metric_names:
        return {}
    m_mod = importlib.import_module("utils.metrics")
    metric_fns = {}
    for name in metric_names:
        if not hasattr(m_mod, name):
            raise ValueError(f"Metric '{name}' not found in utils.metrics")
        metric_fns[name] = getattr(m_mod, name)
    print(f"• Loaded metrics: {', '.join(metric_fns)}")
    return metric_fns


# ---------------------------------------------------------------------------
# 6.  Walk through the Attack Question tree
# ---------------------------------------------------------------------------
def iterate_question_files(questions_dir: str):
    """
    Recursively yield rows from every JSON file:
        question, label, source_sentence, type (= first sub‑folder), _file
    """
    q_root = Path(questions_dir)
    for root, _, files in os.walk(q_root):
        json_files = [f for f in files if f.endswith(".json")]
        if not json_files:
            continue

        # First folder under Attack Question becomes the 'type'
        rel = Path(root).relative_to(q_root)
        q_type = rel.parts[0] if rel.parts else "unknown"

        for jf in json_files:
            fp = Path(root) / jf
            with open(fp, "r") as f:
                data = json5.load(f)

            for row in data:
                row["_file"] = str(fp)
                # if JSON already has 'type' keep it, else inject derived one
                row.setdefault("type", q_type)
                yield row


# ---------------------------------------------------------------------------
# 7.  Main workflow
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Import chosen RAG class
    rag_cls_path = RAG_CLASS_REGISTRY[args.rag_type]
    rag_cls = dynamic_import(rag_cls_path)
    print(f"• Using RAG type: {args.rag_type}  ({rag_cls_path})")

    # Build / load RAG (handles adaptive chunking)
    rag = build_or_load_rag(
        rag_cls,
        docs_dir=args.docs_dir,
        index_name=args.index_name,
        args=args,
        use_adaptive_chunking=args.adaptive_chunking,
    )

    # Metric functions
    metric_fns = load_metrics(args.metrics)

    # Process questions
    results: List[Dict[str, Any]] = []
    for qrow in iterate_question_files(args.questions_dir):
        q_text = qrow["question"]
        q_type = qrow.get("type", "unknown")
        reference = qrow.get("reference")  # optional

        answer, ctx = rag.generate(q_text, top_k=args.top_k, include_context=True)

        metric_scores = {}
        for name, fn in metric_fns.items():
            try:
                metric_scores[name] = fn(
                    predicted=answer,
                    reference=reference,
                    context=ctx,
                )
            except Exception as e:
                metric_scores[name] = f"ERROR: {e}"

        results.append(
            {
                "question":        q_text,
                "question_type":   q_type,
                "label":           qrow.get("label"),
                "source_sentence": qrow.get("source_sentence"),
                "answer":          answer,
                "metrics":         metric_scores,
                "file":            qrow["_file"],
            }
        )
        print(f"✓ {q_type:15s} | {q_text[:60]}…")

    # Save
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✔  Saved {len(results)} records → {out_path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
