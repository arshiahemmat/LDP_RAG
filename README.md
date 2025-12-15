# RAG‑Pipeline 🚀

*A flexible, plug‑and‑play **Retrieval‑Augmented Generation** evaluation harness*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-Add Your Own-lightgrey)
![Status](https://img.shields.io/badge/Build-passing-brightgreen)

</div>

---

## 📚 Table of Contents

1. [Project layout](#-project-layout)
2. [Installation](#-installation)
3. [Environment variables](#-environment-variables)
4. [Quick start](#-quick-start)
5. [CLI reference](#-cli-reference)
6. [Output format](#-output-format)
7. [Extending the pipeline](#-extending-the-pipeline)
8. [Troubleshooting](#-troubleshooting)
9. [License](#-license)

---

## 📁 Project layout

```text
project/
│
├─ models/                   # ➜ each file defines ONE RAG class
│   ├─ normal_rag.py         #   • NormalRAG
│   ├─ privacy_gate_rag.py   #   • PrivacyGateRAG
│   ├─ ldp_rag.py            #   • LDPRAG
│   └─ adaptive_rag.py       #   • AdaptiveRAG
│
├─ utils/
│   ├─ metrics.py            # custom evaluation metrics
│   └─ adaptive_chunking.py  # AdaptiveChunkSplitter (or get_splitter())
│
├─ main.py                   # single CLI entry‑point
└─ README.md                 # you are here
```

```text
datasets/
└─ dataset-ldp/
    └─ Data_LDP/
        ├─ Docs/                    # knowledge base (.docx)
        ├─ Attack Question/         # prompts to ask the RAG
        │   ├─ Customer Service/1.json … 10.json
        │   ├─ E‑commerce/ …
        │   └─ Healthcare/ …
        └─ MetaDatas/               # optional extras
```

---

## ⚙️ Installation

```bash
# create & activate virtual‑env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# core libraries
pip install -U pip wheel
pip install langchain faiss-cpu openai google-generativeai json5

# optional (pretty output inspection)
pip install pandas
```

> **Python 3.9 or newer** is highly recommended (pre‑built FAISS wheels available).

---

## 🔑 Environment variables

| Variable         | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `OPENAI_API_KEY` | API key for GPT‑style back‑ends (e.g. AvalAI proxy) |
| `GOOGLE_API_KEY` | Only required when using Gemini RAG                 |

You can also override via `--api_key`/`--base_url` flags.

---

## 🏃‍♂️ Quick start

```bash
python main.py \
  --rag_type normal \
  --metrics faithfulness answer_relevancy \
  --adaptive_chunking \
  --output_path ./runs/demo.jsonl
```

**What happens?**

1. Builds / loads a FAISS index from `datasets/dataset-ldp/Data_LDP/Docs`.
2. Walks the entire *Attack Question* tree (`*.json` files).
3. For every question it:

   * retrieves `top_k` (default **4**) relevant chunks,
   * generates an answer with the chosen RAG back‑end,
   * evaluates requested metrics, and
   * writes one JSON‑Lines record to `runs/demo.jsonl`.

---

## 💻 CLI reference

| Flag                  | Default                                         | Description                                             |
| --------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| `--rag_type`          | *(required)*                                    | `normal`, `privacy_gate`, `ldp`, `adaptive`             |
| `--docs_dir`          | `datasets/dataset-ldp/Data_LDP/Docs`            | Knowledge‑base folder                                   |
| `--questions_dir`     | `datasets/dataset-ldp/Data_LDP/Attack Question` | Prompts root                                            |
| `--metrics`           | —                                               | Space‑separated list from `utils.metrics`               |
| `--adaptive_chunking` | `false`                                         | Toggle `utils.adaptive_chunking.AdaptiveChunkSplitter`  |
| `--output_path`       | *(required)*                                    | Destination `.jsonl` file                               |
| `--index_name`        | `faiss_index`                                   | Sub‑folder in `RAG_Database/`                           |
| `--top_k`             | `4`                                             | Context chunks to retrieve                              |
| LLM flags             | —                                               | `--llm_name`, `--model_name`, `--api_key`, `--base_url` |

---

## 📄 Output format

Every **line** in the output file is a self‑contained JSON object:

```jsonc
{
  "question": "How would this medical detail impact treatment plans …?",
  "question_type": "Healthcare",
  "label": "Sensitive",
  "source_sentence": "CKD increases the risk of cardiovascular disease, …",
  "answer": "First, clinicians would …",
  "metrics": {
    "faithfulness": 0.93,
    "answer_relevancy": 0.82
  },
  "file": "/…/Attack Question/Healthcare/7.json"
}
```

```python
import pandas as pd
df = pd.read_json("runs/demo.jsonl", lines=True)
```

---


## 🔌 Extending the pipeline

| Task                       | How                                                                                                                        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Add a new RAG back‑end** | 1) Create `models/my_rag.py` exposing one class †  2) Register in `RAG_CLASS_REGISTRY` → `"my_rag": "models.my_rag:MyRAG"` |
| **Add a metric**           | Implement a function in `utils/metrics.py`, then call `--metrics my_metric`.                                               |
| **Custom chunker**         | Expose `AdaptiveChunkSplitter` or `get_splitter()` in `utils/adaptive_chunking.py`; enable with `--adaptive_chunking`.     |

† Class must provide `.load_documents()`, `.save_vector_store()`, `.load_vector_store()`, and `.generate()`.

---


## 🐛 Troubleshooting

| Symptom                      | Remedy                                              |
| ---------------------------- | --------------------------------------------------- |
| `ModuleNotFoundError: faiss` | `pip install faiss-cpu` *(or* `faiss-gpu`*)*        |
| 401 / 403 from LLM           | Check `OPENAI_API_KEY` and `--base_url`             |
| “No documents loaded yet”    | Ensure `.docx` files actually exist in `--docs_dir` |
| Metric `KeyError`            | Verify the metric name passed in `--metrics`        |
