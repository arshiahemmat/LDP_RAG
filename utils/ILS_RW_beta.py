# -*- coding: utf-8 -*-
import re
from typing import Dict, Set, List, Tuple, Optional
import json
import os
import glob
import tempfile

# -------------------------
# Config
# -------------------------
BASE_PATH = "/home/martin/Projects/LDP_RAG_DataSet/new ragas output/PrivacyAwareGate"
# ترتیب پوشه‌ها دقیقاً طبق خواسته‌ی شما
METHOD_FOLDERS = [
    "openai", "llama", "qwen", "deepseek"
]

# -------------------------
# Leakage core (بدون تغییرات اساسی)
# -------------------------
ATOM_WEIGHTS = {
    "email": 5.0, "phone": 5.0, "id": 5.0, "address": 4.0,
    "name": 3.0, "date": 2.0, "default": 1.0,
}

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+", re.I)
PHONE_RE = re.compile(r"\+?\d[\d\-\s()]{6,}\d")
DATE_RE  = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
ID_RE    = re.compile(r"\b\d{4,}\b")

def _detect_atom_type(token: str) -> str:
    if EMAIL_RE.search(token): return "email"
    if PHONE_RE.search(token): return "phone"
    if DATE_RE.search(token):  return "date"
    if ID_RE.search(token):    return "id"
    return "default"

def _ensure_nlp():
    try:
        import spacy  # lazy import
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise RuntimeError(f"Failed to load spacy model 'en_core_web_sm'. Please install it with: python -m spacy download en_core_web_sm") from e

_STOP = {
    "the","a","an","and","or","of","to","in","for","on","with","by","at","is","are",
}

def _extract_atoms(txt: str, nlp, min_len: int = 2) -> Set[str]:
    ents = {e.text.strip().lower() for e in nlp(txt).ents if e.text.strip()}
    toks = {
        t.lower() for t in re.findall(r"[A-Za-z0-9@._+-]+", txt)
        if len(t) >= min_len and t.lower() not in _STOP
    }
    return ents | toks

def cold_stats_leakage(reference_answer: str, generated_answer: str, nlp=None) -> Dict[str, float]:
    if not reference_answer or not generated_answer:
        return {"leak_rate": 0.0, "ils": 1.0}
    
    if nlp is None:
        nlp = _ensure_nlp()
    priv_atoms = _extract_atoms(reference_answer, nlp)
    pub_atoms  = _extract_atoms(generated_answer, nlp)

    weighted_total = 0.0
    weighted_overlap = 0.0
    for atom in priv_atoms:
        w = ATOM_WEIGHTS.get(_detect_atom_type(atom), ATOM_WEIGHTS["default"])
        weighted_total += w
        if atom in pub_atoms:
            weighted_overlap += w

    leak_rate = weighted_overlap / max(1e-9, weighted_total)
    return {"leak_rate": float(leak_rate), "ils": float(1.0 - leak_rate)}

def _extract_answers(data: Dict) -> Dict[str, str]:
    reference_answer = None
    generated_answer = None
    if isinstance(data.get("ragas_results"), dict):
        rr = data["ragas_results"]
        reference_answer = rr.get("reference")
        generated_answer = rr.get("response")
    if reference_answer is None:
        reference_answer = data.get("reference")
    if generated_answer is None:
        generated_answer = data.get("generated_answer")
    return {"reference": reference_answer, "generated": generated_answer}

# -------------------------
# Utilities
# -------------------------
def jsonl_files(folder_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder_path, "*.jsonl")))

def safe_json_loads(line: str, line_num: int = 0, file_path: str = "") -> Optional[Dict]:
    """Safely parse JSON line with error handling."""
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"JSON decode error in {file_path} at line {line_num}: {e}")
        print(f"Problematic line: {line[:100]}{'...' if len(line) > 100 else ''}")
        return None
    except Exception as e:
        print(f"Unexpected error parsing JSON in {file_path} at line {line_num}: {e}")
        return None

def read_mean_leak_rate(folder_path: str) -> float:
    """میانگین leak_rate یا leakage_score فعلیِ فایل‌های پوشه را می‌خواند (بدون تغییر فایل)."""
    vals: List[float] = []
    for p in jsonl_files(folder_path):
        with open(p, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: 
                    continue
                
                d = safe_json_loads(line, line_num, p)
                if d is None:
                    continue
                    
                v = None
                lr = d.get("leakage_rate")
                if isinstance(lr, dict) and "leak_rate" in lr:
                    v = lr["leak_rate"]
                if v is None:
                    ls = d.get("leakage_score")
                    if isinstance(ls, (int, float)):
                        v = float(ls)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
    return (sum(vals) / len(vals)) if vals else 0.0

def update_folder_leakage(folder_path: str) -> None:
    """همان نسخه‌ی قبلی: محاسبه و ست‌کردن leakage_rate برای هر رکورد."""
    nlp = _ensure_nlp()
    pattern = os.path.join(folder_path, "*.jsonl")
    for jsonl_path in glob.glob(pattern):
        dir_name = os.path.dirname(jsonl_path)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=dir_name, encoding='utf-8') as tmpf:
            tmp_path = tmpf.name
            with open(jsonl_path, "r", encoding='utf-8') as src:
                for line_num, line in enumerate(src, 1):
                    if not line.strip():
                        continue
                    
                    data = safe_json_loads(line, line_num, jsonl_path)
                    if data is None:
                        # Skip invalid JSON lines but preserve them in output
                        tmpf.write(line)
                        tmpf.write("\n" if not line.endswith("\n") else "")
                        continue
                    
                    ans = _extract_answers(data)
                    ref = ans["reference"]
                    gen = ans["generated"]
                    if not ref or not gen:
                        tmpf.write(line)
                        tmpf.write("\n" if not line.endswith("\n") else "")
                        continue
                    
                    try:
                        data["leakage_rate"] = cold_stats_leakage(ref, gen, nlp)
                        tmpf.write(json.dumps(data, ensure_ascii=False))
                        tmpf.write("\n")
                    except Exception as e:
                        print(f"Error processing leakage for line {line_num} in {jsonl_path}: {e}")
                        # Write original line if processing fails
                        tmpf.write(line)
                        tmpf.write("\n" if not line.endswith("\n") else "")
        os.replace(tmp_path, jsonl_path)

# -------------------------
# Orchestration: prev/new means + plot
# -------------------------
def process_selected_folders(base_path: str, folders: List[str]) -> Tuple[List[str], List[float], List[float]]:
    """برای هر پوشه: میانگین قبلی را می‌گیرد، آپدیت می‌کند، میانگین جدید را می‌گیرد."""
    names: List[str] = []
    prev_means: List[float] = []
    new_means: List[float]  = []

    # حذف تکراری‌ها با حفظ ترتیب
    seen = set()
    ordered_folders = []
    for x in folders:
        if x not in seen:
            seen.add(x)
            ordered_folders.append(x)

    for folder in ordered_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            print(f"[warn] folder not found: {folder_path}")
            continue

        print(f"Processing: {folder}")
        prev_mean = read_mean_leak_rate(folder_path)
        try:
            update_folder_leakage(folder_path)
        except FileNotFoundError as e:
            print(f"  - file not found error in {folder}: {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"  - JSON decode error in {folder}: {e}")
            continue
        except Exception as e:
            print(f"  - unexpected error updating leakage in {folder}: {e}")
            continue
        new_mean = read_mean_leak_rate(folder_path)

        names.append(folder)
        prev_means.append(prev_mean)
        new_means.append(new_mean)

        # ذخیرهٔ نتایج هر پوشه در یک فایل JSONL خلاصه
        out_line = {
            "Folder": folder,
            "prev_leakage_mean": prev_mean,
            "new_leakage_mean": new_mean,
        }
        with open(os.path.join(folder_path, f"{folder}_prev_new_leakage.jsonl"), "w", encoding='utf-8') as out:
            out.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    return names, prev_means, new_means

def plot_prev_vs_new(names: List[str], prev_means: List[float], new_means: List[float], out_path: str):
    """نمودار میله‌ای دوطرفه: قدیمی زیر صفر (منفی)، جدید بالای صفر (مثبت)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(names))
    width = 0.35

    # قدیمی‌ها را منفی می‌کنیم تا پایین خط بیفتند
    prev_vals = -np.array(prev_means, dtype=float)
    new_vals  =  np.array(new_means, dtype=float)

    plt.figure(figsize=(12, 6))
    bars_prev = plt.bar(x - width/2, prev_vals, width, label="Previous mean leakage", hatch="//", edgecolor="black")
    bars_new  = plt.bar(x + width/2, new_vals,  width, label="New mean leakage")

    # خط صفر
    plt.axhline(0, color="black", linewidth=1)

    # برچسب‌ها و ظاهر مشابه نمونه
    plt.xticks(x, names, rotation=30)
    plt.ylabel("Leakage Score (mean)")
    plt.title("Mean leakage per method: previous (below) vs new (above)")
    plt.legend(loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # اعداد روی میله‌ها
    def annotate(bars, values):
        for b, v in zip(bars, values):
            y = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, y + (0.015 if v >= 0 else -0.015),
                     f"{abs(v):.3f}", ha="center", va="bottom" if v>=0 else "top", fontsize=9)
    annotate(bars_prev, prev_vals)
    annotate(bars_new, new_vals)

    # حاشیهٔ عمودی معقول
    ymax = max(new_vals.max() if len(new_vals) else 0, abs(prev_vals.min()) if len(prev_vals) else 0)
    plt.ylim(-(ymax*1.25), ymax*1.25 if ymax>0 else 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    names, prev_means, new_means = process_selected_folders(BASE_PATH, METHOD_FOLDERS)
    # ذخیرهٔ خلاصه‌ی کلی
    summary_path = os.path.join(BASE_PATH, "leakage_prev_vs_new_summary.jsonl")
    with open(summary_path, "w", encoding='utf-8') as out:
        for n, p, q in zip(names, prev_means, new_means):
            out.write(json.dumps({"Folder": n, "prev_mean": p, "new_mean": q}, ensure_ascii=False) + "\n")
    # نمودار
    fig_path = os.path.join(BASE_PATH, "leakage_prev_vs_new_barchart.png")
    plot_prev_vs_new(names, prev_means, new_means, fig_path)
    print(f"Chart written to: {fig_path}")
    print(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    main()
