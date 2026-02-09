# HRDE_RAG.ipynb — Hybrid RAG + Local LLM (HRDE-style Rumor Verification)

This repository contains a **single Colab-oriented notebook**: `HRDE_RAG.ipynb`.

The notebook implements an **HRDE-style Retrieval-Augmented Generation (RAG)** pipeline for **Chinese health rumor verification**:

- **Hybrid retrieval:** FAISS (semantic embeddings) + BM25 (keyword)
- **Optional HyDE reranking:** generate a pseudo-document for better retrieval
- **Local generation:** run a Hugging Face causal LM locally (default in notebook: Qwen3 4B Instruct)
- **Corpus builder:** stream **Hush-cd/HealthRCN** from Hugging Face datasets-server and write `data/reference_docs.jsonl`
- **Evaluation:** accuracy/F1 + optional RRR scoring (Relevance, Reliability, Richness)

> ⚠️ Disclaimer: This is a research/demo notebook and **not medical advice**.

---

## What’s inside the notebook (same structure as `HRDE_RAG.ipynb`)

The notebook is organized into these major sections:

1. **Mount Drive + point to your ZIP**  
2. **Install demo requirements** (from your extracted folder / `requirements.txt`)
3. **Download Model** (Hugging Face `snapshot_download`)
4. **Generate reference docs** from **HealthRCN** → `data/reference_docs.jsonl`
5. **Pipeline code** (imports → chunking/tokenization → indexing → retrieval → LLM → HRDE pipeline)
6. **Build the retrieval index** (FAISS + BM25)
7. **Live demo** (type a claim, get verdict)
8. **Evaluation** (build eval set, run loop, compute metrics, save `eval/summary.json`)

---

## Installation (dependencies)

### Option A — Google Colab (recommended)
Open `HRDE_RAG.ipynb` in Colab and run the **Install requirements** cell.

Typical dependencies used in the notebook include:
- `torch`, `transformers`, `accelerate`
- `sentence-transformers`
- `faiss` (or `faiss-cpu`)
- `rank-bm25`
- `jieba`
- `numpy`, `tqdm`, `requests`
- `scikit-learn`, `matplotlib`

### Option B — Local environment
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

> Note: FAISS installation can be OS-dependent. If `faiss` fails, try `faiss-cpu` or a conda-based install.

---

## How to run (experimentation)

There is **no separate training script** in this repo. Everything runs inside the notebook.

1. Open **`HRDE_RAG.ipynb`**
2. Run the sections **in order**:
   - Download model
   - Build reference docs
   - Build index
   - Run live demo
   - Run evaluation

The notebook writes outputs into these folders (created automatically):
- `data/` — reference docs + evaluation samples
- `index/` — FAISS + BM25 artifacts
- `eval/` — per-sample outputs, plots, and `summary.json`

---

## Example: run the model on a sample input file

The notebook includes a **live demo loop**, but you can also test multiple claims by putting one claim per line in a text file, e.g.:

**`input_claims.txt`**
```
维生素C可以治愈感冒。
抗生素对细菌感染有效，但对病毒无效。
```

Then in a notebook cell (after the index is built), run:

```python
# Example batch inference from a plain text file (one claim per line)
input_path = "input_claims.txt"

with open(input_path, "r", encoding="utf-8") as f:
    claims = [line.strip() for line in f if line.strip()]

for claim in claims:
    out = detect_cached(
        claim=claim,
        top_n_faiss=12,
        top_n_bm25=12,
        final_k=6,
        use_hyde=True,
        sim_threshold=0.3,
    )
    print("CLAIM:", claim)
    print(out["raw_answer"])
    print("-" * 60)
```

> `detect_cached(...)` is defined inside the notebook (Section **4.7**).

---

## Evaluation

Run the evaluation section in the notebook:

- It generates `data/eval_samples.jsonl`
- Runs the pipeline for each sample
- Saves:
  - `eval/per_sample.jsonl`
  - `eval/summary.json`  ✅ (this file contains the final metrics)

### Metrics produced
- `valid_rate` (how often the output label is parseable)
- `accuracy`, `f1` (computed on valid samples)
- Optional RRR judge metrics:
  - `relevance_avg`, `reliability_avg`, `richness_avg`
  - `rrr_n_scored`

---

## Troubleshooting

**RRR averages are `null`**  
RRR scoring requires the judge to output strict JSON. If parsing fails, averages may remain `null`.

**Colab Drive storage not changing**  
If the model downloads into `/content/...`, it is **temporary**. Download into `/content/drive/...` to persist.

**OOM / slow generation**  
Reduce `FINAL_K`, `TOP_N_FAISS`, `TOP_N_BM25`, disable HyDE, or use a smaller LLM.

---

## Citation

If you use this notebook in academic work, please cite the HRDE paper and the HealthRCN dataset (see the paper’s bibliography / dataset card).

---

## License

Add a `LICENSE` file (MIT / Apache-2.0 / etc.) to define usage terms.
