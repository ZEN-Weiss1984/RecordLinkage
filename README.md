# RecordLinkage-Demo 🪄 | Hybrid Deduplication & Linking Toolkit

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
[![PyPI](https://img.shields.io/pypi/v/recordlinkage?label=recordlinkage)](https://pypi.org/project/recordlinkage/) 
[![LinkTransformer Docs](https://img.shields.io/badge/LinkTransformer-docs-orange)](https://linktransformer.github.io/) 
[![Build](https://img.shields.io/github/actions/workflow/status/<YOUR-ORG>/<YOUR-REPO>/ci.yml?label=CI)](https://github.com/<YOUR-ORG>/<YOUR-REPO>/actions)

A **batteries-included demo repo** that compares traditional string-similarity techniques with modern Transformer-based models for **record deduplication** and **entity linkage**.  
It showcases how to:

* **Deduplicate** a *primary* table via both classic algorithms (Levenshtein, Jaccard, Sequence-Matcher) and Transformer embeddings.  
* **Link** an *alternate* table to the primary table, picking the best algorithm family **per distortion type** (character swaps, missing spaces, OCR noise, etc.).  
* Evaluate precision / recall on two benchmark sets (**test01** and **test02** with 8 variant sheets).  
* Use **FAISS** for ANN search and **LinkTransformer** for plug-and-play Transformer models.

---

## 📂 Repository Layout

| File | Purpose | Techniques Inside |
|------|---------|-------------------|
| `duplicate_recordlinkage.py` | Deduplicate the primary dataset using the **recordlinkage** toolkit. | Levenshtein & Rule-based indexing.  [oai_citation:0‡recordlinkage.readthedocs.io](https://recordlinkage.readthedocs.io/?utm_source=chatgpt.com) [oai_citation:1‡pypi.org](https://pypi.org/project/recordlinkage/?utm_source=chatgpt.com) |
| `duplicate_transformer.py`   | Deduplicate with Transformer embeddings. | LinkTransformer + FAISS.  [oai_citation:2‡linktransformer.github.io](https://linktransformer.github.io/?utm_source=chatgpt.com) [oai_citation:3‡github.com](https://github.com/dell-research-harvard/linktransformer?utm_source=chatgpt.com) [oai_citation:4‡pypi.org](https://pypi.org/project/faiss-cpu/?utm_source=chatgpt.com) |
| `link.py`                    | Link alternate → primary on **test01** & *test02 sheets 7/8*. | MiniLM/Siamese coarse-to-fine pipeline. |
| `match.py`                   | Link on *test02 sheets 1 & 3*. | Levenshtein 🔀 Jaccard 🔀 `difflib.SequenceMatcher`.  [oai_citation:5‡github.com](https://github.com/rapidfuzz/Levenshtein?utm_source=chatgpt.com) [oai_citation:6‡scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html?utm_source=chatgpt.com) [oai_citation:7‡docs.python.org](https://docs.python.org/3/library/difflib.html?utm_source=chatgpt.com) [oai_citation:8‡stackoverflow.com](https://stackoverflow.com/questions/4802137/how-to-use-sequencematcher-to-find-similarity-between-two-strings?utm_source=chatgpt.com) |
| `match_02.py`                | Link on *test02 sheets 2/4/5/6*. | Character-Frequency & **N-gram** similarity.  [oai_citation:9‡pypi.org](https://pypi.org/project/ngram/?utm_source=chatgpt.com) [oai_citation:10‡analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2021/09/what-are-n-grams-and-how-to-implement-them-in-python/?utm_source=chatgpt.com) |

---

## 🚀 Quick Start

```bash
# 1.  Clone
git clone https://github.com/<YOUR-ORG>/<YOUR-REPO>.git
cd <YOUR-REPO>

# 2.  Install dependencies
pip install -r requirements.txt   # Python 3.8+

# 3.  Run traditional deduplication
python duplicate_recordlinkage.py --in data/primary.csv --out results/dup_rl.csv

# 4.  Run Transformer deduplication
python duplicate_transformer.py --in data/primary.csv --model all-MiniLM-L6-v2

# 5.  Link alternate records (sheet 2 example)
python match_02.py --sheet 2 --in data/test02.xlsx --out results/sheet2_links.csv
