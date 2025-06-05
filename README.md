# RecordLinkage-Demo 🪄 | Hybrid Deduplication & Linking Toolkit

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/) 
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

## 📁 Project Structure

This repository is structured into clear modules for data input, model execution, and output analysis:

### 🔸 `dataset/`

Contains all the input data used in this project:
- `primary.csv` — Main record table requiring **deduplication**.
- `alternate.csv` — Alternate name variants to be **linked** back to the primary table.
- `test_01.xlsx` — General-purpose test set for evaluating linkage accuracy.
- `test_02.xlsx` — A specialized test set containing **8 sheets**, each simulating a different name distortion type (e.g., missing characters, swapped tokens, spacing noise, or OCR-like corruptions).

### 🔸 `output/`

Holds all **deduplication and linkage result files**, including:
- CSVs of predicted matches with scores, such as `test01_links.csv` or `test02_sheet4_links.csv`.
- These results are used to compute precision, recall, and false positive/negative rates.
- HTML or JSON logs can also be saved here depending on evaluation script extensions.
- deduplicated_primary.csv is the deduplicated primary file
- Among them, the main file is the CSV file after deduplication and concatenation of the primary and alternate links.
  

### 🔸 `picture/`

Contains **visualizations** of model results:
- Precision-Recall curves
- Confusion matrices
- Clustering graphs or similarity distributions
These images are generated to support reporting, debugging, or academic paper figures.

### 🔸 `duplicate_recordlinkage.py`

Performs **record deduplication** on `primary.csv` using classic methods from the `recordlinkage` library:
- Includes blocking, indexing, and comparisons like Levenshtein distance and rule-based similarity.

### 🔸 `duplicate_transformer.py`

Uses **Transformer-based embeddings** to deduplicate the primary table:
- Based on [LinkTransformer](https://linktransformer.github.io/), leveraging pretrained models (e.g., MiniLM).
- Employs vector similarity (cosine, FAISS) to detect semantic duplicates.

### 🔸 `link.py`

A Transformer-based **entity linking pipeline**:
- Used for `test_01` and `test_02` sheets **7 and 8**.
- First applies a **Siamese model** for approximate filtering (FAISS), then uses a **Transformer** for fine-grained similarity scoring.
- Designed for efficiency and scalability on large datasets.

### 🔸 `match.py`

A traditional string similarity linkage script:
- Applied to `test_02` sheets **1 and 3**.
- Implements algorithms such as **Levenshtein**, **Jaccard**, and **SequenceMatcher** to compare test and primary names.

### 🔸 `match_02.py`

Alternative rule-based linkage for challenging noisy variants:
- Used for `test_02` sheets **2, 4, 5, and 6**.
- Applies **character frequency** analysis and **n-gram overlap** techniques, which perform better on corruptions like spacing loss or OCR noise.


Each script can be run independently with configurable parameters. All results will be saved to the `output/` folder, and optional figures will be saved in the `picture/` folder.

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

```

## 📖 Reference

If you use this project or any of its components in academic work, please consider citing the following paper:

> Arora, A., & Dell, M. (2024).  
> **LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models**.  
> *Proceedings of ACL 2024*.  
> [https://linktransformer.github.io/](https://linktransformer.github.io/)

BibTeX citation:
```bibtex
@inproceedings{arora2024linktransformer,
  title     = {LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models},
  author    = {Arora, A. and Dell, M.},
  booktitle = {Proceedings of ACL 2024},
  year      = {2024},
  url       = {https://linktransformer.github.io/}
}
```

## 🤝 Contributing

Issues and pull requests are warmly welcomed!  
For substantial changes, please open a discussion first so we can align on design and expectations.

If you have any questions, feedback, or collaboration ideas, feel free to contact us:

- 📧 Email: [2022141520147@stu.scu.edu.cn](mailto:2022141520147@stu.scu.edu.cn)
