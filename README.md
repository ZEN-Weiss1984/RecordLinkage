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

📁 Project Structure Description

This repository is organized into multiple components, each responsible for a specific part of the deduplication and linkage pipeline:
	•	dataset/
This folder contains all the input data used in the project.
	•	primary.csv holds the main records that require deduplication.
	•	alternate.csv contains alternate names or variants that need to be linked back to the primary entities.
	•	test_01.xlsx is a general test set for linkage evaluation.
	•	test_02.xlsx contains eight separate sheets, each simulating a specific type of name corruption or variation (e.g., swapped characters, missing spaces, character omissions). These are used to assess the robustness of different linkage methods.
	•	output/
This directory stores the results of deduplication and linkage tasks.
It includes CSV files for matched pairs, similarity scores, and prediction outputs for each test case.
For example, results like test01_links.csv and test02_sheet4_links.csv are generated here.
These files are suitable for further analysis or submission.
	•	picture/
This folder contains visual outputs such as precision-recall curves, confusion matrices, and network graph plots.
These visualizations help compare the performance of different models and methods on each test scenario.
Each image corresponds to one test set or method configuration.
	•	duplicate_recordlinkage.py
A script that performs deduplication on the primary dataset using classic record linkage techniques.
It uses the recordlinkage library with indexing, blocking, and comparison strategies such as Levenshtein and rule-based filters.
	•	duplicate_transformer.py
A deduplication script based on Transformer embeddings.
It uses the LinkTransformer framework to encode names, apply vector similarity, and identify duplicates with semantic similarity.
	•	link.py
This script performs Transformer-based linking between datasets.
It uses a two-stage pipeline:
	1.	A Siamese model generates compact embeddings for fast retrieval using FAISS (as a coarse filter).
	2.	A Transformer (e.g., MiniLM) computes fine-grained semantic similarity.
This script is used to link test_01 and test02’s sheet7 and sheet8 to the primary dataset.
	•	match.py
A traditional linkage method using multiple string similarity algorithms: Levenshtein distance, Jaccard similarity, and Python’s SequenceMatcher.
This script is used specifically for test_02 sheets 1 and 3, which involve character permutations or token-level changes.
	•	match_02.py
This script uses alternative traditional methods including character frequency matching and n-gram similarity, which are effective for handling heavily distorted or noisy inputs (e.g., OCR errors).
It is applied to test_02 sheets 2, 4, 5, and 6.

Each script is modular and can be run independently by specifying the corresponding input data and configuration parameters. Results from all scripts are saved into output/, and optional visualizations are saved to picture/.

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

## 📊 Output Artefacts
	•	CSV files in output/ contain matched IDs & similarity scores.
	•	HTML reports summarise precision / recall and error examples.
	•	PNG/SVG plots live in picture/ for quick drop-into slides.

## 📖 Reference

Arora, A., & Dell, M. (2024). LinkTransformer: A Unified Package for Record Linkage with Transformer Language Models. Proceedings of ACL 2024.
https://linktransformer.github.io/  ￼

If you use this repository or LinkTransformer in academic work, please cite the paper above.

⸻

## 🤝 Contributing

Issues and pull requests are warmly welcomed! For substantial changes, please open a discussion first so we can align on design.
