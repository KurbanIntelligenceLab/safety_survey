# safety_survey

Code behind our bibliometric survey of AI-safety research (2018–2026): how the
corpus was collected, how papers were screened and labelled, how the taxonomy
was derived and validated, and how every number and figure in the paper was
computed.

The repo holds **code only**. The corpus itself
(titles, abstracts, labels, embeddings, cached PDFs) is not in git; see
[Data](#data).

## What the pipeline does

```
 DBLP queries + full proceedings census + citation snowball
        │  01_collect      candidate papers from 10 venues
        ▼
 02_enrich                 abstracts from Semantic Scholar
        ▼
 03_classify               LLM gate: is this a safety paper? + provisional category
        ▼
 04_validate               blind re-labelling by other models, pool audit
        ▼
 05_taxonomy               SPECTER → UMAP → HDBSCAN clustering; final category labels
        ▼
 06_fulltext               PDFs, method/conclusion sections, full-text re-classification
        ▼
 07_open_problems          per-subfield "what is open" census and its validation
        ▼
 08_stats                  the counts, shares and growth rates quoted in the paper
```

The final corpus is 12,987 safety papers in 14 categories.

A note on where the categories come from: the category names began as
hand-written keyword groups (`pipeline/01_collect/scrape_ai_safety_papers.py`,
`taxonomy_expanded.py`) and were used in the gate prompt
(`pipeline/03_classify/classify_llm.py`). The clustering in `05_taxonomy`
then recovered and corrected them, and `apply_cluster_labels.py` assigns each
paper its final label from its cluster.

## Layout

| Folder | Contents |
|---|---|
| `pipeline/` | The corpus pipeline, one sub-folder per stage. Script-by-script index in [`pipeline/README.md`](pipeline/README.md). |
| `requirements.txt` | Python packages the scripts import. |
| `.env.example` | The environment variables the scripts read. |

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in the keys you need
set -a; . ./.env; set +a
```

Collection, merging and scoring scripts need only `requests` or the standard
library. The clustering and embedding scripts need the heavier packages
(`torch`, `sentence-transformers`, `umap-learn`, `hdbscan`).

## Running

Run everything **from the repo root**; paths are relative to it.

```bash
python pipeline/01_collect/collect_census.py collect
python pipeline/05_taxonomy/bertopic_compare.py
```

Scripts in one stage folder can import helpers from another (for example the
canonical prompt in `03_classify/classify_llm.py`); each one adds the stage
folders to `sys.path` itself, so no `PYTHONPATH` is needed.

The `*.sbatch` files are SLURM jobs for the steps that are too heavy for a
login node. Submit them from the repo root with your own account:
`sbatch -A <account> pipeline/07_open_problems/embed_op_sections.sbatch`.

The `*_workflow.js` files are Claude Code Workflow scripts. They fan one
agent out per prompt batch for the steps where an LLM does the labelling
(the safety gate, full-text re-classification, the open-problem census).
The matching `build_*` script writes the batches and the `merge_*` script
collects the results.

## Data

The pipeline reads and writes a `paper_survey/` folder in the repo root
(git-ignored). The main files, in the order they are produced:

| File | Produced by |
|---|---|
| `papers_all.json` | `01_collect/scrape_ai_safety_papers.py` |
| `candidates_expanded.json`, `snowball_candidates.json`, `_census/` | the other `01_collect` scripts |
| `papers_enriched.json`, `snowball_enriched.json` | `02_enrich` |
| `papers_llm.json`, `papers_llm_cleaned.json` | `03_classify` |
| `_val_*`, `_pool_audit/` | `04_validate` |
| `_census/corpus_clusterlabeled.json` (the final labelled corpus) | `05_taxonomy/apply_cluster_labels.py` |
| `fulltext_index.json`, `_fulltext12987/` | `06_fulltext`, `07_open_problems` |
| `paper_stats.json`, `paper_numbers_*.json` | `08_stats` |

The paper listing and audit data are available from the authors on request.
