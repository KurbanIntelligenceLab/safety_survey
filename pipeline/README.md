# Pipeline scripts

Run from the repo root. Data goes to `paper_survey/` (git-ignored).

## 01_collect — find candidate papers

| Script | What it does |
|---|---|
| `scrape_ai_safety_papers.py` | Original collection: 72 keyword queries against DBLP, dedup, keyword taxonomy. Also holds the shared DBLP helpers. |
| `taxonomy_expanded.py` | Expanded keyword lists and queries; order-independent multi-label keyword classifier. |
| `analyze_classification.py` | Diagnostics on the original 1,941-paper set (multi-label overlap, keyword-order effects). |
| `draw_handlabel_sample.py` | Random sample for manual labelling. |
| `collect_expanded.py` | Re-runs DBLP with the expanded queries; keeps only new papers. |
| `pilot_expand.py`, `collect_v2_venues.py` | Same queries, extra venues (NLP and security). |
| `snowball.py`, `snowball_nlp.py` | Citation snowball: in-venue references of corpus papers that are not yet in the corpus. |
| `screen_snowball.py` | Title screen of snowball candidates before enrichment. |
| `collect_census.py` | Full proceedings census per venue and year from DBLP. |
| `collect_neurips2025.py`, `collect_aies2025.py` | Scrapers for the two 2025 proceedings DBLP did not have yet. |
| `screen_census.py` | Title-screens the census and removes papers already classified. |
| `merge_census_cohorts.py` | Folds the 2025 scrapes into the census candidates. |
| `dedup_census.py`, `clean_titles.py` | Deduplication and HTML-entity cleanup of titles. |
| `estimate_comparable_2025.py` | Like-for-like 2025 count under the original title-query method. |

## 02_enrich — add abstracts

| Script | What it does |
|---|---|
| `enrich_abstracts_smoke.py` | Small coverage test of Semantic Scholar. |
| `enrich_abstracts.py` | Abstracts for the query pool (batch endpoint, DOI then title). |
| `enrich_snowball.py` | Abstracts for the screened snowball candidates. |

## 03_classify — LLM safety gate and provisional category

| Script | What it does |
|---|---|
| `classify_llm.py` | The canonical prompt, schema and category list. Can also call the API directly. |
| `merge_for_classify.py` | Merges query pool and snowball into one input with stable ids. |
| `build_batches.py` | Writes one prompt file per batch. |
| `gate_workflow.js` | Runs one agent per batch. |
| `merge_gate_results.py`, `merge_shards.py` | Collect batch results back onto the papers. |
| `prep_2025_batches.py`, `merge_2025_backfill.py` | The same steps for the 2025 backfill. |
| `derive_13cat.py` | Derives the 13-category corpus from the 16-category one. |
| `scan_container_records.py`, `apply_proceedings_drops.py` | Find and drop proceedings front-matter that is not a paper. |
| `analyze_lowconf_misc.py` | Looks at low-confidence and miscellaneous papers. |

## 04_validate — check the labels

| Script | What it does |
|---|---|
| `validate_compare.py` | Stored labels vs an independent blind re-labelling. |
| `build_val_opus_input.py`, `validate_opus.py` | 250-paper stratified sample re-labelled by a second Claude model. |
| `classify_hf.py`, `validate_crossfamily.py` | The same sample re-labelled by a local open-weights model, and its scoring. |
| `qwen_crossfamily_validate.py`, `openai_crossfamily_validate.py` | Gate and category agreement with Qwen and with an OpenAI model. |
| `build_pool_audit_input.py`, `merge_pool_audit.py`, `compare_pool_audit.py` | Blind re-classification audit of all pooled safety papers. |
| `verify_corpus.py` | Checks that corpus rows correspond to real papers. |

## 05_taxonomy — derive the categories bottom-up

| Script | What it does |
|---|---|
| `unsupervised_cluster.py` | TF-IDF/LSA k-means baseline. |
| `embed_cluster.py` | SPECTER embeddings and k-means. |
| `build_cluster_corpus_census.py`, `embed_census_block.py` | Build and embed the combined post-census corpus. |
| `bertopic_compare.py` | Tier 1: SPECTER → UMAP → HDBSCAN, compared with the LLM categories. Shared helpers for this stage. |
| `bertopic_tier2.py` | Tier 2: recursive clustering of the large LLM-safety cluster. |
| `bertopic_tier1_13cat.py`, `verify_tier2_probe_13cat.py` | Re-runs of both tiers on the 13-category scope. |
| `c0_separability_probe.py`, `c0_probe_ci.py` | Linear probe inside the large cluster, with bootstrap intervals. |
| `cluster_stability.py`, `cluster_stability_census.py`, `cluster_census_sweep.py` | Seed stability and hyperparameter sensitivity. |
| `apply_cluster_labels.py` | Assigns each paper its final category from its cluster. |

## 06_fulltext — full text and re-classification

| Script | What it does |
|---|---|
| `fulltext_pilot.py` | Feasibility pilot. |
| `build_fulltext_index.py`, `build_census_fulltext_index.py`, `build_index_12987.py` | Map each paper to open-access PDF URLs. |
| `fetch_fulltext.py`, `reextract_sections.py` | Download PDFs and extract method and conclusion sections. |
| `build_reclassify_sample.py`, `build_fulltext_reclass_full.py` | Prompt batches for re-classifying from full text (sample, then whole corpus). |
| `classify_fulltext_workflow.js` | Runs one agent per batch. |
| `score_fulltext_reclass_full.py` | Full-text labels vs abstract-only labels. |

## 07_open_problems — the "what is open" census

| Script | What it does |
|---|---|
| `build_op_batches.py`, `op_classify_workflow.js`, `merge_op_results.py`, `score_op_census.py` | Layer 2: classify every paper's closing section against its subfield's open-problem labels; prevalence per label. |
| `build_layer1_sample.py`, `layer1_deepcode_workflow.js`, `rebuild_layer1_prompts_fixed.py`, `merge_layer1.py` | Layer 1: deep coding of about 50 papers per subfield as ground truth. |
| `score_agreement.py` | Layer 1 vs Layer 2 agreement. |
| `gapmine_workflow.js`, `report_gapmine.py` | Looks for recurring open problems the label set missed. |
| `embed_op_sections.py` | SPECTER embeddings of the closing sections. |
| `op_cluster_validate.py`, `op_label_probe.py`, `op_merge_test.py` | Do the open-problem labels show up in the embeddings (clustering, one-vs-rest AUC, merging confusable labels). |
| `build_validation_xlsx.py` | Writes the validation workbook. |
| `*.sbatch` | SLURM jobs for the four embedding and probe scripts. |

## 08_stats — numbers for the paper

| Script | What it does |
|---|---|
| `paper_stats.py` | Counts, shares, growth and co-occurrence from the classified corpus. |
| `build_paper_numbers.py` | Every derived number the text cites, for the 10-venue corpus. |
| `diff_2025_stats.py` | Before/after comparison for the 2025 backfill. |
