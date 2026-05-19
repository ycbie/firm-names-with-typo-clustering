# Firm Names With Typo Clustering

Character-level Transformer workflow for clustering noisy English company names.

This repo is a cleaned demo version of a Myanmar exporter-name cleaning project. The real data is not included; the `examples/` folder uses synthetic company names with similar typo patterns.

## Background

The original task had 50,000+ distinct exporter-name strings. Besides company names, the trade records also had customs/export-station information, HS codes, exporter/company codes, and an incomplete list of correct company names.

Those extra fields were useful, but messy. HS codes and company codes were missing for some records and could be wrong. I used them as human-checking evidence, not as rules for firm identity.

The first training set came from fuzzy matching typo-heavy names against the incomplete correct-name list, then manually checking the candidates with the auxiliary fields. That gave roughly 3,000 initial training rows. The Transformer workflow starts from that kind of `id,name` training set.

## What It Does

The project tries to group different spellings of the same firm while avoiding false merges.

The model is intentionally conservative:

- minimal normalization only;
- character-level Transformer embeddings;
- exact normalized matches first;
- centroid cosine similarity with a threshold;
- uncertain names stay as singletons;
- changed clusters are meant to be reviewed by a human.

## Demo Data

The demo data is small and fake:

- `fake_training.csv`: 24 names for 6 known firms;
- `fake_full_sample.csv`: 18 names to cluster;
- `fake_pending.csv`: 9 pending names, including 3 variants of a new unseen firm.

No real company names are included.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Quick Demo

Train a small character Transformer and cluster the fake full sample:

```powershell
name-cluster auto-cluster `
  --train examples/fake_training.csv `
  --full examples/fake_full_sample.csv `
  --output reports/demo_auto_cluster.xlsx `
  --threshold 0.90 `
  --epochs 20 `
  --batch-size 16 `
  --d-model 96 `
  --cache-path models/demo_char_transformer.pt
```

Then try assigning pending names into existing clusters:

```powershell
name-cluster classify-singletons `
  --train examples/fake_training.csv `
  --pending examples/fake_pending.csv `
  --output reports/demo_singleton_classification.xlsx `
  --threshold 0.90 `
  --epochs 20 `
  --batch-size 16 `
  --d-model 96 `
  --cache-path models/demo_char_transformer.pt
```

## Real Workflow

In a real cleaning job, the loop looks like this:

1. Build an initial `id,name` training set by fuzzy matching and manual checking.
2. Run `auto-cluster` on the full name list.
3. Review suspicious clusters and update the training set.
4. Run `merge-training` to find possible training-set merges.
5. Run `classify-singletons` to put high-confidence singleton names into existing clusters.
6. Run `recluster-pending` when remaining names may form new clusters among themselves.
7. Fine-tune the cached Transformer after each reviewed update.

First round, train from scratch:

```powershell
name-cluster auto-cluster `
  --train train_round1.xlsx `
  --full full_names.xlsx `
  --output reports/round1.xlsx `
  --cache-path models/char_transformer.pt
```

Later rounds, continue from the cached model:

```powershell
name-cluster classify-singletons `
  --train train_round2_reviewed.xlsx `
  --pending pending_round2.xlsx `
  --output reports/round2_singletons.xlsx `
  --cache-path models/char_transformer.pt `
  --fine-tune `
  --epochs 30 `
  --lr 5e-5
```

## Commands

- `auto-cluster`: train or load a model, assign IDs to a full sample, and create singleton IDs for names below threshold.
- `classify-singletons`: assign pending names into existing reliable clusters; output changed clusters for review.
- `merge-training`: propose high-confidence merges among existing training IDs.
- `recluster-pending`: cluster pending names among themselves and output new candidates for review.

## Input Format

Training data needs at least two columns:

```text
id,name
1,Golden River Trading Co Ltd
1,Golden River Tradng Company Limited
2,Asia Jade Export Company
2,Asia Jaed Export Co
```

Full-sample and pending files need a company-name column. By default, the first column is used. CSV and Excel files are supported.

## Docs

- [Workflow report summary](docs/workflow_report.md)
- [Original code mapping](docs/original_code_mapping.md)
- [Fine-tuning workflow](docs/fine_tuning_workflow.md)
- [Demo validation](docs/demo_validation.md)
