# Myanmar Exporter Name Clustering

Character-level Transformer workflow for clustering typo-heavy English company names.

This repository is a cleaned version of the Myanmar exporter-name cleaning project. The real exporter names, intermediate spreadsheets, model checkpoints, and final empirical data files are not included. The `examples/` folder contains a small synthetic dataset that mimics common noise patterns such as typos, abbreviations, punctuation changes, and spacing differences.

## Workflow

The project follows the two core workflows described in the work report:

1. **Automatic clustering**: learn character-level firm-name similarity from a labeled training set, build cluster centroids, and assign IDs to the full sample.
2. **Singleton classification**: train on cleaned clusters, classify pending singleton names into existing reliable clusters with a strict threshold, and send changed clusters to human review.

There is also a training-set purification command that proposes high-threshold merges between existing training clusters.

The code is a refactor of the original dated research scripts, not a different algorithm. It keeps the original project's core choices: minimal normalization only, a character-level Transformer trained with contrastive pairs, exact normalized matches first, centroid cosine thresholds, conservative singleton retention, and human review of changed clusters.

## First Train, Then Fine-Tune

The recommended iterative use is:

1. Train the first Transformer from scratch with `auto-cluster` or `classify-singletons`.
2. Manually review changed clusters and update the training set.
3. Reuse the same `--cache-path` and add `--fine-tune` for later rounds.

Example first round:

```powershell
name-cluster auto-cluster `
  --train train_round1.xlsx `
  --full full_names.xlsx `
  --output reports/round1.xlsx `
  --cache-path models/char_transformer.pt
```

Example later round after human review:

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

If the cache does not exist, the command trains from scratch. If the cache exists and the model structure matches, `--fine-tune` continues training from that model and overwrites the cache. If the model structure changed, the code retrains from scratch because the old cache is no longer compatible.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Demo With Synthetic Data

The demo uses fake company names only.

```powershell
name-cluster auto-cluster `
  --train examples/fake_training.csv `
  --full examples/fake_full_sample.csv `
  --output reports/demo_auto_cluster.xlsx `
  --threshold 0.90 `
  --min-cluster-size 2 `
  --epochs 20 `
  --batch-size 16 `
  --d-model 96 `
  --cache-path models/demo_char_transformer.pt
```

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

To demonstrate fine-tuning on the same synthetic setup:

```powershell
name-cluster classify-singletons `
  --train examples/fake_training.csv `
  --pending examples/fake_pending.csv `
  --output reports/demo_singleton_finetuned.xlsx `
  --threshold 0.90 `
  --epochs 5 `
  --lr 5e-5 `
  --batch-size 16 `
  --d-model 96 `
  --cache-path models/demo_char_transformer.pt `
  --fine-tune
```

```powershell
name-cluster recluster-pending `
  --train examples/fake_training.csv `
  --pending examples/fake_pending.csv `
  --output reports/demo_pending_recluster.xlsx `
  --threshold 0.90 `
  --epochs 20 `
  --batch-size 16 `
  --d-model 96 `
  --cache-path models/demo_char_transformer.pt
```

## Expected Input Format

Training data must have at least two columns:

```text
id,name
1,Golden River Trading Co Ltd
1,Golden River Tradng Company Limited
```

Full-sample or pending data must include a company-name column. By default, the first column is used. You can pass `--name-col your_column_name`.

## Data Policy

Real raw data and generated outputs are deliberately excluded by `.gitignore`. Keep production data under `data/raw/`, `data/interim/`, `data/processed/`, or outside this repository. Keep model checkpoints under `models/`.

## Original Script Mapping

See [docs/original_code_mapping.md](docs/original_code_mapping.md) for how the cleaned modules correspond to the original Python files.
