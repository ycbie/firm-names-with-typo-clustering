# Fine-Tuning Workflow

The repository now supports a first-train-then-fine-tune workflow.

## Why Fine-Tune

The original project repeatedly updated the training set after manual review. Re-training from scratch is faithful but expensive. Fine-tuning keeps the learned character-level similarity model and adapts it to newly confirmed training rows.

This is useful when the training set grows gradually through:

- confirmed singleton assignments;
- confirmed training-cluster merges;
- confirmed pending-name reclusters.

## Round 1: Train From Scratch

Use a stable cache path and omit `--fine-tune`:

```powershell
name-cluster auto-cluster `
  --train train_round1.xlsx `
  --full full_names.xlsx `
  --output reports/round1_auto_cluster.xlsx `
  --cache-path models/char_transformer.pt
```

The command trains a Transformer from scratch and saves it to `models/char_transformer.pt`.

## Later Rounds: Fine-Tune

After manual review, update the training set and reuse the same cache:

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

With `--fine-tune`, the command loads the cached model, continues contrastive training on the updated training set, saves the updated model back to the same cache, and then performs the requested clustering or classification step.

## What Changes Between Rounds

Before a fine-tuning round:

- the model cache contains the previous Transformer;
- the training file contains newly human-confirmed names and IDs;
- pending or full-sample files contain names to classify or cluster.

After a fine-tuning round:

- the cache contains the updated Transformer;
- outputs contain model-proposed assignments, merge candidates, or pending-name clusters;
- humans review the changed clusters before the next training file is created.

## Practical Settings

Use fewer epochs and a smaller learning rate for fine-tuning than for the first training round. A typical pattern is:

- first training: `--epochs 100` to `200`, `--lr 2e-4`;
- fine-tuning: `--epochs 10` to `40`, `--lr 5e-5`.

Keep the architecture settings stable across rounds, especially `--max-len`, `--d-model`, number of heads, and number of layers. If architecture settings differ from the cache, the code retrains from scratch instead of loading an incompatible model.

## Caveat

Fine-tuning is faster, but it can inherit mistakes from earlier rounds. Conservative thresholds and manual review remain essential. If the training set has been heavily corrected or reorganized, a fresh from-scratch training run can still be preferable.
