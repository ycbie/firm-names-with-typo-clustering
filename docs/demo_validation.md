# Demo Validation

The synthetic demo was run locally from the same code pushed to GitHub.

## Automatic Clustering

Command:

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

Observed behavior:

- Clear typo variants such as `Golden Rivar Tradng Co Ltd`, `Asia Jaed Exports Company`, `Myanmr Star Food Co Ltd`, and `Sunpeak Logstics Limited` were assigned to the intended existing clusters.
- Ambiguous or different-firm names such as `Golden River Export House`, `Asia Ruby Mining Co`, `Myanmar Sunrise Textile Ltd`, and `Moon Lake Agricultural Export` were kept as new singleton IDs.

This matches the project objective: prefer conservative singleton retention over false merges.

## Singleton Classification

Command:

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

Observed behavior:

- Five likely typo variants were assigned into existing clusters.
- `Delta Marine Product Exporter` and the synthetic `Northern Timber...` names stayed unclassified against existing training clusters at the chosen threshold.

## Interpretation

The demo is intentionally small and synthetic. It proves that the repository can reproduce the workflow and that the model learns useful typo-level similarity on a toy dataset. It should not be interpreted as a benchmark for the full Myanmar exporter dataset.

For real data, quality still depends on:

- the cleanliness of the labeled training clusters;
- high assignment thresholds;
- manual review of changed clusters;
- repeating the human-in-the-loop update cycle.

## Pending-Name Reclustering

The repository also includes `recluster-pending`, corresponding to the original singleton reclustering script. It clusters pending names among themselves by exact normalized duplicates and Transformer-embedding cosine connected components, then exports candidate clusters for manual review.
