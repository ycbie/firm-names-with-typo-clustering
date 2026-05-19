# Original Code Mapping

This repository is intentionally a cleaned, reusable refactor of the original working folder. It does not introduce a new unrelated method. The original folder contained many dated scripts and spreadsheet outputs; this repository keeps the final algorithmic workflow and removes hard-coded local paths, private data files, color-formatting details, and one-off exploratory variants.

## Core Components

| Clean module | Original source scripts | What is preserved |
| --- | --- | --- |
| `normalization.py` | 0221 model-cache/classification script; 0224 training-merge script; 0224 singleton-reclustering script; 0225 final classification script | Minimal normalization: lowercase, replace non-alphanumeric characters with spaces, collapse spaces. No stopword list and no suffix rules. |
| `model.py` | Same four scripts | Character vocabulary, padded character encoding, `CharTransformerEncoder`, mean pooling over non-padding tokens, L2-normalized embeddings, positive same-ID pairs, symmetric InfoNCE loss, validation loss, early stopping, model cache. |
| `workflows.py::workflow_auto_cluster` | PDF Workflow A, plus the centroid/exact assignment pattern from the 0221 and 0225 final classification scripts | Train on labeled IDs, compute cluster centroids, exact normalized match first, nearest-centroid cosine assignment above threshold, otherwise create singleton IDs. |
| `workflows.py::workflow_classify_singletons` | PDF Workflow B, mainly the 0225 final classification script | Restrict candidate clusters by minimum training cluster size, remove names already present in training after normalization, assign pending names only by exact match or high-threshold centroid similarity, export changed clusters and unclassified names. |
| `workflows.py::workflow_merge_training` | 0224 training-merge script | Training-set purification by exact-name-overlap and high-threshold centroid cosine merge candidates, exported for manual review. |
| `workflows.py::workflow_recluster_pending` | 0224 singleton-reclustering script | Internal reclustering of pending names by exact normalized duplicates and embedding cosine connected components, exported for manual review. |
| `cli.py` | Refactor layer only | Replaces hard-coded Windows paths with command-line arguments. This is an engineering wrapper around the original workflow, not a new algorithm. |

## Added Engineering Optimization

The current repository also adds an optional `--fine-tune` switch. This does not change the modeling idea; it changes how later rounds are trained. The first round can still train from scratch, while later human-reviewed rounds can continue training from the cached Transformer instead of starting over.

## Deliberately Removed From GitHub

- Real Myanmar exporter names and all private raw/intermediate/final data.
- Model checkpoints such as `.pt` and `.pth`.
- Large `.xlsx`, `.dta`, and generated result files.
- Hard-coded local file paths.
- Excel coloring helpers used only for manual viewing convenience.
- Early fuzzy-matching experiments and one-off tuning scripts.

## Method Boundary

The repository does not add external semantic embeddings, LLM matching, rule-based stopwords, suffix dictionaries, address logic, HS-code logic, exporter-code logic, or financial-data logic. It stays within the method described in the report: character-level Transformer embeddings plus conservative threshold-based clustering/classification and human review.

## How To Read The Demo

The `examples/` CSV files are synthetic. They are designed to show the same kind of spelling noise as the real task:

- missing or swapped letters, such as `Tradng`, `Rivar`, and `Jaed`;
- punctuation and spacing variants;
- abbreviated legal terms such as `Co`, `Ltd`, and `Limited`;
- names that share tokens but should remain separate, such as `Asia Jade Export` vs. `Asia Ruby Mining`.

The demo is therefore a small teaching example for the workflow, not a substitute for the private Myanmar exporter dataset.
