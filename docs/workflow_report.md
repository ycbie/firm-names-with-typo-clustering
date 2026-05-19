# Workflow Summary

## Background

The original task involved 50,657 unique Myanmar exporter names after raw string deduplication. The names contain typos, abbreviations, punctuation differences, inconsistent spacing, and no unified firm identifier. The objective is to identify different spellings that refer to the same firm while avoiding false merges between distinct firms.

## Why Not Fuzzy Matching

Early fuzzy matching used manually built stopwords such as common country, company, and legal-suffix terms. Repeated threshold and stopword tuning produced too many singletons and too few useful large clusters. This motivated a model that learns character-level similarity patterns directly.

## Character-Level Transformer

Firm names are treated as character sequences. The encoder is trained from scratch with positive name pairs from known same-firm clusters and in-batch negatives through an InfoNCE-style contrastive loss.

The model output is a normalized embedding. Cluster prototypes are computed as the mean embedding of names within each training ID.

## Workflow A: Automatic Clustering

1. Minimally normalize names: lowercase, replace non-alphanumeric characters with spaces, collapse repeated spaces.
2. Train a character-level Transformer from labeled clusters.
3. Compute one centroid per training cluster.
4. For each full-sample name, first assign exact normalized matches.
5. For the rest, assign the nearest centroid only if cosine similarity exceeds a chosen threshold.
6. Keep unassigned names as new singleton IDs.

## Workflow B: Singleton Classification

1. Train or load the Transformer on cleaned clusters.
2. Restrict candidate clusters to reliable clusters, usually size at least 3.
3. Assign pending singleton names only by exact match or high-threshold centroid similarity.
4. Export changed clusters for human review.
5. Merge confirmed names into the training set and repeat until no reliable additions remain.

## Training-Set Purification

The `merge-training` command proposes cluster merges by comparing centroids of existing training IDs. These are high-threshold merge candidates for manual inspection, not blind final truth.

## Final Reported Result

The cleaned project outcome described in the report was 7,531 clusters covering 32,730 of 50,657 unique firm names. Remaining names were left as singletons because conservative precision is more important than forcing all names into clusters.
