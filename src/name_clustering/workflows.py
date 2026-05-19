from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .io import read_full_sample, read_pending, read_training, split_training_by_cluster_size, write_excel_sheets
from .model import ModelConfig, encode_texts, get_or_train_model


class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent = {item: item for item in items}

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def exact_name_map(df_train: pd.DataFrame) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for name_norm, group in df_train.groupby("name_norm"):
        ids = group["id"].dropna().astype(int).unique()
        if len(ids) == 1:
            mapping[name_norm] = int(ids[0])
    return mapping


def cluster_centroids(df_train: pd.DataFrame, embeddings: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    ids = sorted(df_train["id"].astype(int).unique())
    centroids = []
    for cluster_id in ids:
        idx = np.where(df_train["id"].astype(int).to_numpy() == cluster_id)[0]
        centroid = embeddings[idx].mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
    return np.vstack(centroids), ids


def nearest_centroid(embeddings: np.ndarray, centroids: np.ndarray, ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    scores = embeddings @ centroids.T
    best_idx = scores.argmax(axis=1)
    best_scores = scores[np.arange(scores.shape[0]), best_idx]
    best_ids = np.array([ids[idx] for idx in best_idx], dtype=int)
    return best_ids, best_scores


@dataclass
class AssignmentConfig:
    threshold: float = 0.95
    min_train_cluster_size: int = 2
    cache_path: str | None = "models/char_transformer_cache.pt"


def workflow_auto_cluster(
    train_path: str,
    full_path: str,
    output_path: str,
    name_col: str | None = None,
    assignment: AssignmentConfig = AssignmentConfig(),
    model_config: ModelConfig = ModelConfig(),
) -> None:
    df_train = read_training(train_path)
    df_full = read_full_sample(full_path, name_col=name_col)
    df_train_use, _ = split_training_by_cluster_size(df_train, assignment.min_train_cluster_size)
    model, stoi = get_or_train_model(df_train_use, assignment.cache_path, model_config)

    train_embeddings = encode_texts(model, stoi, df_train_use["name_norm"].tolist())
    centroids, centroid_ids = cluster_centroids(df_train_use, train_embeddings)
    exact = exact_name_map(df_train_use)

    missing_exact = ~df_full["name_norm"].isin(exact)
    full_embeddings = encode_texts(model, stoi, df_full.loc[missing_exact, "name_norm"].tolist())
    best_ids, best_scores = nearest_centroid(full_embeddings, centroids, centroid_ids)

    df_full["cluster_id"] = pd.NA
    df_full["assignment_score"] = pd.NA
    df_full["assignment_method"] = "unassigned"
    df_full.loc[df_full["name_norm"].isin(exact), "cluster_id"] = df_full.loc[df_full["name_norm"].isin(exact), "name_norm"].map(exact)
    df_full.loc[df_full["name_norm"].isin(exact), "assignment_score"] = 1.0
    df_full.loc[df_full["name_norm"].isin(exact), "assignment_method"] = "exact"

    target_index = df_full.index[missing_exact]
    if len(target_index) > 0:
        accepted = best_scores >= assignment.threshold
        df_full.loc[target_index[accepted], "cluster_id"] = best_ids[accepted]
        df_full.loc[target_index, "assignment_score"] = best_scores
        df_full.loc[target_index[accepted], "assignment_method"] = "centroid"

    next_id = max(centroid_ids) + 1 if centroid_ids else 1
    for idx in df_full.index[df_full["cluster_id"].isna()]:
        df_full.loc[idx, "cluster_id"] = next_id
        df_full.loc[idx, "assignment_method"] = "new_singleton"
        next_id += 1
    write_excel_sheets(output_path, {"full_sample_clustered": df_full})


def workflow_classify_singletons(
    train_path: str,
    pending_path: str,
    output_path: str,
    name_col: str | None = None,
    headerless: bool = False,
    assignment: AssignmentConfig = AssignmentConfig(),
    model_config: ModelConfig = ModelConfig(),
) -> None:
    df_train = read_training(train_path)
    df_train_use, df_train_singleton = split_training_by_cluster_size(df_train, assignment.min_train_cluster_size)
    df_pending = read_pending(pending_path, name_col=name_col, headerless=headerless)
    df_pending = df_pending[~df_pending["name_norm"].isin(set(df_train_use["name_norm"]))].copy()

    model, stoi = get_or_train_model(df_train_use, assignment.cache_path, model_config)
    train_embeddings = encode_texts(model, stoi, df_train_use["name_norm"].tolist())
    centroids, centroid_ids = cluster_centroids(df_train_use, train_embeddings)
    exact = exact_name_map(df_train_use)

    pending_embeddings = encode_texts(model, stoi, df_pending["name_norm"].tolist())
    best_ids, best_scores = nearest_centroid(pending_embeddings, centroids, centroid_ids)

    df_pending["assigned_id"] = pd.NA
    df_pending["assignment_score"] = best_scores
    df_pending["assignment_method"] = "unclassified"
    exact_mask = df_pending["name_norm"].isin(exact)
    df_pending.loc[exact_mask, "assigned_id"] = df_pending.loc[exact_mask, "name_norm"].map(exact)
    df_pending.loc[exact_mask, "assignment_score"] = 1.0
    df_pending.loc[exact_mask, "assignment_method"] = "exact"

    centroid_mask = (~exact_mask) & (df_pending["assignment_score"] >= assignment.threshold)
    df_pending.loc[centroid_mask, "assigned_id"] = best_ids[centroid_mask.to_numpy()]
    df_pending.loc[centroid_mask, "assignment_method"] = "centroid"

    assigned = df_pending[df_pending["assigned_id"].notna()].copy()
    assigned_train_shape = assigned[["assigned_id", "pending_name"]].rename(columns={"assigned_id": "id", "pending_name": "name_raw"})
    all_clusters = pd.concat([df_train_use[["id", "name_raw"]], assigned_train_shape], ignore_index=True)

    changed_ids = set(assigned["assigned_id"].astype(int))
    clusters_with_new = all_clusters[all_clusters["id"].astype(int).isin(changed_ids)].copy()
    unclassified = df_pending[df_pending["assigned_id"].isna()].copy()

    write_excel_sheets(
        output_path,
        {
            "sheet1_all_clusters": all_clusters,
            "sheet2_clusters_with_new": clusters_with_new,
            "sheet3_unclassified": unclassified,
            "sheet4_train_singletons": df_train_singleton,
        },
    )


def workflow_merge_training(
    train_path: str,
    output_path: str,
    threshold: float = 0.93,
    topk: int = 20,
    cache_path: str | None = "models/char_transformer_cache.pt",
    model_config: ModelConfig = ModelConfig(),
) -> None:
    df_train = read_training(train_path)
    model, stoi = get_or_train_model(df_train, cache_path, model_config)
    embeddings = encode_texts(model, stoi, df_train["name_norm"].tolist())
    centroids, ids = cluster_centroids(df_train, embeddings)
    scores = centroids @ centroids.T

    uf = UnionFind(ids)
    merge_edges = []
    for name_norm, group in df_train.groupby("name_norm"):
        overlap_ids = sorted(group["id"].dropna().astype(int).unique())
        if name_norm and len(overlap_ids) > 1:
            anchor = overlap_ids[0]
            for right_id in overlap_ids[1:]:
                uf.union(anchor, right_id)
                merge_edges.append(
                    {
                        "left_id": anchor,
                        "right_id": right_id,
                        "cosine": 1.0,
                        "method": "exact_name_overlap",
                    }
                )

    for i, left_id in enumerate(ids):
        nearest = np.argsort(scores[i])[::-1][1 : topk + 1]
        for j in nearest:
            score = float(scores[i, j])
            if score >= threshold:
                right_id = ids[j]
                uf.union(left_id, right_id)
                merge_edges.append({"left_id": left_id, "right_id": right_id, "cosine": score, "method": "centroid"})

    id_map = {old_id: uf.find(old_id) for old_id in ids}
    unique_new_ids = {root: idx + 1 for idx, root in enumerate(sorted(set(id_map.values())))}
    df_out = df_train.copy()
    df_out["old_id"] = df_out["id"]
    df_out["id"] = df_out["old_id"].map(lambda value: unique_new_ids[id_map[int(value)]])

    changed_new_ids = df_out.groupby("id")["old_id"].nunique()
    changed_new_ids = set(changed_new_ids[changed_new_ids > 1].index)
    changed_rows = df_out[df_out["id"].isin(changed_new_ids)].copy()

    write_excel_sheets(
        output_path,
        {
            "sheet1_merged_training": df_out[["id", "name_raw"]],
            "sheet2_changed_rows": changed_rows[["id", "name_raw", "old_id"]],
            "sheet3_merge_edges": pd.DataFrame(merge_edges),
        },
    )
