from pathlib import Path
from typing import Tuple

import pandas as pd

from .normalization import normalize_name


def read_training(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("Training file must contain at least two columns: id and firm name.")
    df = df.iloc[:, :2].copy()
    df.columns = ["id", "name_raw"]
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    if df["id"].isna().any() or (df["id"] % 1 != 0).any():
        raise ValueError("Training id column must be integer-like.")
    df["id"] = df["id"].astype(int)
    df["name_norm"] = df["name_raw"].apply(normalize_name)
    return df[df["name_norm"] != ""].reset_index(drop=True)


def read_full_sample(path: str | Path, name_col: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    if name_col is None:
        name_col = str(df.columns[0])
    if name_col not in df.columns:
        raise ValueError(f"Column not found: {name_col}")
    out = df.copy()
    out["name_raw"] = out[name_col]
    out["name_norm"] = out["name_raw"].apply(normalize_name)
    return out


def read_pending(path: str | Path, name_col: str | None = None, headerless: bool = False) -> pd.DataFrame:
    if headerless:
        df = pd.read_excel(path, header=None)
        df.columns = [f"col_{idx}" for idx in range(df.shape[1])]
        name_col = "col_0" if name_col is None else name_col
    else:
        df = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
        name_col = str(df.columns[0]) if name_col is None else name_col
    if name_col not in df.columns:
        raise ValueError(f"Column not found: {name_col}")
    out = df.copy()
    out["pending_name"] = out[name_col]
    out["name_norm"] = out["pending_name"].apply(normalize_name)
    return out[out["name_norm"] != ""].reset_index(drop=True)


def split_training_by_cluster_size(df_train: pd.DataFrame, min_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = df_train.groupby("id").size()
    keep_ids = set(counts[counts >= min_size].index)
    return df_train[df_train["id"].isin(keep_ids)].copy(), df_train[~df_train["id"].isin(keep_ids)].copy()


def write_excel_sheets(path: str | Path, sheets: dict[str, pd.DataFrame]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
