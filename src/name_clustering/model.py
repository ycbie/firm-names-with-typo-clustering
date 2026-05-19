import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PAD = "<PAD>"
UNK = "<UNK>"


@dataclass
class ModelConfig:
    seed: int = 42
    max_len: int = 80
    d_model: int = 192
    n_head: int = 6
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    epochs: int = 200
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.05
    early_stop_patience: int = 6
    early_stop_min_delta: float = 1e-4
    max_pos_pairs_per_class: int = 300
    temperature: float = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_char_vocab(texts: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = set()
    for text in tqdm(texts, desc="build char vocab", leave=False):
        chars.update(text)
    itos = [PAD, UNK] + sorted(chars)
    stoi = {ch: idx for idx, ch in enumerate(itos)}
    return stoi, {idx: ch for ch, idx in stoi.items()}


def encode_text(text: str, stoi: Dict[str, int], max_len: int):
    ids = [stoi.get(ch, stoi[UNK]) for ch in text][:max_len]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(stoi[PAD])
        mask.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


class CharTransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.emb = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.pos = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_head,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x_ids: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x_ids.shape
        pos_ids = torch.arange(seq_len, device=x_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.dropout(self.emb(x_ids) + self.pos(pos_ids))
        hidden = self.encoder(hidden, src_key_padding_mask=(x_mask == 0))
        hidden = self.ln(hidden)
        mask = x_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(pooled, p=2, dim=-1)


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], stoi: Dict[str, int], max_len: int):
        self.pairs = pairs
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        left, right = self.pairs[index]
        left_ids, left_mask = encode_text(left, self.stoi, self.max_len)
        right_ids, right_mask = encode_text(right, self.stoi, self.max_len)
        return left_ids, left_mask, right_ids, right_mask


def build_positive_pairs(df_train: pd.DataFrame, config: ModelConfig) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for _, group in df_train.groupby("id"):
        names = sorted(set(group["name_norm"].dropna().astype(str)))
        if len(names) < 2:
            continue
        all_pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
        if len(all_pairs) > config.max_pos_pairs_per_class:
            all_pairs = random.sample(all_pairs, config.max_pos_pairs_per_class)
        pairs.extend(all_pairs)
    random.shuffle(pairs)
    if not pairs:
        raise ValueError("No positive pairs were built. Training data needs at least one id with 2 names.")
    return pairs


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def evaluate(model: CharTransformerEncoder, loader: DataLoader, device: str, temperature: float) -> float:
    model.eval()
    losses = []
    for left_ids, left_mask, right_ids, right_mask in loader:
        left_ids, left_mask = left_ids.to(device), left_mask.to(device)
        right_ids, right_mask = right_ids.to(device), right_mask.to(device)
        losses.append(info_nce_loss(model(left_ids, left_mask), model(right_ids, right_mask), temperature).item())
    return float(np.mean(losses)) if losses else float("inf")


def train_char_transformer(df_train: pd.DataFrame, config: ModelConfig) -> Tuple[CharTransformerEncoder, Dict[str, int]]:
    set_seed(config.seed)
    device = get_device()
    df_train = df_train[df_train["name_norm"] != ""].copy()
    stoi, _ = build_char_vocab(df_train["name_norm"].tolist())
    pairs = build_positive_pairs(df_train, config)

    split = int(len(pairs) * (1 - config.val_ratio))
    split = max(1, min(split, len(pairs)))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:] if split < len(pairs) else pairs[: max(1, min(256, len(pairs)))]

    train_loader = DataLoader(PairDataset(train_pairs, stoi, config.max_len), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(PairDataset(val_pairs, stoi, config.max_len), batch_size=config.batch_size, shuffle=False)

    model = CharTransformerEncoder(len(stoi), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")
    stale = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for left_ids, left_mask, right_ids, right_mask in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            left_ids, left_mask = left_ids.to(device), left_mask.to(device)
            right_ids, right_mask = right_ids.to(device), right_mask.to(device)
            loss = info_nce_loss(model(left_ids, left_mask), model(right_ids, right_mask), config.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss = evaluate(model, val_loader, device, config.temperature)
        print(f"epoch={epoch:03d} train_loss={np.mean(train_losses):.5f} val_loss={val_loss:.5f}")
        if val_loss < best_val - config.early_stop_min_delta:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.early_stop_patience:
                print("early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, stoi


def save_model_cache(model: CharTransformerEncoder, stoi: Dict[str, int], path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "stoi": stoi, "config": asdict(model.config)}, path)


def load_model_cache(path: str) -> Tuple[CharTransformerEncoder, Dict[str, int]]:
    payload = torch.load(path, map_location=get_device())
    config = ModelConfig(**payload["config"])
    model = CharTransformerEncoder(len(payload["stoi"]), config).to(get_device())
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["stoi"]


def get_or_train_model(df_train: pd.DataFrame, cache_path: str | None, config: ModelConfig):
    if cache_path:
        from pathlib import Path

        if Path(cache_path).exists():
            print(f"loading cached model: {cache_path}")
            return load_model_cache(cache_path)
    model, stoi = train_char_transformer(df_train, config)
    if cache_path:
        save_model_cache(model, stoi, cache_path)
    return model, stoi


@torch.no_grad()
def encode_texts(model: CharTransformerEncoder, stoi: Dict[str, int], texts: List[str], batch_size: int = 512) -> np.ndarray:
    device = get_device()
    model.eval()
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode texts"):
        batch = texts[start : start + batch_size]
        ids, masks = zip(*(encode_text(text, stoi, model.config.max_len) for text in batch))
        x_ids = torch.stack(ids).to(device)
        x_mask = torch.stack(masks).to(device)
        vectors.append(model(x_ids, x_mask).cpu().numpy())
    return np.vstack(vectors) if vectors else np.empty((0, model.config.d_model), dtype=np.float32)
