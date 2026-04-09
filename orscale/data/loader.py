"""
Data loading for OrScale training.

Supports two data sources:
1. modded-nanogpt pre-tokenized .bin files (header: magic=20240520, version=1,
   num_tokens, then uint16 token data).
2. HuggingFace datasets with on-the-fly GPT-2 tokenization (fallback).

Two batching modes:
- Fixed-length: sequential chunks of tokens (simple, fast).
- BOS-aligned: each sequence starts at a document boundary (fairer evaluation).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler


# ---------------------------------------------------------------------------
# Binary shard loading (modded-nanogpt format)
# ---------------------------------------------------------------------------

def load_bin_shard(path: str | Path) -> np.ndarray:
    """Load a modded-nanogpt .bin shard and return tokens as np.uint16 array."""
    path = Path(path)
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, f"Bad magic number in {path}"
    assert header[1] == 1, f"Unsupported version {header[1]} in {path}"
    num_tokens = int(header[2])

    tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(num_tokens,))
    return np.array(tokens)


def load_bin_shards(pattern: str) -> np.ndarray:
    """Load and concatenate all .bin shards matching a glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    shards = [load_bin_shard(f) for f in files]
    return np.concatenate(shards)


# ---------------------------------------------------------------------------
# Fixed-length token dataset
# ---------------------------------------------------------------------------

class TokenDataset(Dataset):
    """
    Dataset that yields fixed-length token chunks from a flat token array.

    Each sample is (input_ids[i:i+seq_len], targets[i+1:i+seq_len+1]).
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_samples = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ---------------------------------------------------------------------------
# Streaming token iterator (for large datasets / DDP)
# ---------------------------------------------------------------------------

class StreamingTokenIterator:
    """
    Streams fixed-length chunks from shards, cycling through files.
    Supports DDP by partitioning shards across ranks.

    Yields (input_ids, targets) tuples, each of shape (batch_size, seq_len).
    """

    def __init__(
        self,
        file_pattern: str,
        seq_len: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.files = sorted(glob.glob(file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No files match: {file_pattern}")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.tokens_per_step = batch_size * seq_len
        self.rng = np.random.RandomState(seed + rank)

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for file_path in self.files:
            tokens = load_bin_shard(file_path)

            # Partition the shard across ranks
            total_tokens = len(tokens)
            chunk_per_rank = total_tokens // self.world_size
            start = self.rank * chunk_per_rank
            end = start + chunk_per_rank
            local_tokens = tokens[start:end]

            # Yield batches from this shard
            pos = 0
            while pos + self.tokens_per_step + 1 <= len(local_tokens):
                buf = local_tokens[pos : pos + self.tokens_per_step + 1]
                buf = buf.astype(np.int64)

                x = torch.from_numpy(buf[:-1]).view(self.batch_size, self.seq_len)
                y = torch.from_numpy(buf[1:]).view(self.batch_size, self.seq_len)
                yield x, y

                pos += self.tokens_per_step


# ---------------------------------------------------------------------------
# HuggingFace fallback
# ---------------------------------------------------------------------------

class HFTokenDataset(Dataset):
    """
    Fallback dataset that loads a HuggingFace dataset, tokenizes on the fly
    with GPT-2 tokenizer, and yields fixed-length chunks.
    """

    def __init__(
        self,
        dataset_name: str = "openwebtext",
        split: str = "train",
        seq_len: int = 1024,
        max_tokens: int | None = None,
    ):
        try:
            from datasets import load_dataset
            import tiktoken
        except ImportError:
            raise ImportError(
                "Install datasets and tiktoken for HuggingFace fallback: "
                "pip install datasets tiktoken"
            )

        enc = tiktoken.get_encoding("gpt2")
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

        all_tokens = []
        for example in ds:
            text = example.get("text", "")
            tokens = enc.encode_ordinary(text)
            all_tokens.extend(tokens)
            if max_tokens and len(all_tokens) >= max_tokens:
                all_tokens = all_tokens[:max_tokens]
                break

        self.tokens = np.array(all_tokens, dtype=np.uint16)
        self.seq_len = seq_len
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dataloader(
    data_config: dict,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader | StreamingTokenIterator:
    """
    Create a data loader from config.

    Config keys:
        train_pattern / val_pattern: glob for .bin files (preferred)
        hf_dataset: HuggingFace dataset name (fallback)
        streaming: if True, use streaming iterator instead of Dataset

    Returns a DataLoader or StreamingTokenIterator.
    """
    pattern_key = "train_pattern" if split == "train" else "val_pattern"
    pattern = data_config.get(pattern_key)

    if pattern and glob.glob(pattern):
        if data_config.get("streaming", False):
            return StreamingTokenIterator(
                pattern, seq_len, batch_size,
                rank=rank, world_size=world_size, seed=seed,
            )

        tokens = load_bin_shards(pattern)
        dataset = TokenDataset(tokens, seq_len)
    else:
        hf_name = data_config.get("hf_dataset", "openwebtext")
        hf_split = "train" if split == "train" else "validation"
        max_tokens = data_config.get("max_tokens")
        dataset = HFTokenDataset(hf_name, hf_split, seq_len, max_tokens)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=(split == "train"),
            seed=seed,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
