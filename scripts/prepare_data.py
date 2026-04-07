#!/usr/bin/env python3
"""
Download and tokenize FineWeb-Edu into sharded .bin files.

Produces the same format as modded-nanogpt: a 256-int32 header followed by
uint16 token data.  Shard 0 is reserved for validation; the rest are training.

Usage:
    python scripts/prepare_data.py --version 10B
    python scripts/prepare_data.py --version 100B --shard_size 200000000 --out data/fineweb100B
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def write_datafile(filename: str, toks: np.ndarray) -> None:
    assert len(toks) < 2**31
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520   # magic
    header[1] = 1          # version
    header[2] = len(toks)  # token count
    if not isinstance(toks, np.ndarray) or toks.dtype != np.uint16:
        assert all(0 <= t < 2**16 for t in toks)
        toks = np.array(toks, dtype=np.uint16)
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb data for OrScale")
    parser.add_argument("-v", "--version", type=str, default="10B",
                        choices=["10B", "100B"], help="FineWeb subsample size")
    parser.add_argument("-s", "--shard_size", type=int, default=10**8,
                        help="Tokens per shard (default: 100M)")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="Output directory (default: data/fineweb{version})")
    args = parser.parse_args()

    remote_name = "sample-10BT" if args.version == "10B" else "sample-100BT"
    out_dir = args.out or os.path.join("data", f"fineweb{args.version}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading HuggingFaceFW/fineweb ({remote_name}) ...")
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def tokenize(doc):
        tokens = [eot] + enc.encode_ordinary(doc["text"])
        tokens_np = np.array(tokens, dtype=np.uint16)
        return tokens_np

    nprocs = max(1, os.cpu_count() - 2)
    shard_size = args.shard_size

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tok",
                                        desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                fname = os.path.join(out_dir, f"fineweb_{split}_{shard_index:06d}.bin")
                remainder = shard_size - token_count
                if progress_bar is not None:
                    progress_bar.update(remainder)
                all_tokens[token_count : token_count + remainder] = tokens[:remainder]
                write_datafile(fname, all_tokens)
                shard_index += 1
                progress_bar = None
                leftover = len(tokens) - remainder
                all_tokens[:leftover] = tokens[remainder:]
                token_count = leftover

        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            fname = os.path.join(out_dir, f"fineweb_{split}_{shard_index:06d}.bin")
            write_datafile(fname, all_tokens[:token_count])

    print(f"Done. {shard_index + 1} shards written to {out_dir}/")


if __name__ == "__main__":
    main()
