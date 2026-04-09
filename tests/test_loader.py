import numpy as np
import pytest

from orscale.data.loader import ShardedTokenDataset, create_dataloader


def _write_bin_shard(path, tokens):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)
    token_array = np.asarray(tokens, dtype=np.uint16)

    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(token_array.tobytes())


def test_create_dataloader_keeps_bin_shards_lazy(tmp_path):
    shard_a = tmp_path / "train_a.bin"
    shard_b = tmp_path / "train_b.bin"
    _write_bin_shard(shard_a, np.arange(0, 9))
    _write_bin_shard(shard_b, np.arange(100, 109))

    loader = create_dataloader(
        {"train_pattern": str(tmp_path / "train_*.bin"), "streaming": False},
        seq_len=4,
        batch_size=2,
        split="train",
        num_workers=0,
    )

    dataset = loader.dataset
    assert isinstance(dataset, ShardedTokenDataset)
    assert dataset._shards is None
    assert len(dataset) == 4

    x0, y0 = dataset[0]
    assert dataset._shards is not None
    np.testing.assert_array_equal(x0.numpy(), np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(y0.numpy(), np.array([1, 2, 3, 4]))

    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 4)
    assert batch_y.shape == (2, 4)


def test_sharded_dataset_spans_multiple_files(tmp_path):
    shard_a = tmp_path / "train_a.bin"
    shard_b = tmp_path / "train_b.bin"
    _write_bin_shard(shard_a, np.arange(0, 9))
    _write_bin_shard(shard_b, np.arange(100, 109))

    dataset = ShardedTokenDataset([str(shard_a), str(shard_b)], seq_len=4)

    samples = [dataset[i][0].numpy() for i in range(len(dataset))]
    np.testing.assert_array_equal(samples[0], np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(samples[1], np.array([4, 5, 6, 7]))
    np.testing.assert_array_equal(samples[2], np.array([100, 101, 102, 103]))
    np.testing.assert_array_equal(samples[3], np.array([104, 105, 106, 107]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
