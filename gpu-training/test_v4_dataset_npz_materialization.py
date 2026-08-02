from __future__ import annotations

from collections import Counter
from dataclasses import fields
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import warnings
import zipfile

import numpy as np

from v4_dataset import (
    V4TrajectoryDataset,
    V4TrajectoryTensors,
    create_v4_smoke_dataset,
    load_v4_dataset_npz,
    save_v4_dataset_npz,
)
from v4_model import V4ActorConfig, V4CriticConfig


def _tiny_configs() -> tuple[V4ActorConfig, V4CriticConfig]:
    return (
        V4ActorConfig(
            max_players=4,
            max_history=2,
            d_model=16,
            layers=1,
            heads=4,
            feedforward=32,
            action_hidden=12,
        ),
        V4CriticConfig(
            privileged_features=12,
            d_model=16,
            hidden_layers=1,
            action_hidden=12,
        ),
    )


class _CountingArchive:
    def __init__(self, archive: np.lib.npyio.NpzFile) -> None:
        self._archive = archive
        self.files = tuple(archive.files)
        self.read_counts: Counter[str] = Counter()
        self.materialized: dict[str, np.ndarray] = {}

    def __enter__(self) -> "_CountingArchive":
        return self

    def __exit__(self, *args: object) -> None:
        self._archive.close()

    def __getitem__(self, name: str) -> np.ndarray:
        self.read_counts[name] += 1
        value = self._archive[name]
        self.materialized[name] = value
        return value


class V4DatasetNpzMaterializationTests(unittest.TestCase):
    def _save_fixture(self, root: Path) -> tuple[Path, V4TrajectoryDataset]:
        actor_config, critic_config = _tiny_configs()
        dataset = create_v4_smoke_dataset(
            actor_config,
            critic_config,
            trajectories=2,
            time_steps=3,
            seed=20260802,
        )
        path = root / "small-real-fixture.npz"
        save_v4_dataset_npz(dataset, path)
        return path, dataset

    def test_real_fixture_reads_each_member_once_without_core_array_copies(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, expected = self._save_fixture(Path(directory))
            real_np_load = np.load
            opened: list[_CountingArchive] = []

            def counting_load(*args: object, **kwargs: object) -> _CountingArchive:
                wrapped = _CountingArchive(real_np_load(*args, **kwargs))
                opened.append(wrapped)
                return wrapped

            with mock.patch("v4_dataset.np.load", side_effect=counting_load):
                loaded = load_v4_dataset_npz(path)

            self.assertEqual(len(opened), 1)
            archive = opened[0]
            self.assertEqual(set(archive.read_counts), set(archive.files))
            self.assertTrue(
                all(count == 1 for count in archive.read_counts.values())
            )
            for field in fields(V4TrajectoryTensors):
                source_array = archive.materialized[field.name]
                loaded_tensor = getattr(loaded.tensors, field.name)
                self.assertEqual(loaded_tensor.data_ptr(), source_array.ctypes.data)

            self.assertEqual(loaded.fingerprint, expected.fingerprint)
            self.assertEqual(
                loaded.loss_contract_fingerprint,
                expected.loss_contract_fingerprint,
            )
            self.assertEqual(
                loaded.loss_eligibility.preparation_format,
                expected.loss_eligibility.preparation_format,
            )
            self.assertEqual(
                loaded.loss_eligibility.preparation_version,
                expected.loss_eligibility.preparation_version,
            )

    def test_duplicate_npz_member_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = self._save_fixture(root)
            duplicate = root / "duplicate-member.npz"
            with zipfile.ZipFile(source, "r") as archive:
                members = [
                    (member, archive.read(member.filename))
                    for member in archive.infolist()
                ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate, "w") as archive:
                    for member, payload in members:
                        archive.writestr(member, payload)
                    member, payload = members[0]
                    archive.writestr(member.filename, payload)

            with self.assertRaisesRegex(ValueError, "duplicate member key"):
                load_v4_dataset_npz(duplicate)


if __name__ == "__main__":
    unittest.main()
