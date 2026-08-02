from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from v5_provenance import (
    V5_EVALUATION_BACKEND_POLICY,
    build_v5_evaluation_provenance,
    resolve_v5_evaluation_source_binding,
    v5_evaluation_runtime_provenance,
    validate_v5_evaluation_provenance,
)


SOURCE_FILES = (
    "gpu-training/v4_env.py",
    "gpu-training/v5_evaluate.py",
    "lib/bot-strategy.ts",
    "training/simulator.ts",
)


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _repository(root: Path) -> str:
    _git(root, "init")
    _git(root, "config", "user.name", "V5 Provenance Test")
    _git(root, "config", "user.email", "v5-provenance@example.invalid")
    _git(root, "config", "core.autocrlf", "false")
    for index, logical in enumerate(SOURCE_FILES):
        path = root.joinpath(*logical.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")
    _git(root, "add", "--", *SOURCE_FILES)
    _git(root, "commit", "-m", "sealed evaluator sources")
    return _git(root, "rev-parse", "HEAD").decode("ascii").strip()


class V5ProvenanceTests(unittest.TestCase):
    def test_build_binds_git_blobs_runtime_and_actual_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()
            commit = _repository(root)
            snapshot = Path(temporary) / "source.tar.zst"
            bundle = Path(temporary) / "source.bundle"
            snapshot.write_bytes(b"snapshot-bytes")
            _git(root, "bundle", "create", str(bundle), "HEAD")
            provenance = build_v5_evaluation_provenance(
                root,
                commit,
                backend="cpu",
                source_snapshot=snapshot,
                git_bundle=bundle,
                source_files=SOURCE_FILES,
            )
            self.assertEqual(provenance["source"]["sourceCommit"], commit)
            self.assertEqual(
                provenance["source"]["normalSourceSha256"],
                hashlib.sha256((root / "lib/bot-strategy.ts").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["artifacts"]["sourceSnapshotSha256"],
                hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["artifacts"]["gitBundleSha256"],
                hashlib.sha256(bundle.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["backendPolicy"], V5_EVALUATION_BACKEND_POLICY)
            self.assertEqual(provenance["runtime"]["backend"], "cpu")
            self.assertEqual(
                validate_v5_evaluation_provenance(
                    provenance,
                    repository_root=root,
                    source_snapshot=snapshot,
                    git_bundle=bundle,
                    source_files=SOURCE_FILES,
                ),
                provenance,
            )

    def test_dirty_or_different_commit_blob_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_commit = _repository(root)
            target = root / "gpu-training/v4_env.py"
            target.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dirty"):
                resolve_v5_evaluation_source_binding(
                    root, first_commit, source_files=SOURCE_FILES
                )

            _git(root, "add", "--", "gpu-training/v4_env.py")
            _git(root, "commit", "-m", "different evaluator")
            with self.assertRaisesRegex(ValueError, "bound commit blob"):
                resolve_v5_evaluation_source_binding(
                    root, first_commit, source_files=SOURCE_FILES
                )

    def test_full_commit_and_actual_archive_digest_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()
            commit = _repository(root)
            with self.assertRaisesRegex(ValueError, "full 40"):
                resolve_v5_evaluation_source_binding(
                    root, commit[:12], source_files=SOURCE_FILES
                )
            snapshot = Path(temporary) / "source.tar.zst"
            snapshot.write_bytes(b"actual")
            with self.assertRaisesRegex(ValueError, "actual file"):
                build_v5_evaluation_provenance(
                    root,
                    commit,
                    backend="cpu",
                    source_snapshot=snapshot,
                    source_snapshot_sha256="0" * 64,
                    source_files=SOURCE_FILES,
                )
            with self.assertRaisesRegex(ValueError, "without its file"):
                build_v5_evaluation_provenance(
                    root,
                    commit,
                    backend="cpu",
                    git_bundle_sha256="0" * 64,
                    source_files=SOURCE_FILES,
                )

    def test_structural_tamper_and_unavailable_cuda_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            commit = _repository(root)
            provenance = build_v5_evaluation_provenance(
                root, commit, backend="cpu", source_files=SOURCE_FILES
            )
            tampered = copy.deepcopy(provenance)
            tampered["source"]["files"]["gpu-training/v4_env.py"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "named source digest"):
                validate_v5_evaluation_provenance(
                    tampered, source_files=SOURCE_FILES
                )
            incomplete = copy.deepcopy(provenance)
            incomplete["source"]["files"].pop("training/simulator.ts")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_v5_evaluation_provenance(incomplete)
            with mock.patch("v5_provenance.torch.cuda.is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    v5_evaluation_runtime_provenance("cuda")


if __name__ == "__main__":
    unittest.main()
