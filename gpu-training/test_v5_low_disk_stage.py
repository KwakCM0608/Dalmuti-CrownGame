from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from test_v5_dataset import _fixture_arrays
from v5_dataset import (
    load_v5_index_manifest,
    publish_v5_index_manifest,
    publish_v5_shard,
)
import v5_low_disk_stage
from v5_low_disk_stage import (
    LOW_DISK_STAGE_PLAN_NAME,
    PERSISTENT_TIER,
    PROMOTION_RECEIPT_ROOT_NAME,
    SOURCE_INDEX_RECORD_NAME,
    VOLATILE_TIER,
    build_v5_low_disk_stage_plan,
    inventory_v5_training_shard,
    load_v5_low_disk_promotion_receipt,
    load_v5_low_disk_stage_plan,
    load_v5_source_index_record,
    publish_v5_low_disk_stage_plan,
    publish_v5_source_index_record,
    verify_and_publish_v5_hybrid_index,
    verify_and_promote_v5_staged_shard,
    verify_v5_hybrid_stage,
)


MIB = 1024 * 1024


def _source_corpus(root: Path) -> tuple[Path, dict[str, Path]]:
    actor, privileged = _fixture_arrays()
    shards: list[Path] = []
    by_manifest: dict[str, Path] = {}
    for index in range(2):
        shard = root / f"source-shard-{index}"
        digest = publish_v5_shard(
            shard,
            actor,
            privileged,
            metadata={"immutableShard": index},
        )
        shards.append(shard)
        by_manifest[digest] = shard
    source_index = root / "source-index"
    publish_v5_index_manifest(source_index, shards, metadata={"corpus": "test"})
    return source_index, by_manifest


def _small_plan(source_index: Path, run_namespace: str = "v5-stage-test-run-001"):
    # Each tiny shard is conservatively charged one placement quantum.
    return build_v5_low_disk_stage_plan(
        source_index,
        run_namespace=run_namespace,
        persistent_free_bytes=MIB,
        volatile_free_bytes=MIB,
        persistent_reserve_bytes=0,
        volatile_reserve_bytes=0,
        minimum_persistent_reserve_bytes=0,
        minimum_volatile_reserve_bytes=0,
    )


def _control_artifacts(
    root: Path, source_index: Path, plan: object
) -> tuple[Path, Path, Path]:
    control = root / "control"
    plan_path = control / LOW_DISK_STAGE_PLAN_NAME
    source_record = control / SOURCE_INDEX_RECORD_NAME
    receipt_root = (
        root
        / "receipt-mount"
        / plan.run_namespace  # type: ignore[attr-defined]
        / PROMOTION_RECEIPT_ROOT_NAME
    )
    publish_v5_low_disk_stage_plan(plan_path, plan)  # type: ignore[arg-type]
    publish_v5_source_index_record(source_index, source_record)
    receipt_root.mkdir(parents=True)
    return plan_path, source_record, receipt_root


def _materialize_stage(
    root: Path,
    source_index: Path,
    sources_by_manifest: dict[str, Path],
    plan: object,
) -> tuple[Path, Path, Path, Path, Path]:
    plan_path, source_record, receipt_root = _control_artifacts(
        root, source_index, plan
    )
    persistent = (
        root
        / "persistent-mount"
        / plan.run_namespace  # type: ignore[attr-defined]
        / "persistent-shards"
    )
    volatile = (
        root
        / "volatile-mount"
        / plan.run_namespace  # type: ignore[attr-defined]
        / "volatile-shards"
    )
    persistent.mkdir(parents=True)
    volatile.mkdir(parents=True)
    tier_roots = {PERSISTENT_TIER: persistent, VOLATILE_TIER: volatile}
    for record in plan.shards:  # type: ignore[attr-defined]
        tier_root = tier_roots[str(record["tier"])]
        incoming = (
            tier_root
            / f".{record['stagedName']}.incoming-transfer{record['index']:02d}"
        )
        shutil.copytree(
            sources_by_manifest[str(record["manifestSha256"])], incoming
        )
        verify_and_promote_v5_staged_shard(
            plan_path,
            shard_index=int(record["index"]),
            incoming_path=incoming,
            tier_root=tier_root,
            receipt_root=receipt_root,
        )
    return plan_path, source_record, receipt_root, persistent, volatile


class V5LowDiskStageTests(unittest.TestCase):
    def test_collection_binding_uses_complete_provenance_recomputation(self) -> None:
        players = {str(player): 1 for player in range(4, 11)}
        nonforced = {str(player): 220_000 for player in range(4, 11)}
        manifests = {"0": "a" * 64}
        verified = {
            "actualDecisionCountsByPlayerCount": players,
            "actualMatchCountsByPlayerCount": players,
            "actualNonforcedDecisionCountsByPlayerCount": nonforced,
            "completeShardIndices": [0],
            "matchCoordinatesSha256": "b" * 64,
            "matchProvenanceContract": "exact-match-provenance-test-v1",
            "shardManifestSha256s": manifests,
            "totalUniqueMatches": 7,
        }
        corpus_gate = {"passed": True}
        collection = SimpleNamespace(
            purpose="production",
            run_namespace="v5-stage-binding-run-001",
            shards=(object(),),
            behavior={"pairId": "c" * 64},
            manifest_sha256="d" * 64,
            document={
                "calibration": {"reportSha256": "e" * 64},
                "matchCounts": players,
                "policyNumericsSha256": "f" * 64,
                "sourceInventorySha256": "1" * 64,
            },
        )
        metadata = {
            "actualCorpusGate": corpus_gate,
            "actualDecisionCountsByPlayerCount": players,
            "actualMatchCountsByPlayerCount": players,
            "actualNonforcedDecisionCountsByPlayerCount": nonforced,
            "behavior": dict(collection.behavior),
            "calibrationReportSha256": "e" * 64,
            "collectionPlanManifestSha256": "d" * 64,
            "completeShardIndices": [0],
            "matchCoordinatesSha256": "b" * 64,
            "matchProvenanceContract": "exact-match-provenance-test-v1",
            "plannedMatchCountsByPlayerCount": players,
            "policyNumericsSha256": "f" * 64,
            "shardManifestSha256s": manifests,
            "sourceInventorySha256": "1" * 64,
            "totalUniqueMatches": 7,
        }
        source = v5_low_disk_stage.V5SourceIndexRecord(
            {
                "counts": {"decisions": 7, "matches": 7, "shards": 1},
                "metadata": metadata,
                "playerCounts": list(range(4, 11)),
            },
            "2" * 64,
        )
        with mock.patch(
            "v5_collection_plan.load_collection_plan", return_value=collection
        ), mock.patch(
            "v5_collection_plan.verify_planned_collection_corpus",
            return_value=verified,
        ) as recompute, mock.patch(
            "v5_collection_plan.validate_actual_nonforced_corpus",
            return_value=corpus_gate,
        ):
            result = v5_low_disk_stage._verify_collection_plan_stage_binding(
                "collection-plan",
                SimpleNamespace(run_namespace=collection.run_namespace),
                source,
                (Path("staged-shard"),),
                (object(),),
            )
        self.assertEqual(result, (players, players, nonforced, corpus_gate))
        recompute.assert_called_once()

    def test_capacity_plan_uses_both_tiers_and_preserves_reserves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, _ = _source_corpus(Path(temporary))
            plan = _small_plan(source)
            tiers = plan.document["tiers"]
            self.assertEqual(tiers[PERSISTENT_TIER]["assignedBytes"], MIB)
            self.assertEqual(tiers[VOLATILE_TIER]["assignedBytes"], MIB)
            self.assertEqual(
                {record["tier"] for record in plan.shards},
                {PERSISTENT_TIER, VOLATILE_TIER},
            )
            self.assertEqual(plan.document["source"]["counts"]["shards"], 2)
            self.assertEqual(
                plan.document["source"]["nonforcedDecisions"], 14
            )

    def test_capacity_and_minimum_reserve_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, _ = _source_corpus(Path(temporary))
            with self.assertRaisesRegex(ValueError, "combined reserved"):
                build_v5_low_disk_stage_plan(
                    source,
                    run_namespace="v5-stage-overflow-run-001",
                    persistent_free_bytes=MIB - 1,
                    volatile_free_bytes=MIB - 1,
                    persistent_reserve_bytes=0,
                    volatile_reserve_bytes=0,
                    minimum_persistent_reserve_bytes=0,
                    minimum_volatile_reserve_bytes=0,
                )
            with self.assertRaisesRegex(ValueError, "below its fail-closed minimum"):
                build_v5_low_disk_stage_plan(
                    source,
                    run_namespace="v5-stage-reserve-run-001",
                    persistent_free_bytes=10 * MIB,
                    volatile_free_bytes=10 * MIB,
                    persistent_reserve_bytes=MIB,
                    volatile_reserve_bytes=0,
                    minimum_persistent_reserve_bytes=2 * MIB,
                    minimum_volatile_reserve_bytes=0,
                )

    def test_plan_publish_load_is_canonical_exclusive_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = _source_corpus(root)
            plan = _small_plan(source)
            target = root / "stage-plan"
            digest = publish_v5_low_disk_stage_plan(target, plan)
            self.assertEqual(digest, plan.manifest_sha256)
            self.assertEqual(load_v5_low_disk_stage_plan(target), plan)
            self.assertEqual(publish_v5_low_disk_stage_plan(target, plan), digest)
            raw = (target / "plan.json").read_bytes()
            value = json.loads(raw)
            value["runNamespace"] = "v5-stage-tampered-run-001"
            (target / "plan.json").write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "sidecar"):
                load_v5_low_disk_stage_plan(target)

    def test_lock_free_publication_survives_every_crash_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "manifest.json": b'{"format":"test"}\n',
                "manifest.json.sha256": b"test-sidecar\n",
            }
            original_write = v5_low_disk_stage._write_fsynced_file

            primary = root / "primary-crash"

            def fail_primary(path: Path, data: bytes) -> None:
                path.write_bytes(data[:1])
                raise RuntimeError("primary write crash")

            with mock.patch.object(
                v5_low_disk_stage,
                "_write_fsynced_file",
                side_effect=fail_primary,
            ), self.assertRaisesRegex(RuntimeError, "primary write crash"):
                v5_low_disk_stage._exclusive_publish_directory(primary, files)
            self.assertFalse(primary.exists())
            self.assertFalse(tuple(root.glob(".primary-crash.staging-*")))
            v5_low_disk_stage._exclusive_publish_directory(primary, files)

            sidecar = root / "sidecar-crash"
            writes = 0

            def fail_sidecar(path: Path, data: bytes) -> None:
                nonlocal writes
                writes += 1
                if writes == 2:
                    path.write_bytes(data[:1])
                    raise RuntimeError("sidecar write crash")
                original_write(path, data)

            with mock.patch.object(
                v5_low_disk_stage,
                "_write_fsynced_file",
                side_effect=fail_sidecar,
            ), self.assertRaisesRegex(RuntimeError, "sidecar write crash"):
                v5_low_disk_stage._exclusive_publish_directory(sidecar, files)
            self.assertFalse(sidecar.exists())
            self.assertFalse(tuple(root.glob(".sidecar-crash.staging-*")))
            v5_low_disk_stage._exclusive_publish_directory(sidecar, files)

            published = root / "post-rename-crash"
            original_rename = v5_low_disk_stage._rename_directory_noreplace

            def crash_after_rename(source: Path, target: Path) -> None:
                original_rename(source, target)
                raise RuntimeError("post-rename crash")

            with mock.patch.object(
                v5_low_disk_stage,
                "_rename_directory_noreplace",
                side_effect=crash_after_rename,
            ):
                v5_low_disk_stage._exclusive_publish_directory(published, files)
            v5_low_disk_stage._exclusive_publish_directory(published, files)
            self.assertEqual(
                {path.name: path.read_bytes() for path in published.iterdir()}, files
            )

            (published / "manifest.json").write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "differs from expected"):
                v5_low_disk_stage._exclusive_publish_directory(published, files)

    def test_source_index_record_preserves_raw_identity_without_resolving_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = _source_corpus(root / "canonical")
            target = root / "control" / SOURCE_INDEX_RECORD_NAME
            digest = publish_v5_source_index_record(source, target)
            record = load_v5_source_index_record(target)
            self.assertEqual(record.manifest_sha256, digest)
            self.assertEqual(publish_v5_source_index_record(source, target), digest)
            self.assertEqual(
                record.document,
                json.loads((source / "manifest.json").read_text(encoding="utf-8")),
            )
            # The canonical source shards can disappear from this host; loading
            # the small evidence record must still never follow relative paths.
            shutil.rmtree(root / "canonical")
            self.assertEqual(load_v5_source_index_record(target), record)
            raw = (target / "manifest.json").read_bytes()
            value = json.loads(raw)
            value["metadata"]["corpus"] = "foreign"
            (target / "manifest.json").write_bytes(
                v5_low_disk_stage._canonical_json_bytes(value)
            )
            with self.assertRaisesRegex(ValueError, "sidecar"):
                load_v5_source_index_record(target)

    def test_stage_plan_rejects_a_different_valid_source_index_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path = root / "control" / LOW_DISK_STAGE_PLAN_NAME
            publish_v5_low_disk_stage_plan(plan_path, plan)
            other_source, _ = _source_corpus(root / "other-canonical")
            other_record = root / "other-control" / SOURCE_INDEX_RECORD_NAME
            publish_v5_source_index_record(other_source, other_record)
            foreign = json.loads(
                (other_record / "manifest.json").read_text(encoding="utf-8")
            )
            foreign["metadata"]["corpus"] = "foreign"
            foreign_raw = v5_low_disk_stage._canonical_json_bytes(foreign)
            foreign_sha = v5_low_disk_stage._sha256_bytes(foreign_raw)
            (other_record / "manifest.json").write_bytes(foreign_raw)
            (other_record / "manifest.json.sha256").write_bytes(
                f"{foreign_sha}  manifest.json\n".encode("ascii")
            )
            receipts = (
                root
                / "r"
                / plan.run_namespace
                / PROMOTION_RECEIPT_ROOT_NAME
            )
            persistent = root / "p" / plan.run_namespace / "persistent-shards"
            volatile = root / "v" / plan.run_namespace / "volatile-shards"
            with self.assertRaisesRegex(ValueError, "differs from the low-disk stage plan"):
                verify_v5_hybrid_stage(
                    plan_path,
                    persistent_root=persistent,
                    volatile_root=volatile,
                    source_index_record=other_record,
                    promotion_receipt_root=receipts,
                )

    def test_cross_filesystem_layout_verifies_and_publishes_standard_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            (
                plan_path,
                source_record,
                receipt_root,
                persistent,
                volatile,
            ) = _materialize_stage(root, source, sources_by_manifest, plan)
            output = root / "control" / "training-index"

            def fake_device(path: Path) -> int:
                return 1 if "persistent-mount" in path.parts else 2

            with mock.patch.object(
                v5_low_disk_stage, "_device_id", side_effect=fake_device
            ):
                digest = verify_and_publish_v5_hybrid_index(
                    plan_path,
                    persistent_root=persistent,
                    volatile_root=volatile,
                    source_index_record=source_record,
                    promotion_receipt_root=receipt_root,
                    output_index=output,
                )
            self.assertEqual(len(digest), 64)
            index = load_v5_index_manifest(output)
            try:
                self.assertEqual(index.decision_count, 14)
                self.assertEqual(index.match_count, 14)
                self.assertEqual(len(index.shard_paths), 2)
                metadata = index.manifest["metadata"]
                self.assertEqual(
                    metadata["lowDiskStagePlanSha256"], plan.manifest_sha256
                )
                self.assertEqual(
                    metadata["lowDiskStageContract"],
                    v5_low_disk_stage.V5_LOW_DISK_STAGE_INDEX_CONTRACT,
                )
                self.assertTrue(
                    any("volatile-mount" in path.parts for path in index.shard_paths)
                )
            finally:
                index.close()
            with mock.patch.object(
                v5_low_disk_stage, "_device_id", side_effect=fake_device
            ):
                verified = verify_v5_hybrid_stage(
                    plan_path,
                    persistent_root=persistent,
                    volatile_root=volatile,
                    source_index_record=source_record,
                    promotion_receipt_root=receipt_root,
                    hybrid_index=output,
                )
            self.assertEqual(verified.hybrid_index_manifest_sha256, digest)
            self.assertEqual(len(verified.shard_paths), 2)

    def test_crash_after_atomic_rename_issues_and_reuses_exact_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-crashwin01"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            with mock.patch.object(
                v5_low_disk_stage,
                "_publish_or_reuse_promotion_receipt",
                side_effect=RuntimeError("simulated crash after rename"),
            ), self.assertRaisesRegex(RuntimeError, "simulated crash"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            target = tier_root / str(record["stagedName"])
            self.assertTrue(target.is_dir())
            self.assertFalse(incoming.exists())
            receipt_path = receipt_root / str(record["stagedName"])
            self.assertFalse(receipt_path.exists())

            recovered = verify_and_promote_v5_staged_shard(
                plan_path,
                shard_index=int(record["index"]),
                incoming_path=incoming,
                tier_root=tier_root,
                receipt_root=receipt_root,
            )
            self.assertEqual(recovered, target)
            first_receipt, first_sha = load_v5_low_disk_promotion_receipt(
                receipt_path
            )
            reused = verify_and_promote_v5_staged_shard(
                plan_path,
                shard_index=int(record["index"]),
                incoming_path=incoming,
                tier_root=tier_root,
                receipt_root=receipt_root,
            )
            second_receipt, second_sha = load_v5_low_disk_promotion_receipt(
                receipt_path
            )
            self.assertEqual(reused, target)
            self.assertEqual((second_receipt, second_sha), (first_receipt, first_sha))

    def test_mid_receipt_sidecar_crash_recovers_after_shard_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-receipt001"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            original_write = v5_low_disk_stage._write_fsynced_file
            writes = 0

            def fail_receipt_sidecar(path: Path, data: bytes) -> None:
                nonlocal writes
                writes += 1
                if writes == 2:
                    path.write_bytes(data[:1])
                    raise RuntimeError("receipt sidecar crash")
                original_write(path, data)

            with mock.patch.object(
                v5_low_disk_stage,
                "_write_fsynced_file",
                side_effect=fail_receipt_sidecar,
            ), self.assertRaisesRegex(RuntimeError, "receipt sidecar crash"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            target = tier_root / str(record["stagedName"])
            self.assertTrue(target.is_dir())
            self.assertFalse(incoming.exists())
            self.assertFalse(
                (receipt_root / str(record["stagedName"])).exists()
            )
            self.assertFalse(tuple(receipt_root.glob(".*.staging-*")))

            recovered = verify_and_promote_v5_staged_shard(
                plan_path,
                shard_index=int(record["index"]),
                incoming_path=incoming,
                tier_root=tier_root,
                receipt_root=receipt_root,
            )
            self.assertEqual(recovered, target)
            load_v5_low_disk_promotion_receipt(
                receipt_root / str(record["stagedName"])
            )

    def test_torn_prelink_lock_temp_is_ignored_then_retired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-lockcrash01"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            lock = tier_root / f".{record['stagedName']}.promote.lock"
            # This is the only artifact a power loss during the fsynced-temp
            # write can expose.  The canonical lock name was never linked.
            pending = tier_root / (
                f"{lock.name}.pending-{'a' * 32}"
            )
            pending.write_bytes(b'{"format":"torn')

            target = verify_and_promote_v5_staged_shard(
                plan_path,
                shard_index=int(record["index"]),
                incoming_path=incoming,
                tier_root=tier_root,
                receipt_root=receipt_root,
            )
            self.assertTrue(target.is_dir())
            self.assertFalse(pending.exists())
            self.assertFalse(lock.exists())
            load_v5_low_disk_promotion_receipt(
                receipt_root / str(record["stagedName"])
            )

    def test_postlink_retirement_error_releases_only_exact_owned_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-retireerr01"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            lock = tier_root / f".{record['stagedName']}.promote.lock"
            foreign_pending = tier_root / f"{lock.name}.pending-{'b' * 32}"
            foreign_pending.mkdir()

            with self.assertRaisesRegex(ValueError, "pending artifact is foreign"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            self.assertFalse(lock.exists())
            self.assertTrue(foreign_pending.is_dir())
            self.assertTrue(incoming.is_dir())

    def test_postlink_foreign_replacement_is_retained_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-replace001"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            lock = tier_root / f".{record['stagedName']}.promote.lock"

            def replace_owned_lock(stage_root: Path, owned_lock: Path) -> None:
                raw = owned_lock.read_bytes()
                owned_lock.unlink()
                replacement = stage_root / ".foreign-lock-replacement"
                replacement.write_bytes(raw)
                replacement.replace(owned_lock)
                raise RuntimeError("simulated foreign replacement")

            with mock.patch.object(
                v5_low_disk_stage,
                "_retire_promotion_lock_pending_files",
                side_effect=replace_owned_lock,
            ), self.assertRaisesRegex(RuntimeError, "foreign replacement"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            self.assertTrue(lock.is_file())
            self.assertTrue(incoming.is_dir())
            self.assertFalse(
                (receipt_root / str(record["stagedName"])).exists()
            )

    def test_active_lock_and_foreign_or_corrupt_targets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-active001"
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], incoming
            )
            lock = tier_root / f".{record['stagedName']}.promote.lock"
            lock.write_bytes(
                v5_low_disk_stage._canonical_json_bytes(
                    v5_low_disk_stage._promotion_lock_document(plan, record)
                )
            )
            with self.assertRaisesRegex(BlockingIOError, "still active"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            self.assertTrue(incoming.is_dir())
            lock.unlink()

            # A valid but different shard at the canonical final name is
            # foreign even when every internal checksum passes.
            target = tier_root / str(record["stagedName"])
            foreign_record = plan.shards[1]
            shutil.copytree(
                sources_by_manifest[str(foreign_record["manifestSha256"])], target
            )
            with self.assertRaisesRegex(ValueError, "staged shard verification drifted"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            shutil.rmtree(target)
            shutil.copytree(
                sources_by_manifest[str(record["manifestSha256"])], target
            )
            (target / "foreign.bin").write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "file inventory"):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )

    def test_same_filesystem_extra_file_and_byte_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, source_record, receipt_root = _control_artifacts(
                root, source, plan
            )
            persistent = root / "one-device" / plan.run_namespace / "persistent-shards"
            volatile = root / "another-path" / plan.run_namespace / "volatile-shards"
            persistent.mkdir(parents=True)
            volatile.mkdir(parents=True)
            tier_roots = {PERSISTENT_TIER: persistent, VOLATILE_TIER: volatile}
            for record in plan.shards:
                shutil.copytree(
                    sources_by_manifest[str(record["manifestSha256"])],
                    tier_roots[str(record["tier"])] / str(record["stagedName"]),
                )
            with self.assertRaisesRegex(ValueError, "different filesystems"):
                verify_and_publish_v5_hybrid_index(
                    plan_path,
                    persistent_root=persistent,
                    volatile_root=volatile,
                    source_index_record=source_record,
                    promotion_receipt_root=receipt_root,
                    output_index=root / "control" / "index-same-device",
                )

            target_record = plan.shards[0]
            target = (
                tier_roots[str(target_record["tier"])]
                / str(target_record["stagedName"])
            )
            (target / "unexpected.bin").write_bytes(b"not checksum bound")
            with self.assertRaisesRegex(ValueError, "file inventory"):
                inventory_v5_training_shard(target)
            (target / "unexpected.bin").unlink()
            action = target / "actor" / "actions.npy"
            with action.open("r+b") as handle:
                handle.seek(-1, 2)
                previous = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([previous[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                inventory_v5_training_shard(target)

    def test_pure_verifier_rejects_tier_receipt_coverage_and_reserve_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = build_v5_low_disk_stage_plan(
                source,
                run_namespace="v5-stage-attacks-run-001",
                persistent_free_bytes=MIB + 1,
                volatile_free_bytes=MIB + 1,
                persistent_reserve_bytes=1,
                volatile_reserve_bytes=1,
                minimum_persistent_reserve_bytes=0,
                minimum_volatile_reserve_bytes=0,
            )
            (
                plan_path,
                source_record,
                receipt_root,
                persistent,
                volatile,
            ) = _materialize_stage(root, source, sources_by_manifest, plan)

            def fake_device(path: Path) -> int:
                return 1 if "persistent-mount" in path.parts else 2

            def verify() -> object:
                with mock.patch.object(
                    v5_low_disk_stage, "_device_id", side_effect=fake_device
                ):
                    return verify_v5_hybrid_stage(
                        plan_path,
                        persistent_root=persistent,
                        volatile_root=volatile,
                        source_index_record=source_record,
                        promotion_receipt_root=receipt_root,
                    )

            self.assertEqual(len(verify().shard_paths), 2)  # type: ignore[attr-defined]

            plan_extra = plan_path / "unexpected-control-file"
            plan_extra.write_bytes(b"foreign")
            with self.assertRaisesRegex(ValueError, "plan inventory"):
                verify()
            plan_extra.unlink()

            extra = persistent / "unexpected-child"
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, "directory coverage drifted"):
                verify()
            extra.rmdir()

            receipt = next(receipt_root.iterdir())
            held_receipt = receipt_root.parent / f"held-{receipt.name}"
            receipt.rename(held_receipt)
            with self.assertRaisesRegex(ValueError, "receipt directory coverage"):
                verify()
            held_receipt.rename(receipt)

            real_disk_usage = shutil.disk_usage

            def exhausted(path: Path) -> object:
                usage = real_disk_usage(path)
                if Path(path).resolve() == persistent.resolve():
                    return mock.Mock(total=usage.total, used=usage.used, free=0)
                return usage

            with mock.patch.object(
                v5_low_disk_stage.shutil, "disk_usage", side_effect=exhausted
            ), self.assertRaisesRegex(ValueError, "reserved headroom"):
                verify()

            receipt_document, _ = load_v5_low_disk_promotion_receipt(receipt)
            foreign = json.loads(json.dumps(receipt_document))
            foreign["runNamespace"] = "v5-stage-foreign-run-001"
            raw = v5_low_disk_stage._canonical_json_bytes(foreign)
            digest = v5_low_disk_stage._sha256_bytes(raw)
            (receipt / "manifest.json").write_bytes(raw)
            (receipt / "manifest.json.sha256").write_bytes(
                f"{digest}  manifest.json\n".encode("ascii")
            )
            with self.assertRaisesRegex(ValueError, "differs from staged bytes"):
                verify()

    def test_output_index_cannot_pollute_a_shard_only_tier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, source_record, receipt_root = _control_artifacts(
                root, source, plan
            )
            persistent = root / "p" / plan.run_namespace / "persistent-shards"
            volatile = root / "v" / plan.run_namespace / "volatile-shards"
            persistent.mkdir(parents=True)
            volatile.mkdir(parents=True)
            tiers = {PERSISTENT_TIER: persistent, VOLATILE_TIER: volatile}
            for record in plan.shards:
                shutil.copytree(
                    sources_by_manifest[str(record["manifestSha256"])],
                    tiers[str(record["tier"])] / str(record["stagedName"]),
                )
            with mock.patch.object(
                v5_low_disk_stage,
                "_device_id",
                side_effect=lambda path: 1 if "p" in path.parts else 2,
            ), self.assertRaisesRegex(ValueError, "outside both"):
                verify_and_publish_v5_hybrid_index(
                    plan_path,
                    persistent_root=persistent,
                    volatile_root=volatile,
                    source_index_record=source_record,
                    promotion_receipt_root=receipt_root,
                    output_index=persistent / "bad-index",
                )

    def test_no_replace_promotion_race_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, sources_by_manifest = _source_corpus(root / "canonical")
            plan = _small_plan(source)
            plan_path, _, receipt_root = _control_artifacts(root, source, plan)
            record = plan.shards[0]
            tier = str(record["tier"])
            tier_root = root / "mount" / plan.run_namespace / f"{tier}-shards"
            tier_root.mkdir(parents=True)
            incoming = tier_root / f".{record['stagedName']}.incoming-racetest01"
            source_shard = sources_by_manifest[str(record["manifestSha256"])]
            shutil.copytree(source_shard, incoming)

            original = v5_low_disk_stage._rename_directory_noreplace

            def create_competing_target(source_path: Path, target_path: Path) -> None:
                target_path.mkdir()
                original(source_path, target_path)

            with mock.patch.object(
                v5_low_disk_stage,
                "_rename_directory_noreplace",
                side_effect=create_competing_target,
            ), self.assertRaises(OSError):
                verify_and_promote_v5_staged_shard(
                    plan_path,
                    shard_index=int(record["index"]),
                    incoming_path=incoming,
                    tier_root=tier_root,
                    receipt_root=receipt_root,
                )
            self.assertTrue(incoming.is_dir())
            self.assertTrue((tier_root / str(record["stagedName"])).is_dir())
            self.assertFalse(
                (tier_root / f".{record['stagedName']}.promote.lock").exists()
            )

    def test_shard_root_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_corpus(root / "canonical")
            link = root / "shard-link"
            original = Path.is_symlink

            def symlink_probe(path: Path) -> bool:
                return path == link or original(path)

            with mock.patch.object(
                Path, "is_symlink", autospec=True, side_effect=symlink_probe
            ), self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                inventory_v5_training_shard(link)


if __name__ == "__main__":
    unittest.main()
