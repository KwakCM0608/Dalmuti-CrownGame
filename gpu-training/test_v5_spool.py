from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_v5_collection_plan import _plan
from v5_collection_plan import (
    expected_planned_shard_metadata,
    publish_collection_plan,
    verify_planned_shard,
)
from v4_env import ACTION_COUNT
from v5_collect_mappo import (
    V5_MAPPO_COLLECTION_CONTRACT,
    V5_MAPPO_REWARD_CONTRACT,
    V5MAPPOCollectionConfig,
    collect_v5_mappo,
    v5_collection_array_partitions,
)
from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_dataset import publish_v5_shard
from v5_model import V5_POLICY_NUMERICS_SHA256
import v5_spool
from v5_spool import (
    CANONICAL_ROOT_NAME,
    INCOMING_SPOOL_ROOT_NAME,
    RAW_ROOT_NAME,
    RECEIPT_ROOT_NAME,
    SPOOL_ROOT_NAME,
    export_v5_planned_shard_spool,
    import_v5_planned_shard_spool,
    load_v5_spool_bundle,
    load_v5_verified_copy_receipt,
    retire_v5_verified_raw_shard,
    retire_v5_verified_spool_bundle,
    validate_v5_archive_member_names,
)


@lru_cache(maxsize=None)
def _collected_single_match_arrays(
    player_count: int = 4, match_start: int = 0
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray]
]:
    def actor_batch(observations, normal_actions):
        logits = np.full((len(observations), ACTION_COUNT), -1000.0, np.float64)
        for row, (observation, normal_action) in enumerate(
            zip(observations, normal_actions, strict=True)
        ):
            legal = np.asarray(observation.legal_mask, dtype=np.bool_)
            logits[row, legal] = -2.0
            logits[row, normal_action] = 0.0
        return logits

    def critic_batch(states):
        return np.zeros(len(states), dtype=np.float64)

    collection = collect_v5_mappo(
        actor_batch,
        critic_batch,
        V5MAPPOCollectionConfig(
            run_namespace="v5-spool-s990000001-run-001",
            seed_base=990_000_001,
            match_counts=((player_count, 1),),
            match_start=match_start,
        ),
    )
    return v5_collection_array_partitions(collection)


def _single_match_arrays(
    player_count: int = 4, match_start: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    actor, privileged = _collected_single_match_arrays(player_count, match_start)
    return (
        {name: value.copy() for name, value in actor.items()},
        {name: value.copy() for name, value in privileged.items()},
    )


def _planned_fixture(root: Path, shard_index: int = 0):
    run_namespace = "v5-spool-s990000001-run-001"
    plan = _plan(
        run_namespace=run_namespace,
        seed_base=990_000_001,
        total_matches=14,
    )
    plan_path = root / "control" / "collection-plan"
    publish_collection_plan(plan_path, plan)
    shard = plan.shards[shard_index]
    raw_root = root / "remote" / run_namespace / RAW_ROOT_NAME
    raw_root.mkdir(parents=True)
    actor, privileged = _single_match_arrays(
        shard.player_count, shard.match_start
    )
    behavior = plan.behavior
    metadata = {
        **expected_planned_shard_metadata(plan, shard),
        "behaviorActorManifestSha256": behavior["actorManifestSha256"],
        "behaviorActorSha256": behavior["actorSha256"],
        "behaviorCriticSha256": behavior["criticSha256"],
        "behaviorModelPairId": behavior["pairId"],
        "behaviorModelPairManifestSha256": behavior["pairManifestSha256"],
        "collectionContract": V5_MAPPO_COLLECTION_CONTRACT,
        "publicContractSha256": V5_PUBLIC_CONTRACT_SHA256,
        "policyNumericsSha256": V5_POLICY_NUMERICS_SHA256,
        "rewardContract": V5_MAPPO_REWARD_CONTRACT,
        "runNamespace": plan.run_namespace,
        "seedBase": plan.seed_base,
        "matchCounts": {str(shard.player_count): shard.match_count},
        "matchStart": shard.match_start,
        "matchShardCount": 1,
        "matchShardIndex": 0,
    }
    source = raw_root / shard.name
    publish_v5_shard(source, actor, privileged, metadata=metadata)
    verify_planned_shard(plan, shard, source)
    return plan, plan_path, shard, raw_root, source


def _export_and_copy(root: Path):
    plan, plan_path, shard, raw_root, source = _planned_fixture(root)
    spool_root = root / "remote" / plan.run_namespace / SPOOL_ROOT_NAME
    bundle = export_v5_planned_shard_spool(
        plan_path,
        shard_index=shard.index,
        raw_root=raw_root,
        spool_root=spool_root,
        minimum_free_after_export_bytes=0,
    )
    incoming_root = root / "local" / plan.run_namespace / INCOMING_SPOOL_ROOT_NAME
    incoming_root.mkdir(parents=True)
    incoming = incoming_root / bundle.name
    shutil.copytree(bundle, incoming)
    return plan, plan_path, shard, raw_root, source, spool_root, bundle, incoming


class V5SpoolTests(unittest.TestCase):
    def test_end_to_end_export_import_receipt_then_exact_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                raw_root,
                source,
                spool_root,
                bundle,
                incoming,
            ) = _export_and_copy(root)
            self.assertTrue(source.is_dir(), "export must not delete raw source")
            spool, spool_sha = load_v5_spool_bundle(bundle)
            self.assertEqual(spool["source"]["shardIndex"], shard.index)
            self.assertEqual(len(spool_sha), 64)

            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            local_receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            receipt = import_v5_planned_shard_spool(
                plan_path,
                shard_index=shard.index,
                bundle_path=incoming,
                canonical_root=canonical,
                receipt_root=local_receipts,
                minimum_free_after_import_bytes=0,
            )
            receipt_document, receipt_sha = load_v5_verified_copy_receipt(receipt)
            self.assertTrue(receipt_document["copyVerified"])
            self.assertEqual(len(receipt_sha), 64)
            verify_planned_shard(plan, shard, canonical / shard.name)

            # Re-import is a verified crash/retry recovery, not an overwrite.
            self.assertEqual(
                import_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    bundle_path=incoming,
                    canonical_root=canonical,
                    receipt_root=local_receipts,
                    minimum_free_after_import_bytes=0,
                ),
                receipt,
            )

            remote_receipts = (
                root / "remote" / plan.run_namespace / RECEIPT_ROOT_NAME
            )
            remote_receipts.mkdir(parents=True)
            remote_receipt = remote_receipts / receipt.name
            shutil.copytree(receipt, remote_receipt)
            with self.assertRaisesRegex(ValueError, "raw V5 shard"):
                retire_v5_verified_spool_bundle(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    spool_root=spool_root,
                    bundle_path=bundle,
                    receipt_path=remote_receipt,
                )
            retired_raw = retire_v5_verified_raw_shard(
                plan_path,
                shard_index=shard.index,
                raw_root=raw_root,
                bundle_path=bundle,
                receipt_path=remote_receipt,
            )
            self.assertEqual(retired_raw, source)
            self.assertFalse(source.exists())
            self.assertTrue(canonical.joinpath(shard.name).is_dir())
            retired_spool = retire_v5_verified_spool_bundle(
                plan_path,
                shard_index=shard.index,
                raw_root=raw_root,
                spool_root=spool_root,
                bundle_path=bundle,
                receipt_path=remote_receipt,
            )
            self.assertEqual(retired_spool, bundle)
            self.assertFalse(bundle.exists())
            self.assertTrue(remote_receipt.is_dir())

    def test_archive_or_receipt_tamper_blocks_import_and_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                raw_root,
                source,
                spool_root,
                bundle,
                incoming,
            ) = _export_and_copy(root)
            archive = incoming / "shard.tar.zst"
            with archive.open("r+b") as handle:
                handle.seek(-1, 2)
                previous = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([previous[0] ^ 1]))
            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            with self.assertRaisesRegex(ValueError, "checksum"):
                import_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    bundle_path=incoming,
                    canonical_root=canonical,
                    receipt_root=receipts,
                    minimum_free_after_import_bytes=0,
                )
            self.assertTrue(source.is_dir())
            self.assertFalse(canonical.joinpath(shard.name).exists())

            missing_receipt = (
                root
                / "remote"
                / plan.run_namespace
                / RECEIPT_ROOT_NAME
                / "copy-missing"
            )
            with self.assertRaises((FileNotFoundError, ValueError)):
                retire_v5_verified_raw_shard(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    bundle_path=bundle,
                    receipt_path=missing_receipt,
                )
            self.assertTrue(source.is_dir())
            self.assertTrue(bundle.is_dir())

    def test_existing_bundle_is_immutable_and_wrong_shard_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                raw_root,
                source,
                spool_root,
                bundle,
                incoming,
            ) = _export_and_copy(root)
            with self.assertRaises(FileExistsError):
                export_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    spool_root=spool_root,
                    minimum_free_after_export_bytes=0,
                )
            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            local_receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            receipt = import_v5_planned_shard_spool(
                plan_path,
                shard_index=shard.index,
                bundle_path=incoming,
                canonical_root=canonical,
                receipt_root=local_receipts,
                minimum_free_after_import_bytes=0,
            )
            remote_receipts = root / "remote" / plan.run_namespace / RECEIPT_ROOT_NAME
            remote_receipts.mkdir(parents=True)
            remote_receipt = remote_receipts / receipt.name
            shutil.copytree(receipt, remote_receipt)
            with self.assertRaisesRegex(ValueError, "planned shard|shard_index|differs"):
                retire_v5_verified_raw_shard(
                    plan_path,
                    shard_index=1,
                    raw_root=raw_root,
                    bundle_path=bundle,
                    receipt_path=remote_receipt,
                )
            self.assertTrue(source.is_dir())

    def test_member_path_traversal_and_alternate_separators_fail_closed(self) -> None:
        unsafe = (
            ["planned/", "planned/../escape"],
            ["planned/", "/absolute"],
            ["planned/", "planned\\escape"],
            ["planned/", "planned/file", "planned/file"],
            ["planned/", "other/file"],
        )
        for names in unsafe:
            with self.subTest(names=names), self.assertRaises(ValueError):
                validate_v5_archive_member_names(
                    names, expected_top_directory="planned"
                )
        validate_v5_archive_member_names(
            ["planned/", "planned/actor/", "planned/actor/actions.npy"],
            expected_top_directory="planned",
        )

    def test_spool_space_preflight_leaves_raw_and_no_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path, shard, raw_root, source = _planned_fixture(root)
            spool_root = root / "remote" / plan.run_namespace / SPOOL_ROOT_NAME
            with self.assertRaisesRegex(ValueError, "insufficient spool space"):
                export_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    spool_root=spool_root,
                    minimum_free_after_export_bytes=10**18,
                )
            self.assertTrue(source.is_dir())
            self.assertEqual(
                [path for path in spool_root.iterdir() if path.name.startswith("spool-")],
                [],
            )
            reservations = spool_root / ".capacity-reservations"
            self.assertTrue(reservations.is_dir())
            self.assertEqual(list(reservations.iterdir()), [])

    def test_concurrent_capacity_reservation_prevents_aggregate_overcommit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, plan_path, shard, raw_root, source = _planned_fixture(root)
            spool_root = root / "remote" / plan.run_namespace / SPOOL_ROOT_NAME
            reservations = spool_root / ".capacity-reservations"
            reservations.mkdir(parents=True)
            token = "a" * 32
            record = {
                "chargeBytes": 10**18,
                "pid": 9999,
                "token": token,
            }
            (reservations / f"{token}.json").write_bytes(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
            )
            with self.assertRaisesRegex(ValueError, "active reservations"):
                export_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    spool_root=spool_root,
                    minimum_free_after_export_bytes=0,
                )
            self.assertTrue(source.is_dir())
            self.assertEqual(
                [path for path in spool_root.iterdir() if path.name.startswith("spool-")],
                [],
            )

    def test_external_archive_swap_after_private_copy_cannot_change_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                _,
                _,
                _,
                _,
                incoming,
            ) = _export_and_copy(root)
            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            original_copy = v5_spool._copy_verified_archive

            def copy_then_corrupt_external(
                source_path: Path,
                destination_path: Path,
                *,
                expected_bytes: int,
                expected_sha256: str,
            ) -> None:
                original_copy(
                    source_path,
                    destination_path,
                    expected_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
                # The externally writable transfer path changes after the
                # private fsync+hash. Extraction must use only the private inode.
                with source_path.open("r+b") as handle:
                    handle.seek(-1, 2)
                    previous = handle.read(1)
                    handle.seek(-1, 2)
                    handle.write(bytes([previous[0] ^ 1]))

            extracted_paths: list[Path] = []
            original_extract = v5_spool._extract_archive

            def record_private_extract(archive: Path, staging: Path) -> None:
                extracted_paths.append(archive)
                self.assertNotEqual(archive, incoming / "shard.tar.zst")
                self.assertTrue(archive.name == "a.tar.zst")
                original_extract(archive, staging)

            with mock.patch.object(
                v5_spool,
                "_copy_verified_archive",
                side_effect=copy_then_corrupt_external,
            ), mock.patch.object(
                v5_spool, "_extract_archive", side_effect=record_private_extract
            ):
                receipt = import_v5_planned_shard_spool(
                    plan_path,
                    shard_index=shard.index,
                    bundle_path=incoming,
                    canonical_root=canonical,
                    receipt_root=receipts,
                    minimum_free_after_import_bytes=0,
                )
            self.assertEqual(len(extracted_paths), 1)
            self.assertTrue(receipt.is_dir())
            verify_planned_shard(plan, shard, canonical / shard.name)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_v5_spool_bundle(incoming)

    def test_tampered_imported_count_blocks_raw_retirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                raw_root,
                source,
                spool_root,
                bundle,
                incoming,
            ) = _export_and_copy(root)
            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            local_receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            receipt = import_v5_planned_shard_spool(
                plan_path,
                shard_index=shard.index,
                bundle_path=incoming,
                canonical_root=canonical,
                receipt_root=local_receipts,
                minimum_free_after_import_bytes=0,
            )
            remote_root = root / "remote" / plan.run_namespace / RECEIPT_ROOT_NAME
            remote_root.mkdir(parents=True)
            remote_receipt = remote_root / receipt.name
            shutil.copytree(receipt, remote_receipt)
            document = json.loads((remote_receipt / "receipt.json").read_text("ascii"))
            document["imported"]["decisionCount"] += 1
            raw = (
                json.dumps(
                    document,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
            )
            digest = hashlib.sha256(raw).hexdigest()
            (remote_receipt / "receipt.json").write_bytes(raw)
            (remote_receipt / "receipt.json.sha256").write_bytes(
                f"{digest}  receipt.json\n".encode("ascii")
            )
            # The tampered receipt is internally canonical/checksummed; the
            # cross-artifact count binding must still reject it.
            load_v5_verified_copy_receipt(remote_receipt)
            with self.assertRaisesRegex(ValueError, "decisionCount"):
                retire_v5_verified_raw_shard(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    bundle_path=bundle,
                    receipt_path=remote_receipt,
                )
            self.assertTrue(source.is_dir())

    def test_quarantine_reverification_failure_preserves_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                plan,
                plan_path,
                shard,
                raw_root,
                source,
                spool_root,
                bundle,
                incoming,
            ) = _export_and_copy(root)
            canonical = root / "local" / plan.run_namespace / CANONICAL_ROOT_NAME
            local_receipts = root / "local" / plan.run_namespace / RECEIPT_ROOT_NAME
            receipt = import_v5_planned_shard_spool(
                plan_path,
                shard_index=shard.index,
                bundle_path=incoming,
                canonical_root=canonical,
                receipt_root=local_receipts,
                minimum_free_after_import_bytes=0,
            )
            remote_root = root / "remote" / plan.run_namespace / RECEIPT_ROOT_NAME
            remote_root.mkdir(parents=True)
            remote_receipt = remote_root / receipt.name
            shutil.copytree(receipt, remote_receipt)
            original_verify = v5_spool.verify_planned_shard
            calls: list[Path] = []

            def mutate_after_quarantine_verify(plan_value, shard_value, path_value):
                candidate = Path(path_value)
                calls.append(candidate)
                if len(calls) == 2:
                    # Simulate another process creating a new canonical target
                    # after the verified inode was moved into quarantine.
                    source.mkdir()
                    (source / "new-owner.marker").write_bytes(b"untouched")
                digest = original_verify(plan_value, shard_value, candidate)
                if len(calls) == 2:
                    action = candidate / "actor" / "actions.npy"
                    with action.open("r+b") as handle:
                        handle.seek(-1, 2)
                        previous = handle.read(1)
                        handle.seek(-1, 2)
                        handle.write(bytes([previous[0] ^ 1]))
                return digest

            with mock.patch.object(
                v5_spool,
                "verify_planned_shard",
                side_effect=mutate_after_quarantine_verify,
            ), self.assertRaisesRegex(ValueError, "checksum"):
                retire_v5_verified_raw_shard(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    bundle_path=bundle,
                    receipt_path=remote_receipt,
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], source)
            self.assertIn(".retiring-", calls[1].name)
            self.assertEqual((source / "new-owner.marker").read_bytes(), b"untouched")
            quarantines = list(raw_root.glob(f".{shard.name}.retiring-*"))
            self.assertEqual(quarantines, [calls[1]])
            self.assertTrue(quarantines[0].is_dir())
            shutil.rmtree(source)
            with self.assertRaisesRegex(ValueError, "completely receipt-retired"):
                retire_v5_verified_spool_bundle(
                    plan_path,
                    shard_index=shard.index,
                    raw_root=raw_root,
                    spool_root=spool_root,
                    bundle_path=bundle,
                    receipt_path=remote_receipt,
                )


if __name__ == "__main__":
    unittest.main()
