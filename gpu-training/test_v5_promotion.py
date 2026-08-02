from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from v5_evaluate import FINAL_MATCH_COUNTS, V5EvaluationConfig
from v5_export import canonical_json_bytes, export_v5_actor_bundle, v5_actor_bundle_digests
from v5_model import V5ActorConfig, V5PublicActor
import v5_promotion
from v5_promotion import (
    V5_ACTOR_IDENTITY_KEYS,
    V5_FINAL_SEED_STEP,
    V5_FIRST_FINAL_SEED_BASE,
    approve_v5_final_holdout,
    authorize_v5_certification_evaluation,
    authorize_v5_final_evaluation,
    authorize_v5_screening_evaluation,
    claim_v5_final_evaluation_shard,
    load_v5_promotion_plan,
    recover_v5_promotion_lock,
    reserve_v5_certification_execution,
    reserve_v5_final_holdout,
    reserve_v5_screening_execution,
    v5_certification_coordinates,
    verify_v5_final_consumption_receipt,
)
from v5_test_provenance_fixture import synthetic_v5_evaluation_provenance


_PROVENANCE = synthetic_v5_evaluation_provenance("promotion")


def _actor_bundle(path: Path, seed: int) -> dict[str, str]:
    torch.manual_seed(seed)
    actor = V5PublicActor(V5ActorConfig(
        history_latents=2,
        d_model=32,
        core_layers=1,
        heads=4,
        feedforward=64,
    ))
    export_v5_actor_bundle(actor, path)
    return v5_actor_bundle_digests(path)


def _result(player: int, matches: int, *, resamples: int, passed: bool = True) -> dict[str, object]:
    return {
        "playerCount": player,
        "matches": matches,
        "plannedMatches": matches,
        "complete": True,
        "meanCandidateMinusNormalChipPerAct": 0.31 if passed else 0.29,
        "matchClustered95": {
            "method": "deterministic-percentile-bootstrap",
            "unit": "complete-five-act-match",
            "clusters": matches,
            "resamples": resamples,
            "low": 0.21,
            "high": 0.41,
        },
        "candidateBeforeNormalPairwise": {
            "candidateBefore": 58,
            "comparisons": 100,
            "rate": 0.58,
        },
    }


def _certification(
    model: dict[str, str], family: str, seed: int, *, resamples: int = 10_000,
    development_passed: bool = True, evaluator_passed: bool = True,
    provenance: dict[str, object] | None = None,
    reservation_binding: dict[str, object] | None = None,
) -> dict[str, object]:
    plan = {str(player): 60 for player in range(4, 11)}
    return {
        "mode": "certification",
        "familyId": family,
        "seedBase": seed,
        "matchPlan": plan,
        "shard": {"count": 1, "index": 0},
        "model": model,
        "completeEvaluation": True,
        "allPlayerCountsPassed": evaluator_passed,
        "certificationReservation": copy.deepcopy(reservation_binding),
        "evaluationProvenance": [{
            "provenance": copy.deepcopy(provenance or _PROVENANCE),
            "shard": {"count": 1, "index": 0},
        }],
        "results": [
            _result(
                player, 60, resamples=resamples, passed=development_passed
            )
            for player in range(4, 11)
        ],
    }


def _final_report(
    model: dict[str, str], family: str, seed: int, *,
    final_claims: list[dict[str, object]], resamples: int = 10_000,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "mode": "final",
        "familyId": family,
        "seedBase": seed,
        "matchPlan": {
            str(player): matches for player, matches in FINAL_MATCH_COUNTS.items()
        },
        "shard": {"count": 1, "index": 0},
        "model": model,
        "completeEvaluation": True,
        "allPlayerCountsPassed": True,
        "finalClaims": final_claims,
        "evaluationProvenance": [
            {
                "provenance": copy.deepcopy(provenance or _PROVENANCE),
                "shard": dict(claim["shard"]),
            }
            for claim in final_claims
        ],
        "results": [
            _result(player, matches, resamples=resamples)
            for player, matches in FINAL_MATCH_COUNTS.items()
        ],
    }


def _screening(
    model: dict[str, str],
    family: str,
    seed: int,
    *,
    provenance: dict[str, object],
    reservation_binding: dict[str, object],
) -> dict[str, object]:
    return {
        "mode": "screening",
        "familyId": family,
        "seedBase": seed,
        "matchPlan": {str(player): 60 for player in range(4, 11)},
        "shard": {"count": 1, "index": 0},
        "model": model,
        "completeEvaluation": True,
        "allPlayerCountsPassed": True,
        "screeningReservation": copy.deepcopy(reservation_binding),
        "evaluationProvenance": [{
            "provenance": copy.deepcopy(provenance),
            "shard": {"count": 1, "index": 0},
        }],
        "results": [
            _result(player, 60, resamples=10_000)
            for player in range(4, 11)
        ],
    }


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


class V5PromotionTests(unittest.TestCase):
    @staticmethod
    def _validator(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise TypeError("fixture report must be a dictionary")
        return dict(value)

    @staticmethod
    def _write_promotion_lock(
        registry: Path,
        *,
        hostname: str = "host-a",
        boot_id: str = "boot-a",
        pid: int = 2_000_000_000,
        process_start_ticks: int = 123,
    ) -> tuple[Path, dict[str, object]]:
        registry.mkdir(parents=True, exist_ok=True)
        value = {
            "bootId": boot_id,
            "format": v5_promotion.V5_PROMOTION_LOCK_FORMAT,
            "hostname": hostname,
            "pid": pid,
            "processStartTicks": process_start_ticks,
            "registryIdentity": v5_promotion._registry_identity(registry),
            "version": v5_promotion.V5_PROMOTION_LOCK_VERSION,
        }
        lock = registry / ".v5-promotion.lock"
        lock.write_bytes(canonical_json_bytes(value))
        return lock, value

    def test_promotion_lock_is_canonical_and_active_recovery_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            with v5_promotion._registry_lock(registry):
                lock = registry / ".v5-promotion.lock"
                value, raw, _ = v5_promotion._load_promotion_lock(lock)
                self.assertEqual(raw, canonical_json_bytes(value))
                self.assertEqual(
                    set(value),
                    {
                        "bootId", "format", "hostname", "pid",
                        "processStartTicks", "registryIdentity", "version",
                    },
                )
                self.assertEqual(
                    value["registryIdentity"],
                    v5_promotion._registry_identity(registry),
                )
                with self.assertRaisesRegex(RuntimeError, "active"):
                    recover_v5_promotion_lock(
                        registry,
                        recovery_reason="must not retire an active promotion lock",
                    )
            self.assertFalse((registry / ".v5-promotion.lock").exists())

    def test_partial_prelink_locks_are_ignored_by_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            registry.mkdir(parents=True)
            lock = registry / ".v5-promotion.lock"
            partial_bytes = b'{"format":"dalmuti-v5-promotion-registry-lock"'
            partials = [
                registry / (
                    v5_promotion._publish_temporary_prefix(lock)
                    + f"crashed-{ordinal}.tmp"
                )
                for ordinal in range(2)
            ]
            for partial in partials:
                partial.write_bytes(partial_bytes)
            self.assertFalse(lock.exists())

            for _ in range(2):
                with v5_promotion._registry_lock(registry):
                    self.assertTrue(lock.exists())
                    v5_promotion._load_promotion_lock(lock)
                    self.assertTrue(all(partial.exists() for partial in partials))
            self.assertTrue(all(partial.read_bytes() == partial_bytes for partial in partials))

    def test_reader_does_not_mutate_an_active_publish_temporary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "record.json"
            expected = {"complete": True}
            v5_promotion._write_exclusive(
                record, canonical_json_bytes(expected)
            )
            active = root / (
                v5_promotion._publish_temporary_prefix(record) + "active.tmp"
            )
            with active.open("xb") as writer:
                writer.write(b'{"complete":')
                writer.flush()
                loaded, _ = v5_promotion._strict_canonical_object(
                    record, "concurrent reader fixture"
                )
                self.assertEqual(loaded, expected)
                self.assertTrue(active.exists())
            self.assertEqual(active.read_bytes(), b'{"complete":')

    def test_lock_postlink_failure_releases_the_exact_owned_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            with mock.patch.object(
                v5_promotion,
                "_fsync_directory",
                side_effect=OSError("simulated post-link durability failure"),
            ):
                with self.assertRaisesRegex(OSError, "post-link"):
                    with v5_promotion._registry_lock(registry):
                        self.fail("lock body must not start after durability failure")
            self.assertFalse((registry / ".v5-promotion.lock").exists())

    def test_promotion_lock_recovery_evidence_and_retry_safe_completion(self) -> None:
        cases = (
            ("process-missing", "boot-a", None, False),
            ("pid-reused", "boot-a", 456, True),
            ("host-rebooted", "old-boot", None, True),
        )
        for expected, old_boot, observed_old_start, may_exist in cases:
            with self.subTest(expected), tempfile.TemporaryDirectory() as temporary:
                registry = Path(temporary) / "registry"
                old_pid = 2_000_000_000
                lock, old_lock = self._write_promotion_lock(
                    registry,
                    boot_id=old_boot,
                    pid=old_pid,
                )

                def process_start(pid: int) -> int | None:
                    return 999 if pid == os.getpid() else observed_old_start

                with mock.patch.object(
                    v5_promotion, "_host_boot_id", return_value="boot-a"
                ), mock.patch.object(
                    v5_promotion.socket, "gethostname", return_value="host-a"
                ), mock.patch.object(
                    v5_promotion, "_process_start_ticks", side_effect=process_start
                ), mock.patch.object(
                    v5_promotion, "_process_may_exist", return_value=may_exist
                ):
                    result = recover_v5_promotion_lock(
                        registry,
                        recovery_reason=f"audited stale lock: {expected}",
                    )
                    retry = recover_v5_promotion_lock(
                        registry,
                        recovery_reason=f"audited stale lock: {expected}",
                    )
                    self.assertTrue(retry["resumedRecovery"])
                    self.assertEqual(retry["oldLockSha256"], result["oldLockSha256"])
                    # A new protected operation validates and clears only the
                    # completed recovery pointers; the archive stays immutable.
                    with v5_promotion._registry_lock(registry):
                        self.assertTrue(lock.exists())

                self.assertFalse(lock.exists())
                receipt, receipt_raw = v5_promotion._load_promotion_recovery_receipt(
                    Path(result["receipt"])
                )
                self.assertEqual(receipt["observedStaleEvidence"], expected)
                self.assertEqual(receipt["oldLock"], old_lock)
                self.assertEqual(
                    receipt_raw,
                    canonical_json_bytes(receipt),
                )
                self.assertEqual(
                    Path(result["retiredLock"]).read_bytes(),
                    canonical_json_bytes(old_lock),
                )
                active, complete = v5_promotion._recovery_pointer_paths(registry)
                self.assertFalse(active.exists())
                self.assertFalse(complete.exists())

    def test_promotion_recovery_refuses_foreign_or_unprovable_active_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            lock, _ = self._write_promotion_lock(registry)
            with mock.patch.object(
                v5_promotion, "_host_boot_id", return_value="boot-a"
            ), mock.patch.object(
                v5_promotion.socket, "gethostname", return_value="host-b"
            ), mock.patch.object(
                v5_promotion, "_process_start_ticks", return_value=999
            ):
                with self.assertRaisesRegex(RuntimeError, "another host"):
                    recover_v5_promotion_lock(
                        registry, recovery_reason="must not trust a remote PID"
                    )
            self.assertTrue(lock.exists())

        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            old_pid = 2_000_000_000
            lock, _ = self._write_promotion_lock(registry, pid=old_pid)

            def unreadable_start(pid: int) -> int | None:
                return 999 if pid == os.getpid() else None

            with mock.patch.object(
                v5_promotion, "_host_boot_id", return_value="boot-a"
            ), mock.patch.object(
                v5_promotion.socket, "gethostname", return_value="host-a"
            ), mock.patch.object(
                v5_promotion, "_process_start_ticks", side_effect=unreadable_start
            ), mock.patch.object(
                v5_promotion, "_process_may_exist", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "may still be active"):
                    recover_v5_promotion_lock(
                        registry,
                        recovery_reason="identity lookup did not prove process death",
                    )
            self.assertTrue(lock.exists())

    def test_promotion_recovery_binds_exact_registry_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            lock, value = self._write_promotion_lock(registry)
            value["registryIdentity"]["inode"] += 1  # type: ignore[index]
            lock.write_bytes(canonical_json_bytes(value))
            with mock.patch.object(
                v5_promotion, "_host_boot_id", return_value="boot-a"
            ), mock.patch.object(
                v5_promotion.socket, "gethostname", return_value="host-a"
            ), mock.patch.object(
                v5_promotion, "_process_start_ticks", return_value=None
            ), mock.patch.object(
                v5_promotion, "_process_may_exist", return_value=False
            ):
                with self.assertRaisesRegex(ValueError, "another registry identity"):
                    recover_v5_promotion_lock(
                        registry,
                        recovery_reason="registry substitution must fail closed",
                    )
            self.assertTrue(lock.exists())

    def test_promotion_recovery_never_replaces_retired_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = Path(temporary) / "registry"
            old_pid = 2_000_000_000
            lock, _ = self._write_promotion_lock(registry, pid=old_pid)
            lock_sha = v5_promotion._sha256_bytes(lock.read_bytes())
            retired = v5_promotion._retired_lock_path(registry, lock_sha)
            retired.parent.mkdir(parents=True, exist_ok=True)
            retired.write_bytes(b"attacker-owned-target")

            def missing_start(pid: int) -> int | None:
                return 999 if pid == os.getpid() else None

            with mock.patch.object(
                v5_promotion, "_host_boot_id", return_value="boot-a"
            ), mock.patch.object(
                v5_promotion.socket, "gethostname", return_value="host-a"
            ), mock.patch.object(
                v5_promotion, "_process_start_ticks", side_effect=missing_start
            ), mock.patch.object(
                v5_promotion, "_process_may_exist", return_value=False
            ):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    recover_v5_promotion_lock(
                        registry,
                        recovery_reason="retired target collision regression",
                    )
            self.assertEqual(retired.read_bytes(), b"attacker-owned-target")
            self.assertTrue(lock.exists())

    def test_promotion_recovery_resumes_both_crash_windows(self) -> None:
        for crash_point in ("before-retirement", "before-completion"):
            with self.subTest(crash_point), tempfile.TemporaryDirectory() as temporary:
                registry = Path(temporary) / "registry"
                old_pid = 2_000_000_000
                lock, _ = self._write_promotion_lock(registry, pid=old_pid)
                real_link = os.link
                failed = False

                def crash_once(source: object, target: object, *args: object, **kwargs: object) -> None:
                    nonlocal failed
                    destination = Path(target)  # type: ignore[arg-type]
                    should_fail = (
                        crash_point == "before-retirement"
                        and destination.parent.name == "retired-locks"
                    ) or (
                        crash_point == "before-completion"
                        and destination.name
                        == ".v5-promotion-lock-recovery-complete.json"
                    )
                    if should_fail and not failed:
                        failed = True
                        raise OSError(f"simulated crash {crash_point}")
                    real_link(source, target, *args, **kwargs)

                def missing_start(pid: int) -> int | None:
                    return 999 if pid == os.getpid() else None

                patches = (
                    mock.patch.object(v5_promotion, "_host_boot_id", return_value="boot-a"),
                    mock.patch.object(v5_promotion.socket, "gethostname", return_value="host-a"),
                    mock.patch.object(v5_promotion, "_process_start_ticks", side_effect=missing_start),
                    mock.patch.object(v5_promotion, "_process_may_exist", return_value=False),
                )
                with patches[0], patches[1], patches[2], patches[3]:
                    with mock.patch.object(v5_promotion.os, "link", side_effect=crash_once):
                        with self.assertRaisesRegex(OSError, "simulated crash"):
                            recover_v5_promotion_lock(
                                registry,
                                recovery_reason=f"resume {crash_point}",
                            )
                    if crash_point == "before-retirement":
                        self.assertTrue(lock.exists())
                    else:
                        self.assertFalse(lock.exists())
                    recovered = recover_v5_promotion_lock(
                        registry,
                        recovery_reason=f"resume {crash_point}",
                    )
                self.assertTrue(recovered["resumedRecovery"])
                self.assertFalse(lock.exists())
                self.assertTrue(Path(recovered["retiredLock"]).exists())

    def _reserve(
        self,
        root: Path,
        *,
        actor_seed: int = 1001,
        label: str = "first",
        shard_count: int = 1,
    ) -> tuple[Path, dict[str, str], dict[str, object]]:
        bundle = root / f"actor-bundle-{label}"
        model = _actor_bundle(bundle, actor_seed)
        registry = root / "registry"
        execution, _, _ = self._reserve_certification_with_screening(
            registry, bundle, model, _PROVENANCE
        )
        coordinates = execution["reservation"]["coordinates"]
        paths: list[Path] = []
        bindings: list[dict[str, object]] = []
        for coordinate in coordinates:
            output = registry.joinpath(*str(coordinate["outputPath"]).split("/"))
            binding = authorize_v5_certification_evaluation(
                execution["reservationPath"],
                model,
                evaluation_provenance=_PROVENANCE,
                family_id=str(coordinate["familyId"]),
                seed_base=int(coordinate["seedBase"]),
                match_plan={player: 60 for player in range(4, 11)},
                match_shard_count=1,
                match_shard_index=0,
                bootstrap_resamples=10_000,
                output_path=output,
            )
            paths.append(output)
            bindings.append(binding)
        first, second = paths
        _write(
            first,
            _certification(
                model,
                str(coordinates[0]["familyId"]),
                int(coordinates[0]["seedBase"]),
                reservation_binding=bindings[0],
            ),
        )
        _write(
            second,
            _certification(
                model,
                str(coordinates[1]["familyId"]),
                int(coordinates[1]["seedBase"]),
                reservation_binding=bindings[1],
            ),
        )
        reservation = reserve_v5_final_holdout(
            registry,
            bundle,
            (first, second),
            final_match_shard_count=shard_count,
        )
        return bundle, model, reservation

    def _reserve_certification_with_screening(
        self,
        registry: Path,
        bundle: Path,
        model: dict[str, str],
        provenance: dict[str, object],
    ) -> tuple[dict[str, object], Path, Path]:
        screening = reserve_v5_screening_execution(
            registry, bundle, provenance
        )
        reservation = screening["reservation"]
        coordinate = reservation["coordinate"]
        output = registry.joinpath(*str(reservation["outputPath"]).split("/"))
        binding = authorize_v5_screening_evaluation(
            screening["reservationPath"],
            model,
            evaluation_provenance=provenance,
            family_id=str(coordinate["familyId"]),
            seed_base=int(coordinate["seedBase"]),
            match_plan={player: 60 for player in range(4, 11)},
            match_shard_count=1,
            match_shard_index=0,
            bootstrap_resamples=10_000,
            output_path=output,
        )
        _write(
            output,
            _screening(
                model,
                str(coordinate["familyId"]),
                int(coordinate["seedBase"]),
                provenance=provenance,
                reservation_binding=binding,
            ),
        )
        digest = v5_promotion._sha256_file(output)
        output.with_name(output.name + ".sha256").write_bytes(
            f"{digest}  {output.name}\n".encode("ascii")
        )
        with mock.patch.object(
            v5_promotion,
            "validate_v5_evaluation_report",
            side_effect=self._validator,
        ):
            execution = reserve_v5_certification_execution(
                registry,
                bundle,
                provenance,
                screening_reservation=screening["reservationPath"],
                screening_report=output,
            )
        return execution, Path(screening["reservationPath"]), output

    def test_reservation_binds_exact_model_certificates_and_first_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            bundle, model, reservation = self._reserve(root)
            plan = load_v5_promotion_plan(reservation["planPath"])
            self.assertEqual(set(plan["model"]), V5_ACTOR_IDENTITY_KEYS)
            self.assertEqual(plan["model"], model)
            self.assertEqual(plan["final"]["seedBase"], V5_FIRST_FINAL_SEED_BASE)
            self.assertEqual(
                plan["final"]["familyId"],
                f"v5-final-holdout-s{V5_FIRST_FINAL_SEED_BASE}",
            )
            self.assertEqual(plan["final"]["bootstrapResamples"], 10_000)
            self.assertFalse(plan["final"]["finalReportTrainingOrTuningUseAllowed"])
            reports = plan["certification"]["reports"]
            self.assertEqual(len(reports), 2)
            self.assertNotEqual(reports[0]["reportSha256"], reports[1]["reportSha256"])
            certification_paths = tuple(
                root.joinpath(
                    "registry",
                    *str(report["certificationReservation"]["outputPath"]).split("/"),
                )
                for report in reports
            )
            retry = reserve_v5_final_holdout(
                root / "registry",
                bundle,
                certification_paths,
                final_match_shard_count=1,
            )
            self.assertEqual(retry["plan"], plan)

    def test_certification_execution_is_preregistered_and_retry_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            bundle = root / "actor-bundle"
            model = _actor_bundle(bundle, 1551)
            first, screening_reservation, screening_report = (
                self._reserve_certification_with_screening(
                    registry, bundle, model, _PROVENANCE
                )
            )
            with mock.patch.object(
                v5_promotion,
                "validate_v5_evaluation_report",
                side_effect=self._validator,
            ):
                retry = reserve_v5_certification_execution(
                    registry,
                    bundle,
                    copy.deepcopy(_PROVENANCE),
                    screening_reservation=screening_reservation,
                    screening_report=screening_report,
                )
            self.assertEqual(first, retry)
            with mock.patch.object(
                v5_promotion,
                "validate_v5_evaluation_report",
                side_effect=self._validator,
            ), self.assertRaisesRegex(ValueError, "differs"):
                reserve_v5_certification_execution(
                    registry,
                    bundle,
                    synthetic_v5_evaluation_provenance("alternate-execution"),
                    screening_reservation=screening_reservation,
                    screening_report=screening_report,
                )

            coordinate = first["reservation"]["coordinates"][0]
            output = registry.joinpath(*str(coordinate["outputPath"]).split("/"))
            binding = authorize_v5_certification_evaluation(
                first["reservationPath"],
                model,
                evaluation_provenance=_PROVENANCE,
                family_id=str(coordinate["familyId"]),
                seed_base=int(coordinate["seedBase"]),
                match_plan={player: 60 for player in range(4, 11)},
                match_shard_count=1,
                match_shard_index=0,
                bootstrap_resamples=10_000,
                output_path=output,
            )
            self.assertEqual(binding["outputPath"], coordinate["outputPath"])
            with self.assertRaisesRegex(ValueError, "differs from"):
                authorize_v5_certification_evaluation(
                    first["reservationPath"],
                    model,
                    evaluation_provenance=synthetic_v5_evaluation_provenance(
                        "alternate-execution"
                    ),
                    family_id=str(coordinate["familyId"]),
                    seed_base=int(coordinate["seedBase"]),
                    match_plan={player: 60 for player in range(4, 11)},
                    match_shard_count=1,
                    match_shard_index=0,
                    bootstrap_resamples=10_000,
                    output_path=output,
                )
            with self.assertRaisesRegex(ValueError, "output path differs"):
                authorize_v5_certification_evaluation(
                    first["reservationPath"],
                    model,
                    evaluation_provenance=_PROVENANCE,
                    family_id=str(coordinate["familyId"]),
                    seed_base=int(coordinate["seedBase"]),
                    match_plan={player: 60 for player in range(4, 11)},
                    match_shard_count=1,
                    match_shard_index=0,
                    bootstrap_resamples=10_000,
                    output_path=root / "cherry-picked.json",
                )

    def test_screening_is_one_shot_for_same_tensor_repackaged_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            torch.manual_seed(1771)
            actor = V5PublicActor(V5ActorConfig(
                history_latents=2,
                d_model=32,
                core_layers=1,
                heads=4,
                feedforward=64,
            ))
            first_bundle = root / "actor-a"
            second_bundle = root / "actor-b"
            export_v5_actor_bundle(actor, first_bundle, metadata={"label": "a"})
            export_v5_actor_bundle(actor, second_bundle, metadata={"label": "b"})
            first_identity = v5_actor_bundle_digests(first_bundle)
            second_identity = v5_actor_bundle_digests(second_bundle)
            self.assertEqual(
                first_identity["tensorStateSha256"],
                second_identity["tensorStateSha256"],
            )
            self.assertNotEqual(
                first_identity["manifestSha256"],
                second_identity["manifestSha256"],
            )
            first = reserve_v5_screening_execution(
                registry, first_bundle, _PROVENANCE
            )
            retry = reserve_v5_screening_execution(
                registry, first_bundle, copy.deepcopy(_PROVENANCE)
            )
            self.assertEqual(first, retry)
            with self.assertRaisesRegex(ValueError, "already has a screening"):
                reserve_v5_screening_execution(
                    registry,
                    first_bundle,
                    synthetic_v5_evaluation_provenance("other-run"),
                )
            with self.assertRaisesRegex(ValueError, "already has a screening"):
                reserve_v5_screening_execution(
                    registry, second_bundle, _PROVENANCE
                )

    def test_screening_and_certification_publish_crash_windows_recover(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry"
            bundle = root / "actor-bundle"
            model = _actor_bundle(bundle, 1772)

            coordinate = v5_promotion._screening_coordinate_for_model(model)
            provenance = v5_promotion._validated_certification_execution_provenance(
                _PROVENANCE
            )
            base = {
                "coordinate": coordinate,
                "evaluationProvenance": provenance,
                "evaluationProvenanceSha256": provenance["provenanceSha256"],
                "format": v5_promotion.V5_SCREENING_RESERVATION_FORMAT,
                "functionalPolicyIdentity": (
                    v5_promotion._functional_policy_identity(model)
                ),
                "model": model,
                "version": v5_promotion.V5_SCREENING_RESERVATION_VERSION,
            }
            screening_id = v5_promotion._screening_reservation_id(base)
            screening_path = (
                registry / "screening-reservations" / f"{screening_id}.json"
            )
            screening_path.parent.mkdir(parents=True)
            primary_orphan = screening_path.parent / (
                v5_promotion._publish_temporary_prefix(screening_path)
                + "crashed.tmp"
            )
            primary_orphan.write_bytes(b'{"partial-screening"')

            screening = reserve_v5_screening_execution(
                registry, bundle, _PROVENANCE
            )
            self.assertEqual(Path(screening["reservationPath"]), screening_path)
            self.assertTrue(primary_orphan.exists())
            self.assertTrue(screening_path.with_name(
                screening_path.name + ".sha256"
            ).exists())

            screening_reservation = screening["reservation"]
            output = registry.joinpath(
                *str(screening_reservation["outputPath"]).split("/")
            )
            binding = authorize_v5_screening_evaluation(
                screening_path,
                model,
                evaluation_provenance=_PROVENANCE,
                family_id=str(coordinate["familyId"]),
                seed_base=int(coordinate["seedBase"]),
                match_plan={player: 60 for player in range(4, 11)},
                match_shard_count=1,
                match_shard_index=0,
                bootstrap_resamples=10_000,
                output_path=output,
            )
            _write(
                output,
                _screening(
                    model,
                    str(coordinate["familyId"]),
                    int(coordinate["seedBase"]),
                    provenance=_PROVENANCE,
                    reservation_binding=binding,
                ),
            )
            output_sha = v5_promotion._sha256_file(output)
            output.with_name(output.name + ".sha256").write_bytes(
                f"{output_sha}  {output.name}\n".encode("ascii")
            )
            with mock.patch.object(
                v5_promotion,
                "validate_v5_evaluation_report",
                side_effect=self._validator,
            ):
                first = reserve_v5_certification_execution(
                    registry,
                    bundle,
                    _PROVENANCE,
                    screening_reservation=screening_path,
                    screening_report=output,
                )

            certification_path = Path(first["reservationPath"])
            certification_sidecar = certification_path.with_name(
                certification_path.name + ".sha256"
            )
            certification_path.unlink()
            certification_sidecar.unlink()
            certification_orphan = certification_path.parent / (
                v5_promotion._publish_temporary_prefix(certification_path)
                + "crashed.tmp"
            )
            certification_orphan.write_bytes(b'{"partial-certification"')
            with mock.patch.object(
                v5_promotion,
                "validate_v5_evaluation_report",
                side_effect=self._validator,
            ):
                second = reserve_v5_certification_execution(
                    registry,
                    bundle,
                    _PROVENANCE,
                    screening_reservation=screening_path,
                    screening_report=output,
                )
            self.assertEqual(first, second)
            self.assertTrue(certification_orphan.exists())

            certification_sidecar.unlink()
            sidecar_orphan = certification_sidecar.parent / (
                v5_promotion._publish_temporary_prefix(certification_sidecar)
                + "crashed.tmp"
            )
            sidecar_orphan.write_bytes(b"partial-checksum")
            with mock.patch.object(
                v5_promotion,
                "validate_v5_evaluation_report",
                side_effect=self._validator,
            ):
                third = reserve_v5_certification_execution(
                    registry,
                    bundle,
                    _PROVENANCE,
                    screening_reservation=screening_path,
                    screening_report=output,
                )
            self.assertEqual(second, third)
            self.assertTrue(sidecar_orphan.exists())
            digest = v5_promotion._sha256_file(certification_path)
            self.assertEqual(
                certification_sidecar.read_bytes(),
                f"{digest}  {certification_path.name}\n".encode("ascii"),
            )

    def test_certification_bootstrap_gate_and_model_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            bundle = root / "actor-bundle"
            model = _actor_bundle(bundle, 2001)
            registry = root / "registry"
            execution, _, _ = self._reserve_certification_with_screening(
                registry, bundle, model, _PROVENANCE
            )
            coordinates = execution["reservation"]["coordinates"]
            outputs = [
                registry.joinpath(*str(item["outputPath"]).split("/"))
                for item in coordinates
            ]
            bindings = [
                authorize_v5_certification_evaluation(
                    execution["reservationPath"],
                    model,
                    evaluation_provenance=_PROVENANCE,
                    family_id=str(item["familyId"]),
                    seed_base=int(item["seedBase"]),
                    match_plan={player: 60 for player in range(4, 11)},
                    match_shard_count=1,
                    match_shard_index=0,
                    bootstrap_resamples=10_000,
                    output_path=output,
                )
                for item, output in zip(coordinates, outputs, strict=True)
            ]
            good, bad = outputs
            _write(
                good,
                _certification(
                    model,
                    str(coordinates[0]["familyId"]),
                    int(coordinates[0]["seedBase"]),
                    reservation_binding=bindings[0],
                ),
            )
            _write(
                bad,
                _certification(
                    model,
                    str(coordinates[1]["familyId"]),
                    int(coordinates[1]["seedBase"]),
                    resamples=1,
                    reservation_binding=bindings[1],
                ),
            )
            with self.assertRaisesRegex(ValueError, "10000"):
                reserve_v5_final_holdout(
                    registry,
                    bundle,
                    (good, bad),
                    final_match_shard_count=1,
                )

            different_source = bad
            _write(
                different_source,
                _certification(
                    model,
                    str(coordinates[1]["familyId"]),
                    int(coordinates[1]["seedBase"]),
                    provenance=synthetic_v5_evaluation_provenance("other-source"),
                    reservation_binding=bindings[1],
                ),
            )
            with self.assertRaisesRegex(
                ValueError, "provenance differs"
            ):
                reserve_v5_final_holdout(
                    registry,
                    bundle,
                    (good, different_source),
                    final_match_shard_count=1,
                )
            failed = _certification(
                model,
                str(coordinates[1]["familyId"]),
                int(coordinates[1]["seedBase"]),
                development_passed=False,
                reservation_binding=bindings[1],
            )
            _write(bad, failed)
            with self.assertRaisesRegex(ValueError, "development gate"):
                reserve_v5_final_holdout(
                    registry,
                    bundle,
                    (good, bad),
                    final_match_shard_count=1,
                )

            wrong_bundle = root / "wrong-actor-bundle"
            wrong_model = _actor_bundle(wrong_bundle, 2002)
            wrong_coordinates = v5_certification_coordinates(wrong_bundle)
            _write(
                bad,
                _certification(
                    wrong_model,
                    str(wrong_coordinates[1]["familyId"]),
                    int(wrong_coordinates[1]["seedBase"]),
                    reservation_binding=bindings[1],
                ),
            )
            with self.assertRaisesRegex(ValueError, "different frozen Actor"):
                reserve_v5_final_holdout(
                    registry,
                    bundle,
                    (good, bad),
                    final_match_shard_count=1,
                )

    def test_final_requires_reserved_family_exact_bootstrap_and_one_consumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            bundle, model, reservation = self._reserve(root)
            plan = reservation["plan"]
            family = plan["final"]["familyId"]
            seed = plan["final"]["seedBase"]
            final_path = root / "final.json"
            claim = claim_v5_final_evaluation_shard(
                reservation["planPath"],
                bundle,
                evaluation_provenance=_PROVENANCE,
                match_shard_count=1,
                match_shard_index=0,
            )["reportBinding"]

            arbitrary = _final_report(
                model,
                "arbitrary-family",
                seed + 7,
                final_claims=[claim],
            )
            _write(final_path, arbitrary)
            with self.assertRaisesRegex(ValueError, "unreserved"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

            final_path.unlink()
            _write(
                final_path,
                _final_report(
                    model, family, seed, final_claims=[claim], resamples=1
                ),
            )
            with self.assertRaisesRegex(ValueError, "10000"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

            wrong_bundle = root / "wrong-bundle"
            _actor_bundle(wrong_bundle, 2002)
            final_path.unlink()
            _write(
                final_path,
                _final_report(
                    model,
                    family,
                    seed,
                    final_claims=[claim],
                    provenance=synthetic_v5_evaluation_provenance("other-source"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "differs from its promotion plan"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

            final_path.unlink()
            _write(
                final_path,
                _final_report(model, family, seed, final_claims=[claim]),
            )
            with self.assertRaisesRegex(ValueError, "model bindings disagree"):
                approve_v5_final_holdout(
                    reservation["planPath"], wrong_bundle, final_path
                )

            approval = approve_v5_final_holdout(
                reservation["planPath"], bundle, final_path
            )
            receipt = verify_v5_final_consumption_receipt(
                approval["receiptPath"],
                reservation["planPath"],
                bundle,
                final_path,
            )
            self.assertTrue(receipt["approved"])
            self.assertTrue(receipt["consumed"])
            self.assertFalse(receipt["finalReportTrainingOrTuningUseAllowed"])
            with self.assertRaisesRegex(FileExistsError, "already consumed"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

    def test_second_actor_uses_next_twenty_million_seed_progression(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            _, _, first = self._reserve(root, actor_seed=3001, label="first")
            _, _, second = self._reserve(root, actor_seed=3002, label="second")
            self.assertEqual(first["plan"]["final"]["seedBase"], V5_FIRST_FINAL_SEED_BASE)
            self.assertEqual(
                second["plan"]["final"]["seedBase"],
                V5_FIRST_FINAL_SEED_BASE + V5_FINAL_SEED_STEP,
            )

    def test_shard_count_boundary_keeps_every_shard_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            _, _, reservation = self._reserve(root, shard_count=2500)
            self.assertEqual(reservation["plan"]["final"]["matchShardCount"], 2500)
            with self.assertRaisesRegex(ValueError, "1..2500"):
                reserve_v5_final_holdout(
                    root / "unused-registry",
                    root / "unused-bundle",
                    (root / "unused-a.json", root / "unused-b.json"),
                    final_match_shard_count=2501,
                )

    def test_shard_claims_are_one_shot_and_approval_requires_exact_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "v5_promotion.validate_v5_evaluation_report", side_effect=self._validator
        ):
            root = Path(temporary)
            bundle, model, reservation = self._reserve(root, shard_count=2)
            plan = reservation["plan"]
            first = claim_v5_final_evaluation_shard(
                reservation["planPath"],
                bundle,
                evaluation_provenance=_PROVENANCE,
                match_shard_count=2,
                match_shard_index=0,
            )
            claim_output = (root / "registry").joinpath(
                *str(first["reportBinding"]["outputPath"]).split("/")
            )
            authorized = authorize_v5_final_evaluation(
                reservation["planPath"],
                first["claimPath"],
                model,
                evaluation_provenance=_PROVENANCE,
                family_id=str(plan["final"]["familyId"]),
                seed_base=int(plan["final"]["seedBase"]),
                match_plan=FINAL_MATCH_COUNTS,
                match_shard_count=2,
                match_shard_index=0,
                bootstrap_resamples=10_000,
                output_path=claim_output,
            )
            self.assertEqual(authorized, first["reportBinding"])
            with self.assertRaisesRegex(ValueError, "promotion plan"):
                claim_v5_final_evaluation_shard(
                    reservation["planPath"],
                    bundle,
                    evaluation_provenance=synthetic_v5_evaluation_provenance(
                        "alternate-final"
                    ),
                    match_shard_count=2,
                    match_shard_index=1,
                )
            with self.assertRaisesRegex(ValueError, "output path differs"):
                authorize_v5_final_evaluation(
                    reservation["planPath"],
                    first["claimPath"],
                    model,
                    evaluation_provenance=_PROVENANCE,
                    family_id=str(plan["final"]["familyId"]),
                    seed_base=int(plan["final"]["seedBase"]),
                    match_plan=FINAL_MATCH_COUNTS,
                    match_shard_count=2,
                    match_shard_index=0,
                    bootstrap_resamples=10_000,
                    output_path=root / "wrong-final.json",
                )
            retry = claim_v5_final_evaluation_shard(
                reservation["planPath"],
                bundle,
                evaluation_provenance=_PROVENANCE,
                match_shard_count=2,
                match_shard_index=0,
            )
            self.assertEqual(retry["claim"], first["claim"])
            with self.assertRaisesRegex(ValueError, "differs from reservation"):
                claim_v5_final_evaluation_shard(
                    reservation["planPath"],
                    bundle,
                    evaluation_provenance=_PROVENANCE,
                    match_shard_count=3,
                    match_shard_index=1,
                )
            final_path = root / "final-sharded.json"
            _write(
                final_path,
                _final_report(
                    model,
                    str(plan["final"]["familyId"]),
                    int(plan["final"]["seedBase"]),
                    final_claims=[first["reportBinding"]],
                ),
            )
            with self.assertRaisesRegex(ValueError, "every reserved shard"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

            second = claim_v5_final_evaluation_shard(
                reservation["planPath"],
                bundle,
                evaluation_provenance=_PROVENANCE,
                match_shard_count=2,
                match_shard_index=1,
            )
            final_path.unlink()
            fake = copy.deepcopy(first["reportBinding"])
            fake["claimId"] = "f" * 64
            _write(
                final_path,
                _final_report(
                    model,
                    str(plan["final"]["familyId"]),
                    int(plan["final"]["seedBase"]),
                    final_claims=[fake, second["reportBinding"]],
                ),
            )
            with self.assertRaisesRegex(ValueError, "canonical registry"):
                approve_v5_final_holdout(
                    reservation["planPath"], bundle, final_path
                )

            final_path.unlink()
            bindings = [first["reportBinding"], second["reportBinding"]]
            _write(
                final_path,
                _final_report(
                    model,
                    str(plan["final"]["familyId"]),
                    int(plan["final"]["seedBase"]),
                    final_claims=bindings,
                ),
            )
            approval = approve_v5_final_holdout(
                reservation["planPath"], bundle, final_path
            )
            self.assertEqual(approval["receipt"]["finalClaims"], bindings)

    def test_evaluator_rejects_one_resample_final_before_collection(self) -> None:
        exact = tuple(sorted(FINAL_MATCH_COUNTS.items()))
        with self.assertRaisesRegex(ValueError, "exactly 10000"):
            V5EvaluationConfig(
                "final",
                "v5-final-holdout-s900000001",
                900_000_001,
                match_counts=exact,
                bootstrap_resamples=1,
            )


if __name__ == "__main__":
    unittest.main()
