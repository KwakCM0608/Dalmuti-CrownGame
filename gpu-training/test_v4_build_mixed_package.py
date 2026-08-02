from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import v4_mixed_package_runtime as runtime
from v4_build_mixed_package import build_package
from v4_mixed_package_runtime import (
    canonical_json_bytes,
    extract_source,
    recheck_snapshot,
    seal_run,
    sha256_bytes,
    stable_snapshot,
    verify_package,
    verify_remote_package_source,
    verify_screening,
    write_status,
)
from v4_model import canonical_v4_policy_numerics_contract


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def _write_sidecar(path: Path) -> str:
    digest = sha256_bytes(path.read_bytes())
    Path(f"{path}.sha256").write_bytes(f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _writable_tree(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
        except OSError:
            pass
    try:
        os.chmod(root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except OSError:
        pass


class MixedPackageBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        _git(self.repository, "init", "--quiet")
        _git(self.repository, "config", "user.email", "test@example.invalid")
        _git(self.repository, "config", "user.name", "V4 Package Test")
        _git(self.repository, "config", "core.autocrlf", "false")
        (self.repository / "docs").mkdir()
        (self.repository / "gpu-training").mkdir()
        (self.repository / "docs" / "ledger.md").write_bytes(b"sealed mixed plan\n")
        (self.repository / "gpu-training" / "entry.py").write_bytes(b"#!/usr/bin/env python3\nprint('workflow')\n")
        (self.repository / "gpu-training" / "payload.py").write_bytes(b"VALUE = 'committed-lf'\n")
        (self.repository / "gpu-training" / "v4_evaluate.py").write_bytes(
            b"def validate_benchmark_report(report, *, expected_mode=None):\n"
            b"    if report.get('format') != 'dalmuti-model-benchmark' or expected_mode != 'screening':\n"
            b"        raise ValueError('invalid benchmark')\n"
        )
        runtime_source = Path(__file__).with_name("v4_mixed_package_runtime.py").read_bytes()
        (self.repository / "gpu-training" / "v4_mixed_package_runtime.py").write_bytes(runtime_source)
        builder_source = Path(__file__).with_name("v4_build_mixed_package.py").read_bytes()
        (self.repository / "gpu-training" / "v4_build_mixed_package.py").write_bytes(builder_source)
        self.recipe_path = "gpu-training/recipe.json"
        source_paths = sorted(
            [
                "docs/ledger.md",
                "gpu-training/entry.py",
                "gpu-training/payload.py",
                self.recipe_path,
                "gpu-training/v4_build_mixed_package.py",
                "gpu-training/v4_evaluate.py",
                "gpu-training/v4_mixed_package_runtime.py",
            ]
        )
        self.recipe = {
            "entrypoint": {
                "argv": ["--source-root", "{source_root}", "--run-directory", "{run_directory}"],
                "path": "gpu-training/entry.py",
            },
            "format": "dalmuti-v4-mixed-package-recipe",
            "ledgerPath": "docs/ledger.md",
            "packageId": "v4-fixedid-ppo-i001-mixedmath-s600000001-test",
            "packagingBuilderPath": "gpu-training/v4_build_mixed_package.py",
            "runContract": {
                "backendMap": ["cpu", "cpu", *(["cuda"] * 12)],
                "environmentSeed": 600000001,
                "policyNumerics": canonical_v4_policy_numerics_contract(),
                "trainingSeed": 610000001,
            },
            "runtimeVerifierPath": "gpu-training/v4_mixed_package_runtime.py",
            "screening": {
                "actsPerMatch": 5,
                "baseSeed": 450000001,
                "bootstrapResamples": 10000,
                "candidateDirectory": "candidate",
                "evaluatorPath": "gpu-training/v4_evaluate.py",
                "familyId": "attempt004-screening-seed450000001",
                "matchesPerPlayerCount": 60,
                "normalBaselineSha256": "a" * 64,
                "observationSchemaSha256": "b" * 64,
                "playerCounts": list(range(4, 11)),
                "reportPath": "screening/epoch-0001.json",
            },
            "sourcePaths": source_paths,
            "version": 1,
        }
        (self.repository / self.recipe_path).write_bytes(canonical_json_bytes(self.recipe))
        _git(self.repository, "add", ".")
        _git(self.repository, "commit", "--quiet", "-m", "sealed test source")
        self.commit = _git(self.repository, "rev-parse", "HEAD")
        self.outputs: list[Path] = []

    def tearDown(self) -> None:
        for output in self.outputs:
            _writable_tree(output)
        self.temporary.cleanup()

    def _build(self, name: str) -> tuple[Path, dict[str, object]]:
        output = self.root / name
        result = dict(build_package(self.repository, self.commit, self.recipe_path, output))
        self.outputs.append(output)
        return output, result

    def test_fixed_commit_is_byte_deterministic_and_ignores_crlf_worktree(self) -> None:
        first, first_result = self._build("package-one")
        # Simulate a Windows autocrlf checkout after the commit was sealed.
        (self.repository / "gpu-training" / "payload.py").write_bytes(b"VALUE = 'dirty-crlf'\r\n")
        second, second_result = self._build("package-two")
        self.assertEqual(first_result["sourceArchiveSha256"], second_result["sourceArchiveSha256"])
        self.assertEqual(
            {item.name: item.read_bytes() for item in first.iterdir()},
            {item.name: item.read_bytes() for item in second.iterdir()},
        )
        archive = next(item for item in first.iterdir() if item.name.endswith(".tar.gz"))
        with tarfile.open(archive, "r:gz") as handle:
            member = handle.getmember("source/gpu-training/payload.py")
            extracted = handle.extractfile(member)
            self.assertIsNotNone(extracted)
            self.assertEqual(extracted.read(), b"VALUE = 'committed-lf'\n")

    def test_binding_covers_ledger_recipe_and_every_source_blob(self) -> None:
        output, result = self._build("package-binding")
        verification = verify_package(output, str(result["packageManifestSha256"]))
        self.assertTrue(verification["passed"])
        binding_path = next(item for item in output.iterdir() if "source-binding" in item.name and not item.name.endswith(".sha256"))
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        self.assertEqual(binding["ledger"]["path"], "docs/ledger.md")
        self.assertEqual(binding["recipe"]["path"], self.recipe_path)
        self.assertEqual(
            [item["path"] for item in binding["sourceFiles"]],
            self.recipe["sourcePaths"],
        )
        self.assertEqual(binding["sourceCommit"], self.commit)
        self.assertEqual(
            binding["sourceInventorySha256"],
            sha256_bytes(canonical_json_bytes(binding["sourceFiles"])),
        )

    def test_stale_sidecar_and_existing_output_fail_closed(self) -> None:
        output, result = self._build("package-stale")
        with self.assertRaises(FileExistsError):
            build_package(self.repository, self.commit, self.recipe_path, output)
        archive = next(item for item in output.iterdir() if item.name.endswith(".tar.gz"))
        os.chmod(archive, 0o600)
        with archive.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "size mismatch|digest mismatch|stale"):
            verify_package(output, str(result["packageManifestSha256"]))

    def test_identical_byte_path_replacement_is_not_a_valid_snapshot(self) -> None:
        target = self.root / "snapshot.bin"
        target.write_bytes(b"same bytes do not mean same verified path")
        snapshot = stable_snapshot(target, "snapshot fixture")
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(target.read_bytes())
        os.replace(replacement, target)
        with self.assertRaisesRegex(ValueError, "changed after verification"):
            recheck_snapshot(snapshot, "snapshot fixture")

    def test_package_path_replacement_during_verification_fails_closed(self) -> None:
        output, result = self._build("package-path-race")
        manifest = output / "package-manifest.json"
        os.chmod(output, 0o700)
        os.chmod(manifest, 0o600)
        original_verify_archive = runtime._verify_archive

        def replace_after_archive_check(payload: bytes, binding: object) -> None:
            original_verify_archive(payload, binding)
            replacement = output / "replacement-manifest.json"
            replacement.write_bytes(manifest.read_bytes())
            os.replace(replacement, manifest)

        with mock.patch.object(
            runtime, "_verify_archive", side_effect=replace_after_archive_check
        ):
            with self.assertRaisesRegex(
                ValueError, "package directory|changed after verification"
            ):
                verify_package(output, str(result["packageManifestSha256"]))

    def test_package_has_no_recursive_remote_launcher(self) -> None:
        output, result = self._build("package-no-launcher")
        self.assertFalse(any(item.name.endswith("-launcher.sh") for item in output.iterdir()))
        manifest = json.loads((output / "package-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {record["role"] for record in manifest["files"]},
            {"source-archive", "source-binding", "verifier", "controller"},
        )
        self.assertTrue(verify_package(output, str(result["packageManifestSha256"]))["passed"])

    def test_remote_source_verification_rejects_unbound_source_file(self) -> None:
        output, result = self._build("package-remote-source")
        run = self.root / "remote-source-run"
        shutil.copytree(output, run / "package")
        extract_source(
            run / "package",
            str(result["packageManifestSha256"]),
            run / "source",
        )
        verification = verify_remote_package_source(
            run, str(result["packageManifestSha256"])
        )
        runtime.recheck_remote_package_source(verification)

        training = run / "source" / "gpu-training"
        os.chmod(training, 0o755)
        (training / "sitecustomize.py").write_bytes(b"raise SystemExit('unbound')\n")
        with self.assertRaisesRegex(
            ValueError, "unbound file|became writable|remains writable"
        ):
            verify_remote_package_source(run, str(result["packageManifestSha256"]))

    def test_remote_source_verification_rejects_unbound_package_file(self) -> None:
        output, result = self._build("package-remote-extra")
        run = self.root / "remote-extra-run"
        shutil.copytree(output, run / "package")
        extract_source(
            run / "package",
            str(result["packageManifestSha256"]),
            run / "source",
        )
        package = run / "package"
        os.chmod(package, 0o755)
        (package / "torch.py").write_bytes(b"raise SystemExit('unbound')\n")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            verify_remote_package_source(run, str(result["packageManifestSha256"]))

    def test_controller_uses_mandatory_manifest_trust_root_not_compiled_hashes(self) -> None:
        output, result = self._build("package-controller")
        controller = next(item for item in output.iterdir() if item.name.endswith("-controller.ps1"))
        text = controller.read_text(encoding="utf-8")
        self.assertIn("[Parameter(Mandatory = $true)] [ValidatePattern('^[0-9a-f]{64}$')] [string] $ExpectedPackageManifestSha256", text)
        self.assertIn("local package verification", text)
        self.assertIn("local sealed-source extraction", text)
        self.assertIn("'execute'", text)
        self.assertIn("'--remote-endpoint'", text)
        self.assertIn("'--behavior-actor-bundle'", text)
        self.assertIn("[string] $FrozenBaselineBundle", text)
        self.assertIn("'--frozen-baseline-bundle'", text)
        self.assertNotIn("FrozenBaselineRepository", text)
        self.assertIn("status/998-failed.json", text)
        self.assertIn("'write-status'", text)
        self.assertIn("controller failed after local run creation", text)
        self.assertNotIn("remote launcher", text.lower())
        self.assertNotIn(str(result["packageManifestSha256"]), text)
        self.assertNotIn(str(result["sourceArchiveSha256"]), text)
        self.assertNotIn(self.recipe["packageId"], text)
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if powershell is not None:
            parser = (
                "$text=[Console]::In.ReadToEnd(); $tokens=$null; $errors=$null; "
                "[System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errors)>$null; "
                "if($errors.Count){$errors|ForEach-Object{$_.Message}; exit 1}"
            )
            parsed = subprocess.run(
                [powershell, "-NoProfile", "-Command", parser],
                input=text,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(parsed.returncode, 0, parsed.stdout + parsed.stderr)

    @staticmethod
    def _decision_audit(include_players: bool) -> dict[str, object]:
        def row(**extra: object) -> dict[str, object]:
            return {
                "actorDecisions": 10,
                "actorRate": 1.0,
                "candidateDecisions": 10,
                "fallbackDecisions": 0,
                "fallbackRate": 0.0,
                **extra,
            }

        value: dict[str, object] = {
            "byAct": [row(act=act) for act in range(1, 6)],
            "overall": row(),
        }
        if include_players:
            value["byPlayerCount"] = [row(playerCount=p) for p in range(4, 11)]
        return value

    def _screening_fixture(self, root: Path, bootstrap_resamples: int) -> tuple[Path, Path]:
        candidate = root / "candidate"
        candidate.mkdir(parents=True)
        actor = candidate / "actor.pt"
        actor.write_bytes(b"actor")
        actor_sha = _write_sidecar(actor)
        candidate_manifest = candidate / "manifest.json"
        candidate_manifest.write_bytes(
            canonical_json_bytes({"files": {"actor.pt": {"sha256": actor_sha}}})
        )
        candidate_manifest_sha = _write_sidecar(candidate_manifest)
        results = []
        for player_count in range(4, 11):
            results.append(
                {
                    "actsPerMatch": 5,
                    "candidateDecisionAudit": self._decision_audit(False),
                    "matchClusters": {
                        "count": 60,
                        "sha256": f"{player_count:064x}",
                        "unit": "seed-matched-match",
                    },
                    "matches": 60,
                    "meanChipDifference": 0.5,
                    "meanChipDifference95": {"high": 0.8, "low": 0.2},
                    "meanChipDifferenceInference": {
                        "clusters": 60,
                        "high": 0.8,
                        "low": 0.2,
                        "mean": 0.5,
                        "method": "deterministic-percentile-bootstrap",
                        "resamples": bootstrap_resamples,
                        "unit": "seed-matched-match",
                    },
                    "playerCount": player_count,
                    "pairwiseCandidateBeforeNormal": {"rate": 0.6},
                }
            )
        report_value = {
            "actsPerMatch": 5,
            "bindingEvidence": {
                "actualFilesVerified": True,
                "actorBundleArtifactSha256": candidate_manifest_sha,
                "actorModelSha256": actor_sha,
            },
            "bindings": {
                "artifactSha256": candidate_manifest_sha,
                "modelSha256": actor_sha,
                "normalBaselineSha256": "a" * 64,
                "observationSchemaSha256": "b" * 64,
            },
            "candidateDecisionAudit": self._decision_audit(True),
            "candidatePolicy": {
                "actorCount": 1,
                "bundleActorSha256s": [actor_sha],
                "bundleArtifactSha256": candidate_manifest_sha,
                "bundleManifestSha256s": [candidate_manifest_sha],
                "compileAutomaticFallback": False,
                "policyNumerics": canonical_v4_policy_numerics_contract(),
                "routing": {"mode": "pure-actor", "runtimeErrorFallback": False},
            },
            "deploymentTriggered": False,
            "evaluationMode": "screening",
            "format": "dalmuti-model-benchmark",
            "matchCountsByPlayerCount": {str(p): 60 for p in range(4, 11)},
            "modelSha256": actor_sha,
            "playerCounts": list(range(4, 11)),
            "results": results,
            "seed": 450000001,
            "seedFamily": {
                "id": "attempt004-screening-seed450000001",
                "mode": "screening",
            },
            "version": 2,
        }
        report = root / f"screening-{bootstrap_resamples}.json"
        report.write_bytes(canonical_json_bytes(report_value))
        _write_sidecar(report)
        return report, candidate

    def _semantic_seal_fixture(
        self, run: Path
    ) -> tuple[Path, Path, Path, str, str, dict[str, object]]:
        status = run / "status"
        status.mkdir(parents=True)
        scratch = run / "scratch"
        report_source, candidate_source = self._screening_fixture(scratch, 10000)
        screening_dir = run / "screening"
        screening_dir.mkdir()
        screening = screening_dir / "epoch-0001.json"
        screening.write_bytes(report_source.read_bytes())
        screening_sha = _write_sidecar(screening)
        candidate = (
            run
            / "training"
            / "train-seed-610000001-run-001"
            / "candidate"
        )
        candidate.parent.mkdir(parents=True)
        shutil.copytree(candidate_source, candidate)
        actor_sha = sha256_bytes((candidate / "actor.pt").read_bytes())
        manifest_sha = sha256_bytes((candidate / "manifest.json").read_bytes())
        hard_gate = run / "training" / "epoch-0001-hard-gates.json"
        hard_gate.write_bytes(
            canonical_json_bytes(
                {
                    "candidateActorSha256": actor_sha,
                    "candidateManifestSha256": manifest_sha,
                    "format": "dalmuti-v4-mixed-training-hard-gates",
                    "passed": True,
                    "version": 1,
                }
            )
        )
        _write_sidecar(hard_gate)
        promotion = screening_dir / "epoch-0001-promotion-gates.json"
        promotion.write_bytes(
            canonical_json_bytes(
                {
                    "allPlayerCountsPassed": True,
                    "format": "dalmuti-v4-mixed-promotion-gates",
                    "gates": {
                        "minimumClustered95LowerBound": 0.15,
                        "minimumMeanChipDifferencePerAct": 0.25,
                        "minimumPairwiseBeforeNormal": 0.55,
                    },
                    "passed": True,
                    "perPlayerCount": {
                        str(player): {
                            "clustered95LowerBound": 0.2,
                            "meanChipDifferencePerAct": 0.5,
                            "pairwiseBeforeNormal": 0.6,
                            "passed": True,
                        }
                        for player in range(4, 11)
                    },
                    "screeningReportSha256": screening_sha,
                    "version": 1,
                }
            )
        )
        promotion_sha = _write_sidecar(promotion)
        shutil.rmtree(scratch)
        recipe = {"runContract": {"fixture": "sealed"}}
        recipe_path = run / "source" / "gpu-training" / "v4_mixed_execution_recipe.json"
        recipe_path.parent.mkdir(parents=True)
        recipe_path.write_bytes(canonical_json_bytes(recipe))
        recipe_sha = sha256_bytes(recipe_path.read_bytes())
        run_contract_sha = sha256_bytes(canonical_json_bytes(recipe["runContract"]))
        package_manifest = run / "package" / "package-manifest.json"
        package_manifest.parent.mkdir()
        package_manifest.write_bytes(canonical_json_bytes({"fixture": "package"}))
        package_sha = sha256_bytes(package_manifest.read_bytes())
        runtime_bindings = run / "control" / "runtime-bindings.json"
        runtime_bindings.parent.mkdir()
        runtime_bindings.write_bytes(
            canonical_json_bytes(
                {
                    "packageManifestSha256": package_sha,
                    "recipeSha256": recipe_sha,
                }
            )
        )
        runtime_sha = _write_sidecar(runtime_bindings)
        finalization_audit = run / "provenance" / "finalization-audit.json"
        finalization_audit.parent.mkdir()
        finalization_audit.write_bytes(canonical_json_bytes({"fixture": "audit"}))
        finalization_sha = _write_sidecar(finalization_audit)
        return (
            status,
            screening,
            promotion,
            screening_sha,
            promotion_sha,
            {
                "package_manifest_sha256": package_sha,
                "recipe_sha256": recipe_sha,
                "run_contract_sha256": run_contract_sha,
                "runtime_bindings_sha256": runtime_sha,
                "finalization_audit": finalization_audit,
                "finalization_audit_sha256": finalization_sha,
            },
        )

    def test_screening_verifier_requires_full_p4_p10_and_10000_bootstraps(self) -> None:
        output, result = self._build("package-screening")
        source_root = self.root / "extracted-source"
        extract_source(output, str(result["packageManifestSha256"]), source_root)
        valid_root = self.root / "valid-screen"
        report, candidate = self._screening_fixture(valid_root, 10000)
        verified = verify_screening(
            output,
            str(result["packageManifestSha256"]),
            source_root,
            report,
            candidate,
        )
        self.assertTrue(verified["passed"])
        for case, mutate in (
            (
                "missing",
                lambda policy: policy.pop("policyNumerics"),
            ),
            (
                "modified",
                lambda policy: policy["policyNumerics"].__setitem__(
                    "mathSdpEnabled", False
                ),
            ),
        ):
            drift_root = self.root / f"{case}-policy-numerics"
            drift_report, drift_candidate = self._screening_fixture(
                drift_root, 10000
            )
            drift_value = json.loads(drift_report.read_text(encoding="utf-8"))
            policy = drift_value["candidatePolicy"]
            self.assertIsInstance(policy, dict)
            mutate(policy)
            drift_report.write_bytes(canonical_json_bytes(drift_value))
            _write_sidecar(drift_report)
            with self.subTest(case=case), self.assertRaisesRegex(
                ValueError, "screening policy numerics"
            ):
                verify_screening(
                    output,
                    str(result["packageManifestSha256"]),
                    source_root,
                    drift_report,
                    drift_candidate,
                )
        invalid_root = self.root / "invalid-screen"
        invalid_report, invalid_candidate = self._screening_fixture(invalid_root, 9999)
        with self.assertRaisesRegex(ValueError, "bootstrap contract mismatch"):
            verify_screening(
                output,
                str(result["packageManifestSha256"]),
                source_root,
                invalid_report,
                invalid_candidate,
            )

    def test_success_status_is_impossible_before_seal(self) -> None:
        run = self.root / "run"
        status = run / "status"
        status.mkdir(parents=True)
        (run / "artifact.bin").write_bytes(b"preserved")
        with self.assertRaisesRegex(ValueError, "requires a completed run seal"):
            write_status(status / "premature.json", "complete", "succeeded", "bad", None)
        seal_path = run / "provenance" / "final-files.json"
        seal_run(run, seal_path, status)
        sealed_artifact = run / "artifact.bin"
        os.chmod(sealed_artifact, 0o600)
        sealed_artifact.write_bytes(b"tampered-after-seal")
        with self.assertRaisesRegex(ValueError, "sealed run (file remains writable|size mismatch|digest mismatch)"):
            write_status(
                status / "tampered-success.json",
                "complete",
                "succeeded",
                "must fail",
                seal_path,
            )
        sealed_artifact.write_bytes(b"preserved")
        os.chmod(sealed_artifact, 0o444)
        value = write_status(
            status / "999-succeeded.json",
            "complete",
            "succeeded",
            "sealed",
            seal_path,
        )
        self.assertEqual(value["runSealSha256"], sha256_bytes(seal_path.read_bytes()))

    def test_status_pair_publication_rolls_back_if_sidecar_creation_fails(self) -> None:
        status = self.root / "atomic-status" / "998-failed.json"
        sidecar = Path(f"{status}.sha256")
        original_open = os.open

        def fail_sidecar(path: object, flags: int, mode: int = 0o777) -> int:
            if Path(path) == sidecar:
                raise OSError("injected sidecar failure")
            return original_open(path, flags, mode)

        with mock.patch.object(os, "open", side_effect=fail_sidecar):
            with self.assertRaisesRegex(OSError, "injected sidecar failure"):
                write_status(
                    status, "workflow", "failed", "injected failure", None
                )
        self.assertFalse(status.exists())
        self.assertFalse(sidecar.exists())
        self.assertFalse((status.parent / ".terminal-status.lock").exists())

    def test_failed_and_succeeded_terminal_statuses_are_mutually_exclusive(self) -> None:
        for first_state, second_state in (("failed", "succeeded"), ("succeeded", "failed")):
            run = self.root / f"terminal-{first_state}"
            status = run / "status"
            status.mkdir(parents=True)
            (run / "artifact.bin").write_bytes(b"sealed")
            seal_path = run / "provenance" / "final-files.json"
            seal_run(run, seal_path, status)
            first = status / f"first-{first_state}.json"
            write_status(
                first,
                "complete",
                first_state,
                "first terminal",
                seal_path if first_state == "succeeded" else None,
            )
            second = status / f"second-{second_state}.json"
            with self.assertRaisesRegex(ValueError, "terminal status is already"):
                write_status(
                    second,
                    "complete",
                    second_state,
                    "second terminal",
                    seal_path if second_state == "succeeded" else None,
                )
            self.assertFalse(second.exists())
            self.assertFalse(Path(f"{second}.sha256").exists())

    def test_remote_seal_binds_promotion_to_exact_screening_bytes(self) -> None:
        run = self.root / "semantic-seal-run"
        status, screening, promotion, screening_sha, promotion_sha, bindings = (
            self._semantic_seal_fixture(run)
        )
        seal_path = run / "provenance" / "final-files.json"
        with mock.patch.object(
            runtime, "_verify_finalization_audit", return_value=({}, {})
        ), mock.patch.object(
            runtime, "_verify_remote_semantic_inventory", return_value={}
        ), mock.patch.object(
            runtime,
            "verify_remote_package_source",
            return_value=mock.Mock(package_snapshots={}, source_snapshots={}),
        ), mock.patch.object(
            runtime, "recheck_remote_package_source"
        ):
            sealed = seal_run(
                run,
                seal_path,
                status,
                screening,
                promotion,
                **bindings,
                profile="remote-semantic",
            )
        self.assertEqual(sealed["screeningReportSha256"], screening_sha)
        self.assertEqual(sealed["promotionReportSha256"], promotion_sha)

    def test_remote_seal_rejects_screening_swap_after_promotion(self) -> None:
        run = self.root / "semantic-swap-run"
        status, screening, promotion, _, _, bindings = self._semantic_seal_fixture(run)
        value = json.loads(screening.read_text(encoding="utf-8"))
        value["seed"] += 1
        screening.write_bytes(canonical_json_bytes(value))
        _write_sidecar(screening)
        with mock.patch.object(
            runtime, "_verify_finalization_audit", return_value=({}, {})
        ), mock.patch.object(
            runtime, "_verify_remote_semantic_inventory", return_value={}
        ), mock.patch.object(
            runtime,
            "verify_remote_package_source",
            return_value=mock.Mock(package_snapshots={}, source_snapshots={}),
        ), mock.patch.object(
            runtime, "recheck_remote_package_source"
        ):
            with self.assertRaisesRegex(ValueError, "different screening bytes"):
                seal_run(
                    run,
                    run / "provenance" / "final-files.json",
                    status,
                    screening,
                    promotion,
                    **bindings,
                    profile="remote-semantic",
                )

    def test_seal_makes_every_non_status_directory_0555(self) -> None:
        run = self.root / "mode-run"
        status = run / "status"
        nested = run / "artifacts" / "nested"
        nested.mkdir(parents=True)
        status.mkdir()
        (nested / "artifact.bin").write_bytes(b"preserved")
        seal_path = run / "provenance" / "final-files.json"
        seal_run(run, seal_path, status)
        for directory in (run, run / "artifacts", nested, run / "provenance"):
            self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o555)
        self.assertNotEqual(stat.S_IMODE(status.stat().st_mode), 0o555)

    def test_success_status_is_removed_if_tree_changes_after_initial_check(self) -> None:
        run = self.root / "post-check-race"
        status = run / "status"
        status.mkdir(parents=True)
        artifact = run / "artifact.bin"
        artifact.write_bytes(b"preserved")
        seal_path = run / "provenance" / "final-files.json"
        seal_run(run, seal_path, status)
        success = status / "999-succeeded.json"
        original_verify_run_seal = runtime.verify_run_seal
        calls = 0

        def mutate_before_second_verification(path: Path) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                os.chmod(artifact, 0o600)
                artifact.write_bytes(b"changed after initial success check")
            return original_verify_run_seal(path)

        with mock.patch.object(
            runtime,
            "verify_run_seal",
            side_effect=mutate_before_second_verification,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "sealed run (file remains writable|size mismatch|digest mismatch)",
            ):
                write_status(
                    success,
                    "complete",
                    "succeeded",
                    "must be rolled back",
                    seal_path,
                )
        self.assertFalse(success.exists())
        self.assertFalse(Path(f"{success}.sha256").exists())

    def test_success_status_rejects_writable_or_extended_sealed_tree(self) -> None:
        run = self.root / "tree-mutation-run"
        status = run / "status"
        nested = run / "artifacts"
        nested.mkdir(parents=True)
        status.mkdir()
        (nested / "artifact.bin").write_bytes(b"preserved")
        seal_path = run / "provenance" / "final-files.json"
        seal_run(run, seal_path, status)
        success = status / "999-succeeded.json"
        os.chmod(nested, 0o755)
        with self.assertRaisesRegex(ValueError, "sealed directory mode is not 0555"):
            write_status(success, "complete", "succeeded", "must fail", seal_path)
        self.assertFalse(success.exists())
        os.chmod(nested, 0o755)
        extra = nested / "unsealed.bin"
        extra.write_bytes(b"new artifact")
        os.chmod(nested, 0o555)
        with self.assertRaisesRegex(ValueError, "unbound file"):
            write_status(success, "complete", "succeeded", "must fail", seal_path)
        self.assertFalse(success.exists())


if __name__ == "__main__":
    unittest.main()
