from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch

from v3_action_conditioned import (
    V3_ACTION_CATALOGUE,
    V3_ACTION_CATALOGUE_VERSION,
)
from v4_export import (
    V4_ACTOR_FORMAT_VERSION,
    V4_LEGACY_ACTOR_FORMAT_VERSION,
    V4_LEGACY_MANIFEST_VERSION,
    V4_MANIFEST_VERSION,
    canonical_json_bytes,
    export_v4_actor_bundle,
    load_v4_actor_checkpoint,
    sha256_file,
    verify_v4_actor_bundle,
)
from v4_model import (
    V4_ACTION_COUNT,
    V4ActorConfig,
    V4CenteredLogitEnsemble,
    V4PublicActor,
)


def _config() -> V4ActorConfig:
    return V4ActorConfig(
        max_players=4,
        max_history=3,
        d_model=16,
        layers=1,
        heads=4,
        feedforward=32,
        action_hidden=12,
    )


def _canonical_catalogue_sha256() -> str:
    payload = json.dumps(
        {
            "version": V3_ACTION_CATALOGUE_VERSION,
            "catalogue": [dict(action) for action in V3_ACTION_CATALOGUE],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_catalogue_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(
        [dict(action) for action in V3_ACTION_CATALOGUE]
    )).hexdigest()


def _read_manifest(bundle: Path) -> dict[str, object]:
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(bundle: Path, manifest: dict[str, object]) -> None:
    path = bundle / "manifest.json"
    path.write_bytes(canonical_json_bytes(manifest))
    digest = sha256_file(path)
    (bundle / "manifest.json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )


def _load_payload(bundle: Path) -> dict[str, object]:
    return torch.load(bundle / "actor.pt", map_location="cpu", weights_only=False)


def _write_payload(
    bundle: Path, payload: dict[str, object], manifest: dict[str, object]
) -> None:
    actor = bundle / "actor.pt"
    torch.save(payload, actor)
    files = manifest["files"]
    assert isinstance(files, dict)
    files["actor.pt"] = {
        "sha256": sha256_file(actor),
        "bytes": actor.stat().st_size,
    }


def _semantic_sha(
    kind: str,
    config: dict[str, object],
    seeds: list[int] | None,
    public: dict[str, object],
    action: dict[str, object],
) -> str:
    return hashlib.sha256(canonical_json_bytes({
        "version": 1,
        "kind": kind,
        "config": config,
        "seeds": seeds,
        "publicInputContract": public,
        "actionSpace": action,
    })).hexdigest()


class V4ExportContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _export_actor(self, name: str = "actor") -> Path:
        bundle = self.root / name
        torch.manual_seed(20260801)
        export_v4_actor_bundle(
            V4PublicActor(_config()).eval(),
            bundle,
            metadata={"seed": 20260801, "labels": ("public", "actor")},
        )
        return bundle

    def test_v2_bundle_uses_versioned_catalogue_and_cross_bound_payload(self) -> None:
        bundle = self._export_actor()
        manifest = verify_v4_actor_bundle(bundle)
        _, payload = load_v4_actor_checkpoint(bundle / "actor.pt")

        self.assertEqual(manifest["version"], V4_MANIFEST_VERSION)
        self.assertEqual(payload["version"], V4_ACTOR_FORMAT_VERSION)
        self.assertEqual(
            manifest["actionSpace"],
            {
                "catalogueVersion": V3_ACTION_CATALOGUE_VERSION,
                "count": V4_ACTION_COUNT,
                "catalogueSha256": _canonical_catalogue_sha256(),
            },
        )
        self.assertNotEqual(
            manifest["actionSpace"]["catalogueSha256"],
            _legacy_catalogue_sha256(),
        )
        self.assertEqual(payload["actionSpace"], manifest["actionSpace"])
        self.assertEqual(
            payload["publicInputContract"], manifest["publicInputContract"]
        )
        self.assertEqual(
            payload["semanticContractSha256"],
            manifest["model"]["payloadSemanticContractSha256"],
        )
        self.assertEqual(payload["metadata"], manifest["metadata"])
        self.assertEqual(manifest["metadata"]["labels"], ["public", "actor"])

    def test_explicit_v1_legacy_boundary_accepts_real_export_shape_only(self) -> None:
        bundle = self._export_actor("legacy")
        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        payload["version"] = V4_LEGACY_ACTOR_FORMAT_VERSION
        for name in (
            "publicInputContract", "actionSpace", "semanticContractSha256",
        ):
            del payload[name]
        manifest["version"] = V4_LEGACY_MANIFEST_VERSION
        model = manifest["model"]
        assert isinstance(model, dict)
        model["formatVersion"] = V4_LEGACY_ACTOR_FORMAT_VERSION
        del model["payloadSemanticContractSha256"]
        action = manifest["actionSpace"]
        assert isinstance(action, dict)
        action["catalogueSha256"] = _legacy_catalogue_sha256()
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)

        verified = verify_v4_actor_bundle(bundle)
        loaded, loaded_payload = load_v4_actor_checkpoint(bundle / "actor.pt")
        self.assertEqual(verified["version"], V4_LEGACY_MANIFEST_VERSION)
        self.assertEqual(loaded_payload["version"], V4_LEGACY_ACTOR_FORMAT_VERSION)
        self.assertIsInstance(loaded, V4PublicActor)

        canonicalized_legacy = copy.deepcopy(manifest)
        canonicalized_legacy["actionSpace"]["catalogueSha256"] = (
            _canonical_catalogue_sha256()
        )
        _write_manifest(bundle, canonicalized_legacy)
        with self.assertRaisesRegex(ValueError, "catalogue contract"):
            verify_v4_actor_bundle(bundle)

    def test_consistently_rehashed_observation_contract_tamper_is_rejected(self) -> None:
        bundle = self._export_actor("semantic-tamper")
        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        payload_config = copy.deepcopy(payload["config"])
        payload_config["observation_schema_version"] = 3
        payload["config"] = payload_config
        model = manifest["model"]
        assert isinstance(model, dict)
        model["config"] = copy.deepcopy(payload_config)
        semantic_sha = _semantic_sha(
            str(payload["kind"]),
            payload_config,
            payload["seeds"],
            payload["publicInputContract"],
            payload["actionSpace"],
        )
        payload["semanticContractSha256"] = semantic_sha
        model["payloadSemanticContractSha256"] = semantic_sha
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)

        with self.assertRaisesRegex(ValueError, "observation_schema_version"):
            verify_v4_actor_bundle(bundle)

    def test_manifest_payload_identity_and_current_action_space_are_exact(self) -> None:
        bundle = self._export_actor("identity")
        manifest = _read_manifest(bundle)
        model = manifest["model"]
        assert isinstance(model, dict)
        model["seeds"] = [41, 43, 47]
        model["kind"] = "centered-logit-ensemble"
        model["ensembleRule"] = "mean of per-actor logits centered over legal actions"
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_v4_actor_bundle(bundle)

        bundle = self._export_actor("action-tamper")
        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        fake_action = copy.deepcopy(payload["actionSpace"])
        fake_action["catalogueSha256"] = "0" * 64
        payload["actionSpace"] = fake_action
        model = manifest["model"]
        assert isinstance(model, dict)
        semantic_sha = _semantic_sha(
            str(payload["kind"]), payload["config"], payload["seeds"],
            payload["publicInputContract"], fake_action,
        )
        payload["semanticContractSha256"] = semantic_sha
        model["payloadSemanticContractSha256"] = semantic_sha
        manifest["actionSpace"] = fake_action
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "action-space contract"):
            verify_v4_actor_bundle(bundle)

    def test_ensemble_seeds_and_extra_payload_fields_fail_closed(self) -> None:
        bundle = self.root / "ensemble"
        export_v4_actor_bundle(
            V4CenteredLogitEnsemble.from_seeds(_config(), (41, 43, 47)).eval(),
            bundle,
        )
        verified = verify_v4_actor_bundle(bundle)
        self.assertEqual(verified["model"]["seeds"], [41, 43, 47])

        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        payload["unexpected"] = "consistently rehashed but unsupported"
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "fields"):
            verify_v4_actor_bundle(bundle)

    def test_versions_and_action_feature_buffer_are_fail_closed(self) -> None:
        bundle = self._export_actor("bool-version")
        manifest = _read_manifest(bundle)
        manifest["version"] = True
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            verify_v4_actor_bundle(bundle)

        bundle = self._export_actor("payload-version")
        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        payload["version"] = 2.0
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            verify_v4_actor_bundle(bundle)

        bundle = self._export_actor("action-features")
        manifest = _read_manifest(bundle)
        payload = _load_payload(bundle)
        state_dict = payload["stateDict"]
        assert isinstance(state_dict, dict)
        state_dict["action_features"] = torch.zeros_like(
            state_dict["action_features"]
        )
        _write_payload(bundle, payload, manifest)
        _write_manifest(bundle, manifest)
        with self.assertRaisesRegex(ValueError, "action features"):
            verify_v4_actor_bundle(bundle)

    def test_stale_or_untracked_onnx_cannot_survive_bundle_reexport(self) -> None:
        bundle = self._export_actor("stale-onnx")
        onnx_path = bundle / "actor.onnx"
        onnx_path.write_bytes(b"stale")
        export_v4_actor_bundle(
            V4PublicActor(_config()).eval(), bundle, include_onnx=False
        )
        self.assertFalse(onnx_path.exists())
        verify_v4_actor_bundle(bundle)

        onnx_path.write_bytes(b"untracked")
        with self.assertRaisesRegex(ValueError, "untracked"):
            verify_v4_actor_bundle(bundle)

    @unittest.skipUnless(
        importlib.util.find_spec("onnx") is not None,
        "optional ONNX dependency is unavailable",
    )
    def test_rehashed_swapped_onnx_is_not_bound_to_actor_checkpoint(self) -> None:
        first = self.root / "onnx-first"
        second = self.root / "onnx-second"
        torch.manual_seed(101)
        export_v4_actor_bundle(
            V4PublicActor(_config()).eval(), first, include_onnx=True
        )
        torch.manual_seed(103)
        export_v4_actor_bundle(
            V4PublicActor(_config()).eval(), second, include_onnx=True
        )
        shutil.copyfile(second / "actor.onnx", first / "actor.onnx")
        manifest = _read_manifest(first)
        files = manifest["files"]
        assert isinstance(files, dict)
        files["actor.onnx"] = {
            "sha256": sha256_file(first / "actor.onnx"),
            "bytes": (first / "actor.onnx").stat().st_size,
        }
        _write_manifest(first, manifest)

        with self.assertRaisesRegex(ValueError, "not bound"):
            verify_v4_actor_bundle(first)


if __name__ == "__main__":
    unittest.main()
