from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from v5_contract import V5_PUBLIC_CONTRACT_SHA256
from v5_export import (
    export_v5_actor_bundle,
    load_v5_actor_bundle,
    load_v5_actor_checkpoint,
    sha256_file,
    tensor_state_sha256,
    v5_actor_bundle_digests,
    verify_v5_actor_bundle,
)
from v5_model import V5ActorConfig, V5PublicActor


def _small_actor() -> V5PublicActor:
    torch.manual_seed(1234)
    return V5PublicActor(V5ActorConfig(
        history_latents=2,
        d_model=32,
        core_layers=1,
        heads=4,
        feedforward=64,
    ))


class V5ExportTests(unittest.TestCase):
    def test_actor_only_bundle_round_trip_and_exact_state_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "actor-bundle"
            actor = _small_actor()
            manifest = export_v5_actor_bundle(
                actor, target, metadata={"trainingRun": "unit-test"}
            )
            self.assertEqual(manifest, verify_v5_actor_bundle(target))
            loaded, loaded_manifest = load_v5_actor_bundle(target)
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(
                tensor_state_sha256(actor.state_dict()),
                tensor_state_sha256(loaded.state_dict()),
            )
            self.assertFalse(manifest["model"]["criticIncluded"])
            self.assertTrue(manifest["model"]["criticExcluded"])
            self.assertFalse(manifest["criticIncluded"])
            self.assertEqual(
                manifest["actorPtSha256"], sha256_file(target / "actor.pt")
            )
            self.assertEqual(
                manifest["publicContractSha256"], V5_PUBLIC_CONTRACT_SHA256
            )
            digests = v5_actor_bundle_digests(target)
            self.assertEqual(
                digests["actorSha256"], sha256_file(target / "actor.pt")
            )
            self.assertEqual(
                digests["manifestSha256"], sha256_file(target / "manifest.json")
            )
            self.assertEqual(
                set(path.name for path in target.iterdir()),
                {
                    "actor.pt", "config.json", "public-contract.json",
                    "manifest.json", "manifest.json.sha256",
                },
            )
            payload = torch.load(target / "actor.pt", weights_only=True)
            self.assertFalse(any(
                token in name.lower()
                for name in payload["stateDict"]
                for token in ("critic", "privileged", "private")
            ))

    def test_publish_is_exclusive_and_bundle_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "actor-bundle"
            actor = _small_actor()
            export_v5_actor_bundle(actor, target)
            with self.assertRaises(FileExistsError):
                export_v5_actor_bundle(actor, target)
            actor_path = target / "actor.pt"
            raw = bytearray(actor_path.read_bytes())
            raw[-1] ^= 1
            actor_path.write_bytes(raw)
            with self.assertRaises(ValueError):
                verify_v5_actor_bundle(target)

    def test_checkpoint_rejects_private_or_critic_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "actor-bundle"
            export_v5_actor_bundle(_small_actor(), target)
            payload = torch.load(target / "actor.pt", weights_only=True)
            payload["stateDict"]["privileged_critic.weight"] = torch.zeros(1)
            malicious = Path(temporary) / "malicious.pt"
            torch.save(payload, malicious)
            with self.assertRaises(ValueError):
                load_v5_actor_checkpoint(malicious)

    def test_config_and_public_contract_are_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "actor-bundle"
            export_v5_actor_bundle(_small_actor(), target)
            for name in ("config.json", "public-contract.json", "manifest.json"):
                raw = (target / name).read_bytes()
                value = json.loads(raw.decode("ascii"))
                expected = json.dumps(
                    value, ensure_ascii=True, sort_keys=True,
                    separators=(",", ":"), allow_nan=False,
                ).encode("ascii")
                self.assertEqual(raw, expected)


if __name__ == "__main__":
    unittest.main()
