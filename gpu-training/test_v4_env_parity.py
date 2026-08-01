from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

try:
    import torch  # noqa: F401 - verifies the environment dependency up front.
except ModuleNotFoundError as error:
    raise unittest.SkipTest("torch is required for V4 parity tests") from error

from v4_verify_env_parity import ParityError, verify_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = REPOSITORY_ROOT / "scripts" / "rl-generate-v4-env-parity-fixtures.mjs"


def _node_binary() -> str | None:
    configured = os.environ.get("NODE_BINARY")
    candidate = configured or shutil.which("node")
    if candidate is None:
        return None
    try:
        version = subprocess.check_output(
            [candidate, "--version"], text=True, timeout=10
        ).strip()
        major, minor, *_ = (int(value) for value in version.lstrip("v").split("."))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return candidate if (major, minor) >= (22, 13) else None


class V4EnvironmentParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        node = _node_binary()
        if node is None:
            raise unittest.SkipTest("Node.js is required to generate parity fixtures")
        cls.node = node

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="dalmuti-v4-parity-")
        self.directory = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _generate(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                self.node,
                "--experimental-strip-types",
                str(GENERATOR),
                "--players",
                "4",
                "--acts",
                "1",
                "--seeds-per-player",
                "1",
                "--seed-base",
                "701",
                "--allow-small-test-fixture",
                "--output",
                str(output),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_small_fixture_is_deterministic_exclusive_and_verifies(self) -> None:
        first = self.directory / "first.ndjson"
        second = self.directory / "second.ndjson"
        generated = self._generate(first)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        regenerated = self._generate(second)
        self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(
            Path(f"{first}.sha256").read_text(encoding="ascii"),
            Path(f"{second}.sha256").read_text(encoding="ascii"),
        )

        result = verify_fixture(first, repository_root=REPOSITORY_ROOT)
        self.assertEqual(result["matches"], 1)
        self.assertGreater(result["decisions"], 0)
        fixture_records = [
            json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()
        ]
        public_events = [
            event
            for record in fixture_records
            if record["type"] == "decision"
            for event in record["eventsAfterAction"]
        ]
        self.assertEqual(
            sum(
                event["type"] == "pass" and event.get("reason") == "dalmuti"
                for event in public_events
            ),
            3,
        )
        self.assertGreaterEqual(
            sum(
                event["type"] == "pass"
                and event.get("reason") == "insufficient-cards"
                for event in public_events
            ),
            1,
        )

        rejected = self._generate(first)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("already exists", rejected.stderr)
        self.assertFalse(Path(f"{first}.partial").exists())

    def test_first_decision_mask_corruption_is_rejected_at_exact_location(self) -> None:
        source = self.directory / "source.ndjson"
        generated = self._generate(source)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        records = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
        ]
        decision = next(record for record in records if record["type"] == "decision")
        mask = decision["legalMaskHex"]
        replacement = "f" if mask[0] != "f" else "e"
        decision["legalMaskHex"] = replacement + mask[1:]

        corrupted = self.directory / "corrupted.ndjson"
        corrupted_bytes = "".join(
            f"{json.dumps(record, separators=(',', ':'), ensure_ascii=False)}\n"
            for record in records
        ).encode("utf-8")
        corrupted.write_bytes(corrupted_bytes)
        Path(f"{corrupted}.sha256").write_text(
            f"{sha256(corrupted_bytes).hexdigest()}\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(
            ParityError,
            r"p4-seed-4000701\.act-1\.decision-0\.legalMaskHex",
        ):
            verify_fixture(corrupted, repository_root=REPOSITORY_ROOT)


if __name__ == "__main__":
    unittest.main()
