from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from v3_ppo_result_contract import load_source_contract


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_PATH_POLICY = {
    "bundleRoot": ".",
    "behaviorModel": "behavior-model.json",
    "dataRoot": "data",
    "outputRoot": "models",
    "resultsRoot": "returned",
    "requireFreshRunDirectories": True,
    "requireMatchingRunIds": True,
    "requireDisjointPaths": True,
    "protectBundleInputs": True,
    "rejectSymbolicLinks": True,
}
EXPECTED_DETERMINISM = {
    "required": True,
    "pythonDontWriteBytecode": True,
    "torchDeterministicAlgorithms": True,
    "warnOnly": False,
    "cublasWorkspaceConfig": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    "cudnnDeterministic": True,
    "cudnnBenchmark": False,
    "cudaMatmulAllowTf32": False,
    "cudnnAllowTf32": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify, train, and package one strict V3 PPO update."
    )
    parser.add_argument("--data", nargs="+", default=["data/*.ndjson"])
    parser.add_argument("--behavior-model", default="behavior-model.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument(
        "--skip-forced-policy-time",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--terminal-rank-auxiliary-coefficient", type=float, default=0.0
    )
    parser.add_argument("--rollout-temperature", type=float, required=True)
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-gradient-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--binding-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--behavior-binding-batch-size", type=int, default=8192
    )
    parser.add_argument("--loader-workers", type=int, default=7)
    parser.add_argument("--seed", type=int, default=202608061)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def is_contained_by(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def paths_overlap(left: Path, right: Path) -> bool:
    return is_contained_by(left, right) or is_contained_by(right, left)


def assert_no_symlink_components(root: Path, path: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the verified bundle root: {path}") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link: {current}")


def assert_regular_bundle_file(root: Path, path: Path, label: str) -> None:
    assert_no_symlink_components(root, path, label)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular bundle file: {path}")


def relative_bundle_argument(root: Path, value: str, label: str) -> Path:
    supplied = Path(value)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError(f"{label} must be a relative path inside the bundle")
    raw = root / supplied
    current = root
    for part in supplied.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symbolic link: {current}")
    resolved = raw.resolve(strict=False)
    if not is_contained_by(resolved, root):
        raise ValueError(f"{label} escapes the verified bundle root")
    return resolved


def resolve_fresh_run_paths(
    root: Path,
    output_value: str,
    results_value: str,
    path_policy: dict,
) -> tuple[str, Path, Path]:
    if path_policy != EXPECTED_PATH_POLICY:
        raise ValueError("gpu-run-config pathPolicy does not match the runner contract")
    output = relative_bundle_argument(root, output_value, "V3 output")
    results = relative_bundle_argument(root, results_value, "V3 results")
    output_root = (root / path_policy["outputRoot"]).resolve(strict=False)
    results_root = (root / path_policy["resultsRoot"]).resolve(strict=False)
    for label, allowed_root in (
        ("V3 output root", output_root),
        ("V3 results root", results_root),
    ):
        assert_no_symlink_components(root, allowed_root, label)
        if allowed_root.exists() and not allowed_root.is_dir():
            raise NotADirectoryError(f"{label} is not a directory: {allowed_root}")
    try:
        output_relative = output.relative_to(output_root)
        results_relative = results.relative_to(results_root)
    except ValueError as error:
        raise ValueError(
            "V3 output and results must stay inside their bundle-approved roots"
        ) from error
    if len(output_relative.parts) != 1 or len(results_relative.parts) != 1:
        raise ValueError(
            "V3 output and results must be direct run directories under "
            "models/ and returned/"
        )
    run_id = output_relative.name
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid V3 run ID: {run_id!r}")
    if results_relative.name != run_id:
        raise ValueError("V3 output and results must use the same run ID")
    if paths_overlap(output, results):
        raise ValueError("V3 output and results paths must be disjoint")
    assert_no_symlink_components(root, output, "V3 output")
    assert_no_symlink_components(root, results, "V3 results")
    if output.exists():
        raise FileExistsError(f"V3 run output must not already exist: {output}")
    if results.exists():
        raise FileExistsError(
            f"V3 run results directory must not already exist: {results}"
        )
    return run_id, output, results


def expected_algorithm(args: argparse.Namespace) -> dict[str, object]:
    return {
        "epochs": args.epochs,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "gamma": args.gamma,
        "gaeLambda": args.gae_lambda,
        "skipForcedPolicyTime": args.skip_forced_policy_time,
        "rolloutTemperature": args.rollout_temperature,
        "clipCoefficient": args.clip_coefficient,
        "valueCoefficient": args.value_coefficient,
        "entropyCoefficient": args.entropy_coefficient,
        "maxGradientNorm": args.max_gradient_norm,
        "targetKl": args.target_kl,
        "bindingTolerance": args.binding_tolerance,
        "behaviorBindingBatchSize": args.behavior_binding_batch_size,
        "loaderWorkers": args.loader_workers,
        "device": args.device,
        "seed": args.seed,
    }


def validate_run_config(config: dict, args: argparse.Namespace) -> None:
    if (
        config.get("format") != "dalmuti-v3-ppo-gpu-run-config"
        or config.get("version") != 2
    ):
        raise ValueError("unsupported V3 GPU run config")
    if config.get("algorithm") != expected_algorithm(args):
        raise ValueError(
            "command-line algorithm arguments do not exactly match "
            "gpu-run-config.json"
        )
    allowed = config.get("allowedTerminalRankAuxiliaryCoefficients")
    if not isinstance(allowed, list) or args.terminal_rank_auxiliary_coefficient not in allowed:
        raise ValueError(
            "terminal rank auxiliary coefficient is not an approved A/B variant"
        )
    if config.get("determinism") != EXPECTED_DETERMINISM:
        raise ValueError("gpu-run-config deterministic contract is invalid")
    if config.get("pathPolicy") != EXPECTED_PATH_POLICY:
        raise ValueError("gpu-run-config path policy is invalid")


def manifest_file_entries(manifest: dict) -> dict[str, dict]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("bundle manifest files must be an array")
    entries: dict[str, dict] = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("bundle manifest contains an invalid file entry")
        relative = entry["path"]
        if relative in entries:
            raise ValueError(f"duplicate bundle manifest path: {relative}")
        entries[relative] = entry
    return entries


def verify_manifest_bound_file(
    root: Path,
    path: Path,
    relative: str,
    entries: dict[str, dict],
    label: str,
) -> None:
    assert_regular_bundle_file(root, path, label)
    entry = entries.get(relative)
    if entry is None:
        raise ValueError(f"{label} is not approved by bundle-manifest.json")
    if path.stat().st_size != entry.get("bytes") or file_sha256(path) != entry.get(
        "sha256"
    ):
        raise ValueError(f"{label} does not match bundle-manifest.json: {path}")


def resolve_protected_inputs(
    root: Path,
    args: argparse.Namespace,
    config: dict,
    manifest: dict,
) -> tuple[Path, list[Path]]:
    entries = manifest_file_entries(manifest)
    behavior_relative = config["pathPolicy"]["behaviorModel"]
    behavior_model = relative_bundle_argument(
        root, args.behavior_model, "V3 behavior model"
    )
    approved_behavior = (root / behavior_relative).resolve(strict=False)
    if behavior_model != approved_behavior:
        raise ValueError("only the bundle-approved V3 behavior model may be used")
    verify_manifest_bound_file(
        root,
        behavior_model,
        behavior_relative,
        entries,
        "V3 behavior model",
    )

    rollout_entries = manifest.get("rollouts")
    if not isinstance(rollout_entries, list) or not rollout_entries:
        raise ValueError("bundle manifest contains no V3 rollouts")
    data_root_name = config["pathPolicy"]["dataRoot"]
    approved_relatives = []
    for rollout in rollout_entries:
        filename = rollout.get("filename") if isinstance(rollout, dict) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".ndjson")
        ):
            raise ValueError("bundle manifest contains an invalid rollout filename")
        approved_relatives.append(f"{data_root_name}/{filename}")
    if len(set(approved_relatives)) != len(approved_relatives):
        raise ValueError("bundle manifest contains duplicate rollout filenames")

    requested: set[Path] = set()
    for pattern in args.data:
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError("V3 data patterns must stay inside the bundle")
        matches = [Path(value) for value in glob.glob(str(root / pattern))]
        if not matches and (root / pattern_path).is_file():
            matches = [root / pattern_path]
        requested.update(path.resolve(strict=False) for path in matches)
    approved_paths = {
        (root / relative).resolve(strict=False) for relative in approved_relatives
    }
    if requested != approved_paths:
        raise ValueError(
            "V3 data arguments must select every and only bundle-approved rollout"
        )
    for relative in approved_relatives:
        path = (root / relative).resolve(strict=False)
        verify_manifest_bound_file(
            root, path, relative, entries, "V3 rollout data"
        )
    data_paths = sorted(approved_paths, key=lambda path: str(path).lower())
    return behavior_model, data_paths


def protected_input_snapshot(paths: list[Path]) -> dict[Path, tuple[int, str]]:
    return {
        path: (path.stat().st_size, file_sha256(path))
        for path in paths
    }


def assert_protected_inputs_unchanged(
    snapshot: dict[Path, tuple[int, str]],
) -> None:
    for path, expected in snapshot.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"protected bundle input disappeared: {path}")
        current = (path.stat().st_size, file_sha256(path))
        if current != expected:
            raise RuntimeError(f"protected bundle input changed during run: {path}")


def run_and_tee(
    command: list[str],
    log_path: Path,
    *,
    append: bool,
    environment: dict[str, str],
) -> None:
    display = subprocess.list2cmdline(command)
    print(f"\n> {display}", flush=True)
    with log_path.open("a" if append else "x", encoding="utf-8") as log:
        log.write(f"\n> {display}\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_after_sealing_log(
    command: list[str],
    log_path: Path,
    *,
    environment: dict[str, str],
) -> None:
    """Run packaging without mutating a file while it is being archived."""
    display = subprocess.list2cmdline(command)
    print(f"\n> {display}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n> {display}\n")
        log.flush()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
    )
    assert process.stdout is not None
    with process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def strict_python_environment(
    seed: int, inherited: dict[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if inherited is None else inherited)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        }
    )
    return environment


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    config_path = root / "gpu-run-config.json"
    manifest_path = root / "bundle-manifest.json"
    assert_regular_bundle_file(root, config_path, "V3 GPU run config")
    assert_regular_bundle_file(root, manifest_path, "V3 bundle manifest")
    load_source_contract(
        manifest_path,
        config_path,
        verify_source_files=True,
    )
    config = read_json_object(config_path, "V3 GPU run config")
    manifest = read_json_object(manifest_path, "V3 bundle manifest")
    if (
        manifest.get("format") != "dalmuti-v3-ppo-gpu-bundle"
        or manifest.get("version") != 1
    ):
        raise ValueError("unsupported V3 bundle manifest")
    validate_run_config(config, args)
    run_id, output, results_dir = resolve_fresh_run_paths(
        root, args.output, args.results_dir, config["pathPolicy"]
    )
    behavior_model, data_paths = resolve_protected_inputs(
        root, args, config, manifest
    )
    protected_paths = [
        (root / relative).resolve(strict=False)
        for relative in manifest_file_entries(manifest)
    ]
    protected_paths.append(manifest_path)
    protected_paths = list(dict.fromkeys(protected_paths))
    for input_path in protected_paths:
        if paths_overlap(output, input_path) or paths_overlap(results_dir, input_path):
            raise ValueError("V3 output/results overlap protected bundle inputs")
    snapshot = protected_input_snapshot(protected_paths)

    output.parent.mkdir(exist_ok=True)
    assert_no_symlink_components(root, output.parent, "V3 output root")
    output.mkdir(exist_ok=False)
    log_path = output / "training.log"
    python = sys.executable
    environment = strict_python_environment(args.seed)

    run_and_tee(
        [python, str(root / "verify_bundle.py")],
        log_path,
        append=False,
        environment=environment,
    )
    assert_protected_inputs_unchanged(snapshot)
    run_and_tee(
        [
            python,
            str(root / "preflight.py"),
            "--device",
            args.device,
            "--deterministic",
            "--seed",
            str(args.seed),
            "--output",
            str(output / "hardware-report.json"),
        ],
        log_path,
        append=True,
        environment=environment,
    )
    assert_protected_inputs_unchanged(snapshot)
    common = [
        "--data",
        *(str(path) for path in data_paths),
        "--behavior-model",
        str(behavior_model),
        "--gamma",
        str(args.gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--terminal-rank-auxiliary-coefficient",
        str(args.terminal_rank_auxiliary_coefficient),
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--binding-tolerance",
        str(args.binding_tolerance),
        "--skip-forced-policy-time"
        if args.skip_forced_policy_time
        else "--no-skip-forced-policy-time",
    ]
    run_and_tee(
        [
            python,
            str(root / "train_v3_ppo.py"),
            *common,
            "--output",
            str(output),
            "--data-verification-output",
            str(output / "data-verification.json"),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--clip-coefficient",
            str(args.clip_coefficient),
            "--value-coefficient",
            str(args.value_coefficient),
            "--entropy-coefficient",
            str(args.entropy_coefficient),
            "--max-gradient-norm",
            str(args.max_gradient_norm),
            "--target-kl",
            str(args.target_kl),
            "--behavior-binding-batch-size",
            str(args.behavior_binding_batch_size),
            "--loader-workers",
            str(args.loader_workers),
            "--seed",
            str(args.seed),
            "--run-id",
            run_id,
            "--bundle-manifest",
            str(manifest_path),
            "--run-config",
            str(config_path),
        ],
        log_path,
        append=True,
        environment=environment,
    )
    assert_protected_inputs_unchanged(snapshot)
    run_after_sealing_log(
        [
            python,
            str(root / "package_v3_ppo_results.py"),
            "--model-dir",
            str(output),
            "--results-dir",
            str(results_dir),
            "--expected-bundle-manifest",
            str(manifest_path),
            "--expected-run-config",
            str(config_path),
        ],
        log_path,
        environment=environment,
    )
    assert_protected_inputs_unchanged(snapshot)
    print(f"V3 PPO handoff results are ready in {results_dir}")


if __name__ == "__main__":
    main()
