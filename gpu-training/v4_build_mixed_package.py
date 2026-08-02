#!/usr/bin/env python3
"""Build a deterministic, immutable V4 mixed-host execution package.

The source tarball is assembled from committed Git blobs, never from worktree
bytes.  This is the deliberate CRLF/autocrlf boundary: a dirty or converted
checkout cannot change the archive for a fixed commit and recipe.  The recipe
itself, the execution ledger, every source hash, and all runtime payloads are
bound by one canonical package manifest.  No real package should be built
until the intended core sources and recipe have been committed.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from v4_mixed_package_runtime import (
    BINDING_FORMAT,
    PACKAGE_FORMAT,
    PLAYER_COUNTS,
    RECIPE_FORMAT,
    canonical_json_bytes,
    safe_leaf,
    safe_relative_path,
    sha256_bytes,
    sha256_file,
    sidecar_bytes,
)


PACKAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{7,127}$")
PLACEHOLDERS = {
    "{package_directory}": "PACKAGE_DIRECTORY",
    "{package_manifest_sha256}": "EXPECTED_MANIFEST_SHA256",
    "{run_directory}": "RUN_DIRECTORY",
    "{source_root}": "SOURCE_ROOT",
}


@dataclass(frozen=True)
class GitBlob:
    path: str
    oid: str
    mode: int
    payload: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.payload)


def _git(repository: Path, arguments: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git command failed ({' '.join(arguments)}): {detail}")
    return result.stdout


def resolve_commit(repository: Path, revision: str) -> str:
    require = lambda condition, message: None if condition else (_ for _ in ()).throw(ValueError(message))
    require(isinstance(revision, str) and revision != "", "commit revision is missing")
    commit = _git(repository, ["rev-parse", "--verify", f"{revision}^{{commit}}"] ).decode("ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is not None, "Git returned an invalid commit id")
    return commit


def _tree(repository: Path, commit: str) -> dict[str, tuple[str, str]]:
    payload = _git(repository, ["ls-tree", "-r", "-z", "--full-tree", commit])
    result: dict[str, tuple[str, str]] = {}
    for raw in payload.split(b"\0"):
        if raw == b"":
            continue
        metadata, separator, path_bytes = raw.partition(b"\t")
        if separator != b"\t":
            raise ValueError("invalid Git tree record")
        try:
            mode, kind, oid = metadata.decode("ascii").split(" ")
            path = path_bytes.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("invalid Git tree encoding") from error
        if kind == "blob":
            result[path] = (mode, oid)
    return result


def committed_blobs(repository: Path, commit: str, paths: Sequence[str]) -> list[GitBlob]:
    tree = _tree(repository, commit)
    blobs: list[GitBlob] = []
    for path in paths:
        if path not in tree:
            raise ValueError(f"recipe source path is not a committed regular file: {path}")
        git_mode, oid = tree[path]
        if git_mode == "100644":
            mode = 0o644
        elif git_mode == "100755":
            mode = 0o755
        else:
            raise ValueError(f"unsupported Git mode for source file: {path} ({git_mode})")
        payload = _git(repository, ["cat-file", "blob", oid])
        blobs.append(GitBlob(path=path, oid=oid, mode=mode, payload=payload))
    return blobs


def _canonical_object(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, Mapping) or payload != canonical_json_bytes(value):
        raise ValueError(f"{label} must be one canonical JSON object")
    return value


def validate_recipe(recipe: Mapping[str, Any], recipe_path: str) -> None:
    expected_fields = {
        "entrypoint",
        "format",
        "ledgerPath",
        "packageId",
        "packagingBuilderPath",
        "runContract",
        "runtimeVerifierPath",
        "screening",
        "sourcePaths",
        "version",
    }
    if set(recipe) != expected_fields:
        raise ValueError("recipe fields are non-canonical")
    if recipe.get("format") != RECIPE_FORMAT or recipe.get("version") != 1:
        raise ValueError("unsupported package recipe")
    package_id = recipe.get("packageId")
    if not isinstance(package_id, str) or PACKAGE_ID_RE.fullmatch(package_id) is None:
        raise ValueError("invalid package id")
    paths = recipe.get("sourcePaths")
    if not isinstance(paths, list) or not paths or any(not isinstance(item, str) for item in paths):
        raise ValueError("recipe sourcePaths are invalid")
    normalized = [str(safe_relative_path(item, "source path")) for item in paths]
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("recipe sourcePaths must be sorted and unique")
    ledger_path = str(safe_relative_path(recipe.get("ledgerPath"), "ledger path"))
    builder_path = str(safe_relative_path(recipe.get("packagingBuilderPath"), "packaging builder path"))
    runtime_path = str(safe_relative_path(recipe.get("runtimeVerifierPath"), "runtime verifier path"))
    entrypoint = recipe.get("entrypoint")
    if not isinstance(entrypoint, Mapping) or set(entrypoint) != {"argv", "path"}:
        raise ValueError("entrypoint recipe is invalid")
    entrypoint_path = str(safe_relative_path(entrypoint.get("path"), "entrypoint path"))
    argv = entrypoint.get("argv")
    if not isinstance(argv, list) or any(not isinstance(token, str) or token == "" or "\x00" in token for token in argv):
        raise ValueError("entrypoint argv is invalid")
    for token in argv:
        if "{" in token or "}" in token:
            if token not in PLACEHOLDERS:
                raise ValueError(f"unsupported entrypoint placeholder: {token}")
    screening = recipe.get("screening")
    expected_screening_fields = {
        "actsPerMatch",
        "baseSeed",
        "bootstrapResamples",
        "candidateDirectory",
        "evaluatorPath",
        "familyId",
        "matchesPerPlayerCount",
        "normalBaselineSha256",
        "observationSchemaSha256",
        "playerCounts",
        "reportPath",
    }
    if not isinstance(screening, Mapping) or set(screening) != expected_screening_fields:
        raise ValueError("screening recipe fields are non-canonical")
    evaluator_path = str(safe_relative_path(screening.get("evaluatorPath"), "evaluator path"))
    str(safe_relative_path(screening.get("candidateDirectory"), "candidate directory"))
    str(safe_relative_path(screening.get("reportPath"), "screening report path"))
    if screening.get("playerCounts") != list(PLAYER_COUNTS) or screening.get("actsPerMatch") != 5:
        raise ValueError("screening must cover five-act p4-p10")
    matches = screening.get("matchesPerPlayerCount")
    if not isinstance(matches, int) or isinstance(matches, bool) or matches < 1:
        raise ValueError("invalid screening match count")
    resamples = screening.get("bootstrapResamples")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 10_000:
        raise ValueError("screening requires at least 10,000 clustered bootstrap resamples")
    base_seed = screening.get("baseSeed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("invalid screening base seed")
    family_id = screening.get("familyId")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("invalid screening family id")
    for field in ("normalBaselineSha256", "observationSchemaSha256"):
        value = screening.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"invalid screening {field}")
    if not isinstance(recipe.get("runContract"), Mapping):
        raise ValueError("runContract must be an object")
    required_paths = {recipe_path, ledger_path, builder_path, runtime_path, entrypoint_path, evaluator_path}
    missing = required_paths - set(normalized)
    if missing:
        raise ValueError(f"required sealed source paths are missing: {sorted(missing)}")


def _archive_bytes(blobs: Sequence[GitBlob], prefix: str = "source") -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for blob in blobs:
            member = tarfile.TarInfo(name=f"{prefix}/{blob.path}")
            member.size = len(blob.payload)
            member.mode = blob.mode
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            archive.addfile(member, io.BytesIO(blob.payload))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=compressed, mtime=0) as output:
        output.write(tar_buffer.getvalue())
    return compressed.getvalue()


def _record(blob: GitBlob) -> dict[str, object]:
    return {
        "gitBlobOid": blob.oid,
        "mode": blob.mode,
        "path": blob.path,
        "sha256": blob.sha256,
        "size": len(blob.payload),
    }


def _shell_array(tokens: Sequence[str]) -> str:
    return " ".join(shlex.quote(token) for token in tokens)


def render_launcher(recipe: Mapping[str, Any], verifier_name: str) -> bytes:
    entrypoint = recipe["entrypoint"]
    screening = recipe["screening"]
    entrypoint_path = shlex.quote(str(entrypoint["path"]))
    argument_templates = _shell_array([str(item) for item in entrypoint["argv"]])
    report_path = shlex.quote(str(screening["reportPath"]))
    candidate_directory = shlex.quote(str(screening["candidateDirectory"]))
    verifier_leaf = shlex.quote(safe_leaf(verifier_name, "verifier filename"))
    script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\\n\\t'

PACKAGE_DIRECTORY=''
EXPECTED_MANIFEST_SHA256=''
RUN_DIRECTORY=''
PYTHON_EXECUTABLE='python3'
while (($#)); do
  case "$1" in
    --package-dir) PACKAGE_DIRECTORY=$2; shift 2 ;;
    --expected-manifest-sha256) EXPECTED_MANIFEST_SHA256=$2; shift 2 ;;
    --run-directory) RUN_DIRECTORY=$2; shift 2 ;;
    --python) PYTHON_EXECUTABLE=$2; shift 2 ;;
    *) printf 'unexpected argument: %s\\n' "$1" >&2; exit 64 ;;
  esac
done
[[ -d "$PACKAGE_DIRECTORY" && -n "$EXPECTED_MANIFEST_SHA256" && -d "$RUN_DIRECTORY" ]]
[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{{64}}$ ]]
readonly PACKAGE_DIRECTORY EXPECTED_MANIFEST_SHA256 RUN_DIRECTORY PYTHON_EXECUTABLE
readonly VERIFIER="$PACKAGE_DIRECTORY/"{verifier_leaf}
readonly SOURCE_ROOT="$RUN_DIRECTORY/source"
readonly STATUS_DIRECTORY="$RUN_DIRECTORY/status"
readonly LOG_DIRECTORY="$RUN_DIRECTORY/logs"
readonly PROVENANCE_DIRECTORY="$RUN_DIRECTORY/provenance"
readonly RUN_SEAL="$PROVENANCE_DIRECTORY/final-files.json"
readonly ENTRYPOINT_REL={entrypoint_path}
readonly SCREENING_REPORT_REL={report_path}
readonly CANDIDATE_DIRECTORY_REL={candidate_directory}
ENTRYPOINT_ARGUMENT_TEMPLATES=({argument_templates})
CHILD_PID=''
SUCCESS=0
FAILURE_REPORTED=0

write_status() {{
  local sequence=$1 stage=$2 state=$3 detail=$4
  "$PYTHON_EXECUTABLE" "$VERIFIER" write-status \
    --output "$STATUS_DIRECTORY/${{sequence}}-${{state}}.json" \
    --stage "$stage" --state "$state" --detail "$detail" >/dev/null
}}

cleanup_child_group() {{
  local pid=${{CHILD_PID:-}}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in {{1..50}}; do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  CHILD_PID=''
}}

on_signal() {{
  local signal=$1 code=$2
  trap - INT TERM HUP
  cleanup_child_group
  if [[ "$FAILURE_REPORTED" == 0 ]]; then
    FAILURE_REPORTED=1
    write_status 998 launcher cancelled "received $signal" || true
  fi
  exit "$code"
}}

on_exit() {{
  local code=$1
  trap - EXIT INT TERM HUP
  cleanup_child_group
  if [[ "$SUCCESS" == 0 && "$FAILURE_REPORTED" == 0 ]]; then
    FAILURE_REPORTED=1
    write_status 998 launcher failed "launcher exited with code $code" || true
  fi
}}

trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
trap 'on_exit $?' EXIT

[[ -f "$VERIFIER" && ! -L "$VERIFIER" ]]
[[ -d "$STATUS_DIRECTORY" && -d "$LOG_DIRECTORY" ]]
[[ ! -e "$SOURCE_ROOT" && ! -e "$PROVENANCE_DIRECTORY" ]]
"$PYTHON_EXECUTABLE" "$VERIFIER" verify-package \
  --package-dir "$PACKAGE_DIRECTORY" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --remote-only >/dev/null
write_status 010 package completed 'remote payload and all fresh sidecars verified'
"$PYTHON_EXECUTABLE" "$VERIFIER" extract-source \
  --package-dir "$PACKAGE_DIRECTORY" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --destination "$SOURCE_ROOT" >/dev/null
write_status 020 source completed 'committed source archive extracted and verified'

WORKFLOW_ARGUMENTS=()
for token in "${{ENTRYPOINT_ARGUMENT_TEMPLATES[@]}}"; do
  case "$token" in
    '{{package_directory}}') WORKFLOW_ARGUMENTS+=("$PACKAGE_DIRECTORY") ;;
    '{{package_manifest_sha256}}') WORKFLOW_ARGUMENTS+=("$EXPECTED_MANIFEST_SHA256") ;;
    '{{run_directory}}') WORKFLOW_ARGUMENTS+=("$RUN_DIRECTORY") ;;
    '{{source_root}}') WORKFLOW_ARGUMENTS+=("$SOURCE_ROOT") ;;
    *) WORKFLOW_ARGUMENTS+=("$token") ;;
  esac
done
write_status 100 workflow started 'sealed V4 mixed-host workflow starting'
setsid "$PYTHON_EXECUTABLE" "$SOURCE_ROOT/$ENTRYPOINT_REL" "${{WORKFLOW_ARGUMENTS[@]}}" &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
workflow_code=$?
set -e
if kill -0 -- "-$CHILD_PID" 2>/dev/null; then
  # A leader that exited while descendants survived is still a failed run.
  cleanup_child_group
  [[ "$workflow_code" != 0 ]] || workflow_code=125
else
  CHILD_PID=''
fi
if [[ "$workflow_code" != 0 ]]; then
  FAILURE_REPORTED=1
  write_status 998 workflow failed "workflow exit code $workflow_code"
  exit "$workflow_code"
fi
write_status 800 workflow completed 'sealed V4 mixed-host workflow completed'

"$PYTHON_EXECUTABLE" "$VERIFIER" verify-screening \
  --package-dir "$PACKAGE_DIRECTORY" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" \
  --source-root "$SOURCE_ROOT" \
  --report "$RUN_DIRECTORY/$SCREENING_REPORT_REL" \
  --candidate "$RUN_DIRECTORY/$CANDIDATE_DIRECTORY_REL" >/dev/null
write_status 900 screening completed 'complete p4-p10 pure-Actor screening verified'

"$PYTHON_EXECUTABLE" "$VERIFIER" seal-run \
  --run-directory "$RUN_DIRECTORY" --output "$RUN_SEAL" \
  --status-directory "$STATUS_DIRECTORY" >/dev/null
# Success is intentionally emitted only after the immutable run seal exists
# and its sidecar has been re-verified by write-status.
"$PYTHON_EXECUTABLE" "$VERIFIER" write-status \
  --output "$STATUS_DIRECTORY/999-succeeded.json" \
  --stage complete --state succeeded \
  --detail 'workflow, full screening verification, and immutable run seal completed' \
  --seal "$RUN_SEAL" >/dev/null
SUCCESS=1
trap - EXIT INT TERM HUP
exit 0
"""
    return script.replace("\r\n", "\n").encode("utf-8")


def render_controller() -> bytes:
    # The mandatory ExpectedPackageManifestSha256 parameter is the sole local
    # trust root.  No package-specific digest or run directory is compiled in.
    script = r'''[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $PackageDirectory,
    [Parameter(Mandatory = $true)] [ValidatePattern('^[0-9a-f]{64}$')] [string] $ExpectedPackageManifestSha256,
    [Parameter(Mandatory = $true)] [string] $LocalRunDirectory,
    [Parameter(Mandatory = $true)] [string] $RemoteEndpoint,
    [Parameter(Mandatory = $true)] [string] $RemoteRunDirectory,
    [Parameter(Mandatory = $true)] [string] $BehaviorActorBundle,
    [Parameter(Mandatory = $true)] [string] $FrozenBaselineBundle,
    [string] $IdentityFile,
    [ValidateRange(1, 65535)] [int] $Port = 22,
    [string] $LocalPython = 'python',
    [string] $RemotePython = 'python3',
    [string] $SshExecutable = 'ssh',
    [string] $ScpExecutable = 'scp'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Quote-PosixShell([string] $Value) {
    $singleQuote = [char]39
    $replacement = [string]::Concat($singleQuote, [char]34, $singleQuote, [char]34, $singleQuote)
    return [string]::Concat($singleQuote, $Value.Replace([string]$singleQuote, $replacement), $singleQuote)
}

function Invoke-Checked([string] $Executable, [string[]] $Arguments, [string] $Description) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

$root = (Resolve-Path -LiteralPath $PackageDirectory).Path
$manifestPath = Join-Path $root 'package-manifest.json'
$manifestSidecarPath = "$manifestPath.sha256"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or -not (Test-Path -LiteralPath $manifestSidecarPath -PathType Leaf)) {
    throw 'Package manifest or its sidecar is missing.'
}
$manifestItem = Get-Item -LiteralPath $manifestPath -Force
$manifestSidecarItem = Get-Item -LiteralPath $manifestSidecarPath -Force
if (($manifestItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or ($manifestSidecarItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Package manifest paths must not be reparse points.' }
$actualManifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualManifestSha256 -ne $ExpectedPackageManifestSha256) { throw 'Package manifest does not match the mandatory trust-root digest.' }
$expectedManifestSidecar = "$actualManifestSha256  package-manifest.json`n"
if ([IO.File]::ReadAllText($manifestSidecarPath) -ne $expectedManifestSidecar) { throw 'Package manifest sidecar is stale or malformed.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.format -ne 'dalmuti-v4-mixed-run-package' -or [int]$manifest.version -ne 1) { throw 'Unsupported package manifest.' }
$seenNames = @{}
$seenRoles = @{}
foreach ($record in @($manifest.files)) {
    $name = [string]$record.name
    $role = [string]$record.role
    $digest = [string]$record.sha256
    if ($name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or [IO.Path]::GetFileName($name) -ne $name) { throw "Unsafe package filename: $name" }
    if ($role -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$' -or $seenNames.ContainsKey($name) -or $seenRoles.ContainsKey($role)) { throw 'Duplicate or unsafe package inventory.' }
    if ($digest -notmatch '^[0-9a-f]{64}$') { throw "Invalid package digest: $name" }
    $seenNames[$name] = $true
    $seenRoles[$role] = $true
    $payload = Join-Path $root $name
    $sidecar = "$payload.sha256"
    $item = Get-Item -LiteralPath $payload -Force
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Package payload is not a regular file: $name" }
    if ([int64]$item.Length -ne [int64]$record.size) { throw "Package payload size mismatch: $name" }
    if ((Get-FileHash -LiteralPath $payload -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) { throw "Package payload digest mismatch: $name" }
    if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) { throw "Package payload sidecar is missing: $name" }
    $sidecarItem = Get-Item -LiteralPath $sidecar -Force
    if (($sidecarItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or [IO.File]::ReadAllText($sidecar) -ne "$digest  $name`n") { throw "Package payload sidecar is stale or malformed: $name" }
}
$verifierRecord = @($manifest.files | Where-Object { $_.role -eq 'verifier' })
if ($verifierRecord.Count -ne 1) { throw 'Package has no unique verifier.' }
$verifierPath = Join-Path $root $verifierRecord[0].name
Invoke-Checked $LocalPython @($verifierPath, 'verify-package', '--package-dir', $root, '--expected-manifest-sha256', $ExpectedPackageManifestSha256) 'local package verification'

$localRun = [IO.Path]::GetFullPath($LocalRunDirectory)
if (Test-Path -LiteralPath $localRun) { throw 'Local run directory is immutable and must be fresh.' }
[IO.Directory]::CreateDirectory($localRun) > $null
[IO.Directory]::CreateDirectory((Join-Path $localRun 'status')) > $null
[IO.Directory]::CreateDirectory((Join-Path $localRun 'logs')) > $null
$sourceRoot = Join-Path $localRun 'source'
try {
    Invoke-Checked $LocalPython @($verifierPath, 'extract-source', '--package-dir', $root, '--expected-manifest-sha256', $ExpectedPackageManifestSha256, '--destination', $sourceRoot) 'local sealed-source extraction'
    $coordinator = Join-Path $sourceRoot 'gpu-training/v4_mixed_local_coordinator.py'
    $coordinatorArgs = @(
    $coordinator, 'execute',
    '--source-root', $sourceRoot,
    '--package-directory', $root,
    '--package-manifest-sha256', $ExpectedPackageManifestSha256,
    '--local-run-directory', $localRun,
    '--remote-endpoint', $RemoteEndpoint,
    '--remote-run-directory', $RemoteRunDirectory,
    '--behavior-actor-bundle', (Resolve-Path -LiteralPath $BehaviorActorBundle).Path,
    '--frozen-baseline-bundle', (Resolve-Path -LiteralPath $FrozenBaselineBundle).Path,
    '--port', [string]$Port,
    '--local-python', $LocalPython,
    '--remote-python', $RemotePython,
    '--ssh-executable', $SshExecutable,
    '--scp-executable', $ScpExecutable
    )
    if ($IdentityFile) { $coordinatorArgs += @('--identity-file', (Resolve-Path -LiteralPath $IdentityFile).Path) }
    Invoke-Checked $LocalPython $coordinatorArgs 'local mixed-host coordinator'
} catch {
    $controllerFailureDetail = "controller failed after local run creation: $($_.Exception.Message)"
    $failurePath = Join-Path $localRun 'status/998-failed.json'
    $failureSidecar = "$failurePath.sha256"
    if (-not (Test-Path -LiteralPath $failurePath) -and -not (Test-Path -LiteralPath $failureSidecar)) {
        try {
            & $LocalPython @($verifierPath, 'write-status', '--output', $failurePath, '--stage', 'controller', '--state', 'failed', '--detail', $controllerFailureDetail) > $null
        } catch { }
    }
    throw
}
Write-Output "Local coordinator completed. Success is valid only after both local and remote run seals verify."
'''
    return script.replace("\r\n", "\n").encode("utf-8")


def _write_payload(directory: Path, name: str, payload: bytes, mode: int) -> dict[str, object]:
    safe_leaf(name, "payload filename")
    path = directory / name
    if path.exists() or Path(f"{path}.sha256").exists():
        raise FileExistsError(f"immutable package payload exists: {name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    digest = sha256_bytes(payload)
    descriptor = os.open(Path(f"{path}.sha256"), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(sidecar_bytes(digest, name))
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return {"name": name, "sha256": digest, "size": len(payload)}


def build_package(repository: Path, revision: str, recipe_path: str, output_directory: Path) -> Mapping[str, Any]:
    repository = repository.resolve(strict=True)
    recipe_path = str(safe_relative_path(recipe_path, "recipe path"))
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError("package output directory is immutable and must be fresh")
    commit = resolve_commit(repository, revision)
    initial_tree = _tree(repository, commit)
    if recipe_path not in initial_tree:
        raise ValueError("package recipe is not committed at the selected commit")
    recipe_blob = committed_blobs(repository, commit, [recipe_path])[0]
    recipe = _canonical_object(recipe_blob.payload, "package recipe")
    validate_recipe(recipe, recipe_path)
    source_paths = [str(item) for item in recipe["sourcePaths"]]
    blobs = committed_blobs(repository, commit, source_paths)
    blob_by_path = {blob.path: blob for blob in blobs}
    builder_blob = blob_by_path[str(recipe["packagingBuilderPath"])]
    if sha256_file(Path(__file__).resolve()) != builder_blob.sha256:
        raise ValueError(
            "running packaging builder bytes do not match the selected committed builder blob"
        )
    package_id = str(recipe["packageId"])
    short_commit = commit[:12]
    archive_name = f"{package_id}-source-{short_commit}.tar.gz"
    binding_name = f"{package_id}-source-binding-{short_commit}.json"
    verifier_name = f"{package_id}-verifier.py"
    controller_name = f"{package_id}-controller.ps1"
    for name in (archive_name, binding_name, verifier_name, controller_name):
        safe_leaf(name, "generated payload filename")

    parent = output_directory.parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.building-", dir=parent))
    try:
        archive_payload = _archive_bytes(blobs)
        archive_record = _write_payload(temporary, archive_name, archive_payload, 0o600)
        source_records = [_record(blob) for blob in blobs]
        ledger_blob = blob_by_path[str(recipe["ledgerPath"])]
        recipe_blob = blob_by_path[recipe_path]
        ledger_binding = {"gitBlobOid": ledger_blob.oid, "path": ledger_blob.path, "sha256": ledger_blob.sha256}
        recipe_binding = {"gitBlobOid": recipe_blob.oid, "path": recipe_blob.path, "sha256": recipe_blob.sha256}
        binding = {
            "archivePrefix": "source",
            "format": BINDING_FORMAT,
            "ledger": ledger_binding,
            "recipe": recipe_binding,
            "runContractSha256": sha256_bytes(canonical_json_bytes(recipe["runContract"])),
            "screeningContractSha256": sha256_bytes(canonical_json_bytes(recipe["screening"])),
            "sourceArchive": archive_record,
            "sourceCommit": commit,
            "sourceFiles": source_records,
            "sourceInventorySha256": sha256_bytes(canonical_json_bytes(source_records)),
            "version": 1,
        }
        binding_record = _write_payload(temporary, binding_name, canonical_json_bytes(binding), 0o600)
        runtime_blob = blob_by_path[str(recipe["runtimeVerifierPath"])]
        verifier_record = _write_payload(temporary, verifier_name, runtime_blob.payload, 0o700)
        controller_record = _write_payload(temporary, controller_name, render_controller(), 0o600)
        files = [
            {**archive_record, "remotePayload": True, "role": "source-archive"},
            {**binding_record, "remotePayload": True, "role": "source-binding"},
            {**verifier_record, "remotePayload": True, "role": "verifier"},
            {**controller_record, "remotePayload": False, "role": "controller"},
        ]
        unsigned_manifest = {
            "files": files,
            "format": PACKAGE_FORMAT,
            "ledger": ledger_binding,
            "packageId": package_id,
            "recipe": recipe_binding,
            "sourceCommit": commit,
            "version": 1,
        }
        manifest = {
            **unsigned_manifest,
            "canonicalSha256": sha256_bytes(canonical_json_bytes(unsigned_manifest)),
        }
        manifest_payload = canonical_json_bytes(manifest)
        manifest_record = _write_payload(temporary, "package-manifest.json", manifest_payload, 0o600)
        expected_names = {"package-manifest.json", "package-manifest.json.sha256"}
        for record in files:
            expected_names.add(str(record["name"]))
            expected_names.add(f"{record['name']}.sha256")
        observed_names = {path.name for path in temporary.iterdir()}
        if observed_names != expected_names:
            raise ValueError("package output inventory drifted during construction")
        committed_verifier = temporary / verifier_name
        verification = subprocess.run(
            [
                sys.executable,
                str(committed_verifier),
                "verify-package",
                "--package-dir",
                str(temporary),
                "--expected-manifest-sha256",
                str(manifest_record["sha256"]),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if verification.returncode != 0:
            detail = verification.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"committed runtime verifier rejected the package: {detail}")
        os.replace(temporary, output_directory)
        for path in output_directory.iterdir():
            if path.name == verifier_name:
                os.chmod(path, 0o555)
            else:
                os.chmod(path, 0o444)
        os.chmod(output_directory, 0o555)
        return {
            "format": "dalmuti-v4-mixed-package-build-result",
            "version": 1,
            "packageId": package_id,
            "outputDirectory": str(output_directory.resolve()),
            "packageManifestSha256": manifest_record["sha256"],
            "sourceArchiveSha256": archive_record["sha256"],
            "sourceBindingSha256": binding_record["sha256"],
            "sourceCommit": commit,
        }
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--commit", required=True, help="exact commit or revision to package")
    parser.add_argument("--recipe", required=True, help="committed canonical recipe path")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    result = build_package(args.repository, args.commit, args.recipe, args.output_directory)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
