import { createHash } from "node:crypto";
import {
  copyFile,
  mkdir,
  readFile,
  writeFile,
} from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { parseArgs } from "node:util";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    model: { type: "string", short: "m" },
    rollout: { type: "string", multiple: true },
    output: {
      type: "string",
      default: "artifacts/rl/ppo-gpu-bundle-v1",
    },
  },
});
if (!values.model) throw new TypeError("--model is required");
if (!values.rollout?.length) {
  throw new TypeError("at least one --rollout is required");
}

const projectRoot = resolve(new URL("..", import.meta.url).pathname.slice(1));
const packageRoot = join(projectRoot, "gpu-training");
const bundleRoot = resolve(projectRoot, values.output);
const bundleDataRoot = join(bundleRoot, "data");
const modelPath = resolve(projectRoot, values.model);
const modelBytes = await readFile(modelPath);
const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
const rolloutPaths = values.rollout.map((path) =>
  resolve(projectRoot, path),
);
const packageFiles = [
  "requirements.txt",
  "preflight.py",
  "verify_bundle.py",
  "ppo-schema.json",
  "actor_critic.py",
  "ppo_dataset.py",
  "verify_ppo_data.py",
  "test_ppo.py",
  "train_ppo.py",
  "package_ppo_results.py",
  "run_gpu_ppo.py",
  "PROMPT_FOR_GPU_PPO.md",
];

for (const rolloutPath of rolloutPaths) {
  const firstLine = (await readFile(rolloutPath, "utf8"))
    .split(/\r?\n/, 1)[0];
  const manifest = JSON.parse(firstLine);
  if (
    manifest.format !== "dalmuti-ppo-ndjson" ||
    manifest.behaviorModel?.sha256 !== modelSha256
  ) {
    throw new Error(
      `${rolloutPath} does not match the supplied behavior model`,
    );
  }
}

await mkdir(bundleDataRoot, { recursive: true });
const manifestFiles = [];
async function copyVerified(source, destination) {
  const content = await readFile(source);
  await copyFile(source, destination);
  manifestFiles.push({
    path: destination.slice(bundleRoot.length + 1).replaceAll("\\", "/"),
    bytes: content.byteLength,
    sha256: createHash("sha256").update(content).digest("hex"),
  });
}

for (const filename of packageFiles) {
  await copyVerified(
    join(packageRoot, filename),
    join(bundleRoot, filename),
  );
}
await copyVerified(modelPath, join(bundleRoot, "behavior-model.json"));
const usedNames = new Set();
for (const rolloutPath of rolloutPaths) {
  const filename = basename(rolloutPath);
  if (usedNames.has(filename)) {
    throw new Error(`duplicate rollout filename: ${filename}`);
  }
  usedNames.add(filename);
  await copyVerified(
    rolloutPath,
    join(bundleDataRoot, filename),
  );
}

const bundleManifest = {
  format: "dalmuti-ppo-gpu-bundle",
  version: 1,
  createdAt: new Date().toISOString(),
  behaviorModelSha256: modelSha256,
  files: manifestFiles,
  totalBytes: manifestFiles.reduce((total, file) => total + file.bytes, 0),
};
await writeFile(
  join(bundleRoot, "bundle-manifest.json"),
  `${JSON.stringify(bundleManifest, null, 2)}\n`,
  "utf8",
);
console.log(`PPO GPU bundle ready: ${bundleRoot}`);
console.log(
  `${manifestFiles.length} files, ` +
    `${(bundleManifest.totalBytes / 1024 / 1024).toFixed(2)} MiB`,
);
