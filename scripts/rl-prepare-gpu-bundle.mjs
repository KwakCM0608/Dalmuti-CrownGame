import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

const projectRoot = resolve(new URL("..", import.meta.url).pathname.slice(1));
const packageRoot = join(projectRoot, "gpu-training");
const rolloutRoot = join(projectRoot, "artifacts", "rl");
const bundleRoot = join(rolloutRoot, "gpu-bundle-v2");
const bundleDataRoot = join(bundleRoot, "data");

const packageFiles = [
  "README.md",
  "requirements.txt",
  "schema.json",
  "dataset.py",
  "model.py",
  "train_bc.py",
  "verify_data.py",
  "verify_bundle.py",
];
const behaviorFiles = Array.from(
  { length: 7 },
  (_, index) => `bc-p${index + 4}-v2.ndjson`,
);
const daggerFiles = ["dagger", "dagger2"].flatMap((prefix) =>
  Array.from(
    { length: 7 },
    (_, index) => `${prefix}-p${index + 4}-v2.ndjson`,
  ),
);
const rolloutFiles = [...behaviorFiles, ...daggerFiles];

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
for (const filename of rolloutFiles) {
  await copyVerified(
    join(rolloutRoot, filename),
    join(bundleDataRoot, basename(filename)),
  );
}

const bundleManifest = {
  format: "dalmuti-gpu-bundle",
  version: 2,
  createdAt: new Date().toISOString(),
  teacherPolicy: "normal",
  files: manifestFiles,
  totalBytes: manifestFiles.reduce((total, file) => total + file.bytes, 0),
};
await writeFile(
  join(bundleRoot, "bundle-manifest.json"),
  `${JSON.stringify(bundleManifest, null, 2)}\n`,
  "utf8",
);

console.log(`GPU bundle ready: ${bundleRoot}`);
console.log(
  `${manifestFiles.length} files, ` +
    `${(bundleManifest.totalBytes / 1024 / 1024).toFixed(2)} MiB`,
);
