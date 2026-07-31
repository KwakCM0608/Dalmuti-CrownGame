import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import { parseArgs } from "node:util";

const cliArgs = process.argv.slice(2);
if (cliArgs[0] === "--") cliArgs.shift();
const { values } = parseArgs({
  args: cliArgs,
  options: {
    directory: { type: "string", short: "d" },
    json: { type: "boolean", default: false },
  },
});
if (!values.directory) throw new TypeError("--directory is required");
const directory = resolve(values.directory);
const manifestPath = resolve(directory, "result-manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
if (
  ![
    "dalmuti-gpu-training-result",
    "dalmuti-ppo-training-result",
  ].includes(manifest.format) ||
  manifest.version !== 1 ||
  !Array.isArray(manifest.files)
) {
  throw new TypeError("unsupported training result manifest");
}

const verified = [];
for (const entry of manifest.files) {
  if (
    typeof entry.path !== "string" ||
    !Number.isSafeInteger(entry.bytes) ||
    typeof entry.sha256 !== "string"
  ) {
    throw new TypeError("invalid training result manifest entry");
  }
  const path = resolve(directory, entry.path);
  const relativePath = relative(directory, path);
  if (
    relativePath.startsWith("..") ||
    isAbsolute(relativePath) ||
    relativePath === ""
  ) {
    throw new RangeError(`result path escapes its directory: ${entry.path}`);
  }
  const fileStat = await stat(path);
  if (!fileStat.isFile() || fileStat.size !== entry.bytes) {
    throw new Error(`result file size mismatch: ${entry.path}`);
  }
  const content = await readFile(path);
  const sha256 = createHash("sha256").update(content).digest("hex");
  if (sha256 !== entry.sha256) {
    throw new Error(`result checksum mismatch: ${entry.path}`);
  }
  verified.push({
    path: entry.path,
    bytes: entry.bytes,
    sha256,
  });
}
const result = {
  directory,
  format: manifest.format,
  files: verified.length,
  totalBytes: verified.reduce((total, entry) => total + entry.bytes, 0),
  verified,
};
if (values.json) {
  console.log(JSON.stringify(result, null, 2));
} else {
  console.log(
    `Verified ${result.files} ${result.format} files ` +
      `(${(result.totalBytes / 1024 / 1024).toFixed(2)} MiB)`,
  );
}
