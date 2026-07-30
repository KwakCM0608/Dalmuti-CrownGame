import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");

test("manifest exposes a standalone install surface and both game shortcuts", () => {
  const manifest = read("app/manifest.ts");

  assert.match(manifest, /id:\s*"\/"/);
  assert.match(manifest, /start_url:\s*"\/"/);
  assert.match(manifest, /scope:\s*"\/"/);
  assert.match(manifest, /display:\s*"standalone"/);
  assert.match(manifest, /icon-192\.png/);
  assert.match(manifest, /icon-512\.png/);
  assert.match(manifest, /icon-maskable-512\.png/);
  assert.match(manifest, /purpose:\s*"maskable"/);
  assert.match(manifest, /name:\s*"빠른 대전"/);
  assert.match(manifest, /name:\s*"온라인 대전"/);
});

test("service worker never caches online API or mutation requests", () => {
  const worker = read("public/sw.js");

  assert.match(worker, /request\.method !== "GET"/);
  assert.match(worker, /pathname\.startsWith\("\/api\/online\/"\)/);
  assert.match(worker, /event\.respondWith\(fetch\(request\)\)/);
  assert.match(worker, /request\.mode === "navigate"/);
  assert.match(worker, /OFFLINE_URL/);
  assert.match(worker, /event\.data\?\.type === "SKIP_WAITING"/);
  assert.doesNotMatch(worker, /self\.skipWaiting\(\)[\s\S]*addEventListener\("install"/);
});

test("updates are offered only on safe entry screens", () => {
  const lifecycle = read("app/components/PwaLifecycle.tsx");
  const layout = read("app/layout.tsx");

  assert.match(lifecycle, /isSafeInstallScreen/);
  assert.match(lifecycle, /welcome-layer/);
  assert.match(lifecycle, /entryShell/);
  assert.match(lifecycle, /waitingWorker\.postMessage\(\{ type: "SKIP_WAITING" \}\)/);
  assert.match(lifecycle, /beforeinstallprompt/);
  assert.match(layout, /viewportFit:\s*"cover"/);
  assert.match(layout, /appleWebApp/);
  assert.match(layout, /<MobileSplash \/>/);
  assert.match(layout, /<PwaLifecycle \/>/);
});

test("installed mobile app opens with the crown splash screen", () => {
  const splash = read("app/components/MobileSplash.tsx");
  const styles = read("app/components/MobileSplash.module.css");
  const wrapper = JSON.parse(read("android-twa/twa-manifest.json"));

  assert.match(splash, /brand-dalmuti-crown\.png|styles\.crown/);
  assert.match(styles, /@media \(display-mode: standalone\)/);
  assert.match(styles, /brand-dalmuti-crown\.png/);
  assert.equal(wrapper.splashScreenFadeOutDuration, 420);
});

test("required offline and install assets exist", () => {
  for (const file of [
    "public/offline.html",
    "public/pwa/icon-192.png",
    "public/pwa/icon-512.png",
    "public/pwa/icon-maskable-512.png",
    "public/pwa/apple-touch-icon.png",
  ]) {
    assert.equal(fs.existsSync(path.join(root, file)), true, `${file} is missing`);
  }
});

test("Android wrapper and Digital Asset Links use the same package and key", () => {
  const wrapper = JSON.parse(read("android-twa/twa-manifest.json"));
  const links = JSON.parse(read("public/.well-known/assetlinks.json"));
  const target = links[0]?.target;

  assert.equal(wrapper.packageId, "lab.dclab.dalmuti");
  assert.equal(target?.package_name, wrapper.packageId);
  assert.deepEqual(
    target?.sha256_cert_fingerprints,
    wrapper.fingerprints.map((fingerprint) => fingerprint.value),
  );
  assert.equal(wrapper.host, "dalmuti-crown-game.byj010608.chatgpt.site");
});

test("mobile online runtime pauses room polling in the background", () => {
  const onlinePage = read("app/online/page.tsx");

  assert.match(
    onlinePage,
    /if \(document\.visibilityState !== "visible"\) \{[\s\S]*pollTimer = null;/,
  );
  assert.match(onlinePage, /document\.visibilityState === "visible" \? 250 : 1_000/);
});
