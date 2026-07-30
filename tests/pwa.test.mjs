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
  assert.match(lifecycle, /function isMobileWeb/);
  assert.match(lifecycle, /userAgentData\?\.mobile/);
  assert.match(lifecycle, /Android\|iPhone\|iPad\|iPod\|Mobile/);
  assert.match(lifecycle, /mobileWeb && installPrompt/);
  assert.match(layout, /viewportFit:\s*"cover"/);
  assert.match(layout, /appleWebApp/);
  assert.doesNotMatch(layout, /MobileSplash/);
  assert.doesNotMatch(layout, /PreferenceRuntime|SettingsDialog/);
  assert.match(layout, /<PwaLifecycle \/>/);
});

test("installed Android app uses one branded splash and system-aware rotation", () => {
  const wrapper = JSON.parse(read("android-twa/twa-manifest.json"));
  const webManifest = read("app/manifest.ts");
  const customizer = read("android-twa/apply-native-customizations.ps1");
  const nativeLauncher = read(
    "android-twa/custom/java/LauncherActivity.java",
  );
  const assetBuilder = read("scripts/build_android_splash_assets.py");
  const androidSplashTheme = read(
    "android-twa/custom/res/values-v31/styles.xml",
  );

  assert.match(wrapper.iconUrl, /icon-512\.png/);
  assert.equal(wrapper.backgroundColor, "#18070c");
  assert.equal(wrapper.splashScreenFadeOutDuration, 220);
  assert.equal(wrapper.appVersionCode, 5);
  assert.equal(wrapper.appVersion, "1.0.4");
  assert.equal(wrapper.orientation, "default");
  assert.doesNotMatch(webManifest, /orientation:\s*"any"/);
  assert.match(customizer, /customRoot "java\\LauncherActivity\.java"/);
  assert.match(customizer, /android:screenOrientation="unspecified"/);
  assert.match(customizer, /DalmutiLaunchTheme/);
  assert.match(nativeLauncher, /Settings\.System\.ACCELEROMETER_ROTATION/);
  assert.match(nativeLauncher, /ActivityInfo\.SCREEN_ORIENTATION_LOCKED/);
  assert.match(nativeLauncher, /ActivityInfo\.SCREEN_ORIENTATION_FULL_USER/);
  assert.match(nativeLauncher, /ScreenOrientation\.ANY/);
  assert.match(nativeLauncher, /ScreenOrientation\.LANDSCAPE/);
  assert.match(nativeLauncher, /ScreenOrientation\.PORTRAIT/);
  assert.match(
    nativeLauncher,
    /builder\.setScreenOrientation\(launchOrientation\)/,
  );
  assert.match(
    androidSplashTheme,
    /android:windowSplashScreenAnimatedIcon[^]*dalmuti_splash_transparent/,
  );
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/drawable-xxxhdpi/splash.png",
      ),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-app-icon-v1.png"),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-splash-v2.png"),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/mipmap-xxxhdpi/ic_launcher.png",
      ),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/mipmap-anydpi-v26/ic_launcher.xml",
      ),
    ),
    true,
  );
  assert.match(assetBuilder, /dalmuti-app-icon-v1\.png/);
  assert.match(assetBuilder, /dalmuti-splash-v2\.png/);
  assert.doesNotMatch(androidSplashTheme, /windowSplashScreenBrandingImage/);
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/drawable/dalmuti_splash_transparent.xml",
      ),
    ),
    true,
  );
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
