import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const sha256 = (file) =>
  createHash("sha256")
    .update(fs.readFileSync(path.join(root, file)))
    .digest("hex");

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
  assert.equal(wrapper.backgroundColor, "#000000");
  assert.equal(wrapper.splashScreenFadeOutDuration, 180);
  assert.equal(wrapper.appVersionCode, 6);
  assert.equal(wrapper.appVersion, "1.0.5");
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
    nativeLauncher,
    /protected boolean shouldLaunchImmediately\(\) \{\s*return false;/,
  );
  assert.match(nativeLauncher, /FrameLayout root = new FrameLayout\(this\)/);
  assert.match(nativeLauncher, /R\.drawable\.splash_glow/);
  assert.match(nativeLauncher, /artwork\.setAlpha\(0f\)/);
  assert.match(nativeLauncher, /artwork\.setScaleX\(0\.94f\)/);
  assert.match(nativeLauncher, /new DecelerateInterpolator\(1\.35f\)/);
  assert.match(nativeLauncher, /ImageView\.ScaleType\.FIT_CENTER/);
  assert.match(
    nativeLauncher,
    /splashHandler\.postDelayed\(this::launchTwaOnce, SPLASH_HANDOFF_AT_MS\)/,
  );
  assert.match(
    nativeLauncher,
    /if \(nativeLaunchDispatched \|\| isFinishing\(\)\)/,
  );
  assert.match(nativeLauncher, /NATIVE_LAUNCH_DISPATCHED_KEY/);
  const nativeTiming = Object.fromEntries(
    [
      "SPLASH_REVEAL_DELAY_MS",
      "SPLASH_REVEAL_DURATION_MS",
      "SPLASH_HANDOFF_AT_MS",
    ].map((name) => {
      const match = nativeLauncher.match(
        new RegExp(`${name} = (\\d+)L`),
      );
      assert.ok(match, `${name} must be declared`);
      return [name, Number(match[1])];
    }),
  );
  assert.ok(
    nativeTiming.SPLASH_REVEAL_DELAY_MS +
      nativeTiming.SPLASH_REVEAL_DURATION_MS <=
      nativeTiming.SPLASH_HANDOFF_AT_MS,
  );
  assert.ok(
    nativeTiming.SPLASH_HANDOFF_AT_MS +
      wrapper.splashScreenFadeOutDuration <=
      1_500,
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
      path.join(
        root,
        "android-twa/custom/res/drawable-xxxhdpi/splash_glow.png",
      ),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-app-icon-v2.png"),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-splash-v3.png"),
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
  assert.match(assetBuilder, /dalmuti-app-icon-v2\.png/);
  assert.match(assetBuilder, /dalmuti-splash-v3\.png/);
  assert.match(assetBuilder, /ImageFilter\.GaussianBlur/);
  assert.equal(
    sha256("android-twa/assets/dalmuti-app-icon-v2.png"),
    "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946",
  );
  assert.equal(
    sha256("android-twa/assets/dalmuti-splash-v3.png"),
    "13fadbea989e85980994d185b44f4a4215f3df59e075d1bdf6056a820756631f",
  );
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
