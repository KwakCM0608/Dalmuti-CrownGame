import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const sha256 = (file) =>
  createHash("sha256")
    .update(fs.readFileSync(path.join(root, file)))
    .digest("hex");
const sha256NormalizedText = (file) =>
  createHash("sha256")
    .update(read(file).replace(/\r\n?/g, "\n"))
    .digest("hex");

test("manifest exposes a standalone install surface and both game shortcuts", () => {
  const manifest = read("app/manifest.ts");

  assert.match(manifest, /id:\s*"\/"/);
  assert.match(manifest, /start_url:\s*"\/"/);
  assert.match(manifest, /scope:\s*"\/"/);
  assert.match(manifest, /display:\s*"standalone"/);
  assert.match(manifest, /background_color:\s*"#000000"/);
  assert.match(manifest, /icon-v2-192\.png/);
  assert.match(manifest, /icon-v2-512\.png/);
  assert.match(manifest, /icon-v2-1024\.png/);
  assert.match(manifest, /icon-maskable-v2-512\.png/);
  assert.match(manifest, /purpose:\s*"maskable"/);
  assert.match(manifest, /name:\s*"빠른 대전"/);
  assert.match(manifest, /name:\s*"온라인 대전"/);
});

test("service worker never caches online API or mutation requests", () => {
  const worker = read("public/sw.js");

  assert.match(worker, /2026-07-31-pwa-no-web-splash-v3/);
  assert.match(worker, /icon-v2-512\.png/);
  assert.doesNotMatch(worker, /installed-splash-v2\.webp/);
  assert.match(worker, /\.then\(\(\) => self\.skipWaiting\(\)\)/);
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
  const lifecycleStyles = read("app/components/PwaLifecycle.module.css");
  const layout = read("app/layout.tsx");
  const globalStyles = read("app/globals.css");

  assert.match(lifecycle, /isSafeInstallScreen/);
  assert.match(lifecycle, /welcome-layer/);
  assert.match(lifecycle, /entryShell/);
  assert.match(lifecycle, /waitingWorker\.postMessage\(\{ type: "SKIP_WAITING" \}\)/);
  assert.match(lifecycle, /beforeinstallprompt/);
  assert.match(lifecycle, /function isMobileWeb/);
  assert.match(lifecycle, /userAgentData\?\.mobile/);
  assert.match(lifecycle, /Android\|iPhone\|iPad\|iPod\|Mobile/);
  assert.match(lifecycle, /mobileWeb && installPrompt/);
  assert.match(lifecycleStyles, /url\("\/pwa\/icon-v2-192\.png"\)/);
  assert.doesNotMatch(lifecycleStyles, /brand-dalmuti-crown\.png/);
  assert.match(layout, /viewportFit:\s*"cover"/);
  assert.match(layout, /appleWebApp/);
  assert.doesNotMatch(layout, /MobileSplash/);
  assert.doesNotMatch(layout, /PreferenceRuntime|SettingsDialog/);
  assert.doesNotMatch(layout, /InstalledPwaSplash/);
  assert.match(layout, /<PwaLifecycle \/>/);
  assert.doesNotMatch(globalStyles, /installed-pwa-splash/);
  assert.doesNotMatch(globalStyles, /installedPwaSplash/);
  assert.doesNotMatch(globalStyles, /installed-splash-v2\.webp/);
  assert.equal(
    fs.existsSync(
      path.join(root, "app/components/InstalledPwaSplash.tsx"),
    ),
    false,
  );
});

test("installed Android app launches immediately without a branded native splash", () => {
  const wrapper = JSON.parse(read("android-twa/twa-manifest.json"));
  const webManifest = read("app/manifest.ts");
  const customizer = read("android-twa/apply-native-customizations.ps1");
  const nativeLauncher = read(
    "android-twa/custom/java/LauncherActivity.java",
  );
  const apkVerifier = read("android-twa/verify-built-apk.ps1");
  const assetBuilder = read("scripts/build_android_splash_assets.py");
  const androidSplashTheme = read(
    "android-twa/custom/res/values-v31/styles.xml",
  );

  assert.match(wrapper.iconUrl, /icon-512\.png/);
  assert.equal(wrapper.backgroundColor, "#000000");
  assert.equal(wrapper.splashScreenFadeOutDuration, 0);
  assert.equal(wrapper.appVersionCode, 9);
  assert.equal(wrapper.appVersion, "1.0.8");
  assert.equal(wrapper.startUrl, "/?source=android-twa");
  assert.equal(wrapper.orientation, "default");
  assert.doesNotMatch(webManifest, /orientation:\s*"any"/);
  assert.match(customizer, /customRoot "java\\LauncherActivity\.java"/);
  assert.match(customizer, /android:screenOrientation="unspecified"/);
  assert.match(customizer, /DalmutiLaunchTheme/);
  assert.match(customizer, /@mipmap\/dalmuti_app_icon_v3/);
  assert.match(customizer, /SPLASH_IMAGE_DRAWABLE/);
  assert.match(customizer, /SPLASH_SCREEN_BACKGROUND_COLOR/);
  assert.match(customizer, /SPLASH_SCREEN_FADE_OUT_DURATION/);
  assert.match(customizer, /browserHelperSplashMetadataNames/);
  assert.match(customizer, /browserHelperSplashDisabled = \$true/);
  assert.match(customizer, /dalmuti-native-assets-v9\.json/);
  assert.match(customizer, /Native LauncherActivity hash mismatch after copy/);
  assert.match(customizer, /Native resource hash mismatch after copy/);
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
    /protected boolean shouldLaunchImmediately\(\) \{\s*return true;/,
  );
  assert.match(
    nativeLauncher,
    /super\.launch\(\s*builder,\s*customTabsCallback,\s*null,/,
  );
  assert.ok(
    nativeLauncher.indexOf("setRequestedOrientation(") <
      nativeLauncher.indexOf("super.onCreate(savedInstanceState)"),
    "wrapper orientation must be chosen before Browser Helper launches",
  );
  for (const brandedSplashCode of [
    /Handler/,
    /postDelayed/,
    /setContentView/,
    /FrameLayout/,
    /\.animate\(/,
    /getSplashScreen/,
    /setOnExitAnimationListener/,
    /launchTwaOnce/,
    /SPLASH_[A-Z_]+_MS/,
    /dalmuti_splash_v4/,
    /dalmuti_splash_glow_v4/,
  ]) {
    assert.doesNotMatch(nativeLauncher, brandedSplashCode);
  }
  assert.match(
    androidSplashTheme,
    /android:windowSplashScreenAnimatedIcon[^]*dalmuti_splash_os_black_v4/,
  );
  assert.match(
    androidSplashTheme,
    /android:windowSplashScreenIconBackgroundColor[^]*@android:color\/black/,
  );
  assert.match(
    androidSplashTheme,
    /android:windowSplashScreenAnimationDuration">0</,
  );
  assert.doesNotMatch(
    androidSplashTheme,
    /ic_launcher|transparent_v4|dalmuti_splash_v4|dalmuti_splash_glow_v4/,
  );
  for (const density of ["mdpi", "hdpi", "xhdpi", "xxhdpi", "xxxhdpi"]) {
    for (const file of [
      "dalmuti_app_icon_v3.png",
      "dalmuti_app_icon_maskable_v3.png",
    ]) {
      assert.equal(
        fs.existsSync(
          path.join(
            root,
            `android-twa/custom/res/mipmap-${density}/${file}`,
          ),
        ),
        true,
        `mipmap-${density}/${file} is missing`,
      );
    }
    for (const file of [
      "dalmuti_splash_v4.png",
      "dalmuti_splash_glow_v4.png",
    ]) {
      assert.equal(
        fs.existsSync(
          path.join(
            root,
            `android-twa/custom/res/drawable-${density}/${file}`,
          ),
        ),
        false,
        `drawable-${density}/${file} must not be packaged`,
      );
    }
  }
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-app-icon-v3.png"),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(root, "android-twa/assets/dalmuti-splash-v4.png"),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/mipmap-xxxhdpi/dalmuti_app_icon_v3.png",
      ),
    ),
    true,
  );
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/mipmap-anydpi-v26/dalmuti_app_icon_v3.xml",
      ),
    ),
    true,
  );
  assert.match(assetBuilder, /dalmuti-app-icon-v3\.png/);
  assert.match(assetBuilder, /dalmuti_app_icon_v3\.png/);
  assert.match(assetBuilder, /dalmuti_splash_v4\.png/);
  assert.doesNotMatch(
    assetBuilder,
    /SPLASH_SOURCE|save_splash_glow|ImageFilter/,
  );
  assert.equal(
    sha256("android-twa/assets/dalmuti-app-icon-v3.png"),
    "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946",
  );
  assert.match(apkVerifier, /dalmuti-native-assets-v9\.json/);
  assert.match(apkVerifier, /dump badging/);
  assert.match(apkVerifier, /dump resources/);
  assert.match(apkVerifier, /dump xmltree/);
  assert.match(apkVerifier, /dump --values resources/);
  assert.match(apkVerifier, /mipmap\/dalmuti_app_icon_v3/);
  assert.match(apkVerifier, /drawable\/dalmuti_splash_v4/);
  assert.match(apkVerifier, /drawable\/dalmuti_splash_glow_v4/);
  assert.match(apkVerifier, /drawable\/dalmuti_splash_os_black_v4/);
  assert.match(apkVerifier, /browserHelperSplashDisabled = \$true/);
  assert.match(apkVerifier, /SPLASH_IMAGE_DRAWABLE/);
  assert.match(apkVerifier, /SPLASH_SCREEN_BACKGROUND_COLOR/);
  assert.match(apkVerifier, /SPLASH_SCREEN_FADE_OUT_DURATION/);
  assert.match(
    apkVerifier,
    /launcherSourceSha256 = "8b5f1f17d5fb3c97f750af951fa5bfd13a3543921b50611bc472362ee45142f3"/,
  );
  assert.match(apkVerifier, /Key=0x0101062c/);
  assert.match(apkVerifier, /Key=0x01010630/);
  assert.match(apkVerifier, /style\/DalmutiLaunchTheme/);
  assert.match(apkVerifier, /Compiled LauncherActivity/);
  assert.match(apkVerifier, /Compiled Android 12\+ launch theme/);
  assert.match(apkVerifier, /resources\.arsc/);
  assert.match(apkVerifier, /appVersion = "1\.0\.8"/);
  assert.match(apkVerifier, /versionCode = 9/);
  assert.doesNotMatch(androidSplashTheme, /windowSplashScreenBrandingImage/);
  assert.equal(
    fs.existsSync(
      path.join(
        root,
        "android-twa/custom/res/drawable/dalmuti_splash_os_black_v4.xml",
      ),
    ),
    true,
  );
  for (const obsoleteFile of [
    "android-twa/custom/res/drawable-xxxhdpi/splash.png",
    "android-twa/custom/res/drawable-xxxhdpi/splash_glow.png",
    "android-twa/custom/res/mipmap-xxxhdpi/ic_launcher.png",
    "android-twa/custom/res/mipmap-xxxhdpi/ic_maskable.png",
    "android-twa/custom/res/mipmap-anydpi-v26/ic_launcher.xml",
    "android-twa/custom/res/drawable/dalmuti_splash_transparent.xml",
    "android-twa/custom/res/drawable/dalmuti_splash_transparent_v4.xml",
  ]) {
    assert.equal(fs.existsSync(path.join(root, obsoleteFile)), false);
  }
});

test(
  "native Android customizer copies and fingerprints versioned resources",
  { skip: process.platform !== "win32" },
  () => {
    const fixtureRoot = fs.mkdtempSync(
      path.join(os.tmpdir(), "dalmuti-native-customizer-"),
    );
    try {
      const resourceRoot = path.join(
        fixtureRoot,
        "app/src/main/res",
      );
      const launcherRoot = path.join(
        fixtureRoot,
        "app/src/main/java/lab/dclab/dalmuti",
      );
      fs.mkdirSync(
        path.join(resourceRoot, "drawable-xxxhdpi"),
        { recursive: true },
      );
      fs.mkdirSync(
        path.join(resourceRoot, "mipmap-xxxhdpi"),
        { recursive: true },
      );
      fs.mkdirSync(launcherRoot, { recursive: true });
      fs.writeFileSync(
        path.join(resourceRoot, "drawable-xxxhdpi/splash.png"),
        "obsolete",
      );
      fs.writeFileSync(
        path.join(
          resourceRoot,
          "drawable-xxxhdpi/dalmuti_splash_v4.png",
        ),
        "obsolete branded splash",
      );
      fs.writeFileSync(
        path.join(
          resourceRoot,
          "drawable-xxxhdpi/dalmuti_splash_glow_v4.png",
        ),
        "obsolete branded glow",
      );
      fs.writeFileSync(
        path.join(resourceRoot, "mipmap-xxxhdpi/ic_launcher.png"),
        "obsolete",
      );
      fs.writeFileSync(
        path.join(launcherRoot, "LauncherActivity.java"),
        "package lab.dclab.dalmuti;",
      );
      fs.writeFileSync(
        path.join(fixtureRoot, "build.gradle"),
        'classpath "com.android.tools.build:gradle:8.9.1"',
      );
      fs.writeFileSync(
        path.join(fixtureRoot, "app/src/main/AndroidManifest.xml"),
        `<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
  <application android:icon="@mipmap/ic_launcher"
      android:roundIcon="@mipmap/ic_launcher">
    <activity android:name="LauncherActivity"
        android:theme="@style/AppTheme"
        android:screenOrientation="default">
      <meta-data
          android:name="android.support.customtabs.trusted.SPLASH_IMAGE_DRAWABLE"
          android:resource="@drawable/splash" />
      <meta-data
          android:name="android.support.customtabs.trusted.SPLASH_SCREEN_BACKGROUND_COLOR"
          android:resource="@color/backgroundColor" />
      <meta-data
          android:name="android.support.customtabs.trusted.SPLASH_SCREEN_FADE_OUT_DURATION"
          android:value="@integer/splashScreenFadeOutDuration" />
    </activity>
  </application>
</manifest>`,
      );

      const result = spawnSync(
        "powershell.exe",
        [
          "-NoProfile",
          "-ExecutionPolicy",
          "Bypass",
          "-File",
          path.join(root, "android-twa/apply-native-customizations.ps1"),
          "-GeneratedProject",
          fixtureRoot,
        ],
        { encoding: "utf8" },
      );
      assert.equal(
        result.status,
        0,
        `${result.stdout}\n${result.stderr}`,
      );

      const customizedManifest = fs.readFileSync(
        path.join(fixtureRoot, "app/src/main/AndroidManifest.xml"),
        "utf8",
      );
      assert.match(
        customizedManifest,
        /android:icon="@mipmap\/dalmuti_app_icon_v3"/,
      );
      assert.match(
        customizedManifest,
        /android:roundIcon="@mipmap\/dalmuti_app_icon_v3"/,
      );
      assert.doesNotMatch(customizedManifest, /SPLASH_IMAGE_DRAWABLE/);
      assert.doesNotMatch(
        customizedManifest,
        /SPLASH_SCREEN_BACKGROUND_COLOR/,
      );
      assert.doesNotMatch(
        customizedManifest,
        /SPLASH_SCREEN_FADE_OUT_DURATION/,
      );
      assert.match(
        customizedManifest,
        /android:screenOrientation="unspecified"/,
      );
      assert.match(
        customizedManifest,
        /android:theme="@style\/DalmutiLaunchTheme"/,
      );

      const copiedIcon =
        "app/src/main/res/mipmap-xxxhdpi/dalmuti_app_icon_v3.png";
      assert.equal(
        sha256(path.relative(root, path.join(fixtureRoot, copiedIcon))),
        sha256(
          "android-twa/custom/res/mipmap-xxxhdpi/dalmuti_app_icon_v3.png",
        ),
      );
      const proof = JSON.parse(
        fs.readFileSync(
          path.join(
            fixtureRoot,
            "app/src/main/assets/dalmuti-native-assets-v9.json",
          ),
          "utf8",
        ),
      );
      assert.equal(proof.schemaVersion, 2);
      assert.equal(proof.appVersion, "1.0.8");
      assert.equal(proof.versionCode, 9);
      assert.equal(
        proof.launcherIconResource,
        "@mipmap/dalmuti_app_icon_v3",
      );
      assert.equal(
        proof.systemSplashResource,
        "@drawable/dalmuti_splash_os_black_v4",
      );
      assert.equal(proof.browserHelperSplashDisabled, true);
      assert.equal(
        proof.launcherSourceSha256,
        sha256NormalizedText(
          "android-twa/custom/java/LauncherActivity.java",
        ),
      );
      assert.match(proof.resourceTreeSha256, /^[0-9a-f]{64}$/);
      assert.equal(
        fs.existsSync(
          path.join(
            fixtureRoot,
            "app/src/main/res/drawable-xxxhdpi/splash.png",
          ),
        ),
        false,
      );
      for (const file of [
        "dalmuti_splash_v4.png",
        "dalmuti_splash_glow_v4.png",
      ]) {
        assert.equal(
          fs.existsSync(
            path.join(
              fixtureRoot,
              `app/src/main/res/drawable-xxxhdpi/${file}`,
            ),
          ),
          false,
        );
      }
      assert.equal(
        fs.existsSync(
          path.join(
            fixtureRoot,
            "app/src/main/res/mipmap-xxxhdpi/ic_launcher.png",
          ),
        ),
        false,
      );
    } finally {
      fs.rmSync(fixtureRoot, { recursive: true, force: true });
    }
  },
);

test("required offline and install assets exist", () => {
  for (const file of [
    "public/offline.html",
    "public/pwa/icon-192.png",
    "public/pwa/icon-512.png",
    "public/pwa/icon-maskable-512.png",
    "public/pwa/apple-touch-icon.png",
    "public/pwa/icon-v2-192.png",
    "public/pwa/icon-v2-512.png",
    "public/pwa/icon-v2-1024.png",
    "public/pwa/icon-maskable-v2-512.png",
    "public/pwa/apple-touch-icon-v2.png",
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
