# DALMUTI Android test wrapper

This directory contains the Bubblewrap/TWA project configuration used to build
the direct-install Android test APK. The signing key and Android SDK stay in
the ignored `.android-tools` directory.

The APK opens the deployed PWA at:

`https://dalmuti-crown-game.byj010608.chatgpt.site`

The wrapper launch path is `/?source=android-twa`. The query is an initial
surface discriminator only, so the web client can synchronously suppress any
PWA-only splash without changing game routing or state.

After Bubblewrap generates or updates the Android project, run
`apply-native-customizations.ps1` before building. It installs the versioned
Android-only launcher icon and removes Browser Helper's branded splash
metadata. Android 12+ still requires a platform-owned starting window, so its
background and 1 dp placeholder icon are both opaque black; the wrapper adds
no artwork, animation, or delay. Android 8 through 11 use the same black window
background while Browser Helper opens the TWA immediately. A slow browser bind
can therefore leave a plain black frame visible briefly, but no custom splash
image is packaged or shown. The custom launcher reads
the system auto-rotate preference on every launch: sensor rotation is enabled
only when the device setting is enabled, otherwise the current portrait or
landscape orientation is locked in both the wrapper and the launched TWA.

The approved source artwork lives in `android-twa/assets`. Regenerate the
committed density-specific launcher icons with
`scripts/build_android_splash_assets.py` after replacing the versioned icon
source. The historical splash source remains unreferenced and is not packaged.

The current test package is version `1.0.9` (`versionCode` 10), signed with the
same local test key so it can update the previous test APK in place.
After building, run `verify-built-apk.ps1 -ApkPath <path-to-apk>`. It requires
the v10 customization proof, the borderless versioned icon v4, and the black Android 12+
placeholder. It also rejects Browser Helper splash metadata and branded splash
or glow resources.

This build is intended for private device testing. Public app-store
distribution requires separate rights clearance for the game name, artwork,
and other protected assets.
