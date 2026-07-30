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
Android-only launcher icon, turns Android 12's mandatory first frame into the
black opening frame of the animation, and reveals one complete DALMUTI splash
from darkness before a seamless TWA hand-off. The opaque black system drawable
prevents Android from substituting the launcher icon, while the controlled exit
fade makes the native reveal feel continuous. The controlled native-to-Chrome
transition is about 1.8 seconds; a slow browser bind or web load can keep the
same final splash frame visible longer without replaying the branding.
The custom launcher reads
the system auto-rotate preference on every launch: sensor rotation is enabled
only when the device setting is enabled, otherwise the current portrait or
landscape orientation is locked in both the wrapper and the launched TWA.

The approved source artwork lives in `android-twa/assets`. Regenerate the
committed density-specific icon and splash resources with
`scripts/build_android_splash_assets.py` after replacing a versioned source.

The current test package is version `1.0.7` (`versionCode` 8), signed with the
same local test key so it can update the previous test APK in place.
After building, run `verify-built-apk.ps1 -ApkPath <path-to-apk>`. It requires
the v8 customization proof and the versioned icon v3, splash v4, and glow
resources
inside the APK, and rejects the previous resource names.

This build is intended for private device testing. Public app-store
distribution requires separate rights clearance for the game name, artwork,
and other protected assets.
