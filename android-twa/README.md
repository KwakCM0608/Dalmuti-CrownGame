# DALMUTI Android test wrapper

This directory contains the Bubblewrap/TWA project configuration used to build
the direct-install Android test APK. The signing key and Android SDK stay in
the ignored `.android-tools` directory.

The APK opens the deployed PWA at:

`https://dalmuti-crown-game.byj010608.chatgpt.site`

After Bubblewrap generates or updates the Android project, run
`apply-native-customizations.ps1` before building. It installs the versioned
Android-only launcher icon, keeps Android 12's mandatory first frame black and
unbranded, and reveals one complete DALMUTI splash from darkness before a
seamless TWA hand-off. The native reveal and fade budget is below 1.5 seconds.
The custom launcher reads
the system auto-rotate preference on every launch: sensor rotation is enabled
only when the device setting is enabled, otherwise the current portrait or
landscape orientation is locked in both the wrapper and the launched TWA.

The approved source artwork lives in `android-twa/assets`. Regenerate the
committed density-specific icon and splash resources with
`scripts/build_android_splash_assets.py` after replacing a versioned source.

The current test package is version `1.0.5` (`versionCode` 6), signed with the
same local test key so it can update the previous test APK in place.

This build is intended for private device testing. Public app-store
distribution requires separate rights clearance for the game name, artwork,
and other protected assets.
