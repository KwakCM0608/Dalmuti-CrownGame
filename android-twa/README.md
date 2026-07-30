# DALMUTI Android test wrapper

This directory contains the Bubblewrap/TWA project configuration used to build
the direct-install Android test APK. The signing key and Android SDK stay in
the ignored `.android-tools` directory.

The APK opens the deployed PWA at:

`https://dalmuti-crown-game.byj010608.chatgpt.site`

After Bubblewrap generates or updates the Android project, run
`apply-native-customizations.ps1` before building. It applies the branded
Android 12+ splash resources and makes rotation follow the device user's
auto-rotate/rotation-lock preference. Regenerate the committed splash assets
with `scripts/build_android_splash_assets.py` when the crown artwork changes.

This build is intended for private device testing. Public app-store
distribution requires separate rights clearance for the game name, artwork,
and other protected assets.
