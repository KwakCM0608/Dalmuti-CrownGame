[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApkPath,
    [string]$AaptPath
)

$ErrorActionPreference = "Stop"
$resolvedApk = (Resolve-Path -LiteralPath $ApkPath).Path
if ([System.IO.Path]::GetExtension($resolvedApk) -ne ".apk") {
    throw "Expected an APK file: $resolvedApk"
}

if ($AaptPath) {
    $resolvedAapt = (Resolve-Path -LiteralPath $AaptPath).Path
} else {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $sdkRoots = @(
        $env:ANDROID_SDK_ROOT,
        $env:ANDROID_HOME,
        (Join-Path $repoRoot ".android-tools\sdk")
    ) |
        Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -Unique
    $aaptCandidates = @(
        foreach ($sdkRoot in $sdkRoots) {
            Get-ChildItem `
                -LiteralPath (Join-Path $sdkRoot "build-tools") `
                -Filter "aapt.exe" `
                -Recurse `
                -File `
                -ErrorAction SilentlyContinue
        }
    )
    $resolvedAapt = (
        $aaptCandidates |
            Sort-Object `
                @{ Expression = { [version]$_.Directory.Name } } `
                -Descending |
            Select-Object -First 1
    ).FullName
}
if (-not $resolvedAapt -or -not (Test-Path -LiteralPath $resolvedAapt)) {
    throw (
        "Android aapt.exe was not found. Pass -AaptPath or configure " +
        "ANDROID_SDK_ROOT."
    )
}

$badgingLines = @(& $resolvedAapt dump badging $resolvedApk 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "aapt failed to read APK badging: $($badgingLines -join "`n")"
}
$badging = $badgingLines -join "`n"
if (
    $badging -notmatch (
        "package: name='lab\.dclab\.dalmuti' " +
        "versionCode='10' versionName='1\.0\.9'"
    )
) {
    throw "APK package or version does not match DALMUTI Android 1.0.9 (10)."
}

$resourceTableLines = @(& $resolvedAapt dump resources $resolvedApk 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw (
        "aapt failed to read the APK resource table: " +
        ($resourceTableLines -join "`n")
    )
}
$resourceTable = $resourceTableLines -join "`n"

$manifestTreeLines = @(
    & $resolvedAapt dump xmltree $resolvedApk AndroidManifest.xml 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw (
        "aapt failed to read the compiled Android manifest: " +
        ($manifestTreeLines -join "`n")
    )
}
$manifestTree = $manifestTreeLines -join "`n"

$resourceValuesLines = @(
    & $resolvedAapt dump --values resources $resolvedApk 2>&1
)
if ($LASTEXITCODE -ne 0) {
    throw (
        "aapt failed to read compiled Android resource values: " +
        ($resourceValuesLines -join "`n")
    )
}
$resourceValues = $resourceValuesLines -join "`n"

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($resolvedApk)
try {
    $entries = @($archive.Entries)
    $proofEntries = @(
        $entries |
            Where-Object {
                $_.FullName -eq "assets/dalmuti-native-assets-v10.json"
            }
    )
    if ($proofEntries.Count -ne 1) {
        throw (
            "APK does not contain exactly one DALMUTI v10 packaging proof. " +
            "Run apply-native-customizations.ps1 before building."
        )
    }

    $reader = New-Object System.IO.StreamReader(
        $proofEntries[0].Open(),
        [System.Text.Encoding]::UTF8,
        $true
    )
    try {
        $proof = $reader.ReadToEnd() | ConvertFrom-Json
    } finally {
        $reader.Dispose()
    }

    $expectedProof = [ordered]@{
        schemaVersion = 2
        packageId = "lab.dclab.dalmuti"
        appVersion = "1.0.9"
        versionCode = 10
        launcherIconResource = "@mipmap/dalmuti_app_icon_v4"
        systemSplashResource = "@drawable/dalmuti_splash_os_black_v4"
        browserHelperSplashDisabled = $true
        iconSourceSha256 = "80664605da730198b2d59d7d5beb3b1dbf7b837a49e9b560f03b8d73d403081c"
        launcherSourceSha256 = "8b5f1f17d5fb3c97f750af951fa5bfd13a3543921b50611bc472362ee45142f3"
    }
    foreach ($entry in $expectedProof.GetEnumerator()) {
        if ([string]$proof.($entry.Key) -ne [string]$entry.Value) {
            throw "APK packaging proof mismatch: $($entry.Key)"
        }
    }
    if ([string]$proof.resourceTreeSha256 -notmatch '^[0-9a-f]{64}$') {
        throw "APK packaging proof has an invalid resource tree fingerprint."
    }

    # AAPT2 intentionally replaces packaged res/ archive filenames with
    # compact identifiers. Verify logical Android resource names and density
    # variants through resources.arsc instead of relying on ZIP entry names.
    $expectedResourceCounts = [ordered]@{
        "lab.dclab.dalmuti:mipmap/dalmuti_app_icon_v4" = 7
        "lab.dclab.dalmuti:mipmap/dalmuti_app_icon_maskable_v4" = 6
        "lab.dclab.dalmuti:drawable/dalmuti_splash_os_black_v4" = 2
    }
    foreach ($entry in $expectedResourceCounts.GetEnumerator()) {
        $actualCount = [regex]::Matches(
            $resourceTable,
            [regex]::Escape($entry.Key)
        ).Count
        if ($actualCount -lt $entry.Value) {
            throw (
                "APK is missing versioned native resource $($entry.Key): " +
                "expected at least $($entry.Value), found $actualCount"
            )
        }
    }

    $themeSpec = [regex]::Match(
        $resourceTable,
        "spec resource (0x[0-9a-f]+) " +
            "lab\.dclab\.dalmuti:style/DalmutiLaunchTheme"
    )
    $osBlackSpec = [regex]::Match(
        $resourceTable,
        "spec resource (0x[0-9a-f]+) " +
            "lab\.dclab\.dalmuti:drawable/dalmuti_splash_os_black_v4"
    )
    $backgroundColorSpec = [regex]::Match(
        $resourceTable,
        "spec resource (0x[0-9a-f]+) " +
            "lab\.dclab\.dalmuti:color/backgroundColor"
    )
    if (
        -not $themeSpec.Success -or
        -not $osBlackSpec.Success -or
        -not $backgroundColorSpec.Success
    ) {
        throw "APK resource IDs for the black launch theme are missing."
    }
    $themeId = $themeSpec.Groups[1].Value
    $osBlackId = $osBlackSpec.Groups[1].Value
    $backgroundColorId = $backgroundColorSpec.Groups[1].Value

    $launcherThemePattern = (
        "(?s)E: activity.*?" +
        "A: android:theme[^`n]*=@$themeId.*?" +
        "A: android:name[^`n]*LauncherActivity"
    )
    if ($manifestTree -notmatch $launcherThemePattern) {
        throw "Compiled LauncherActivity does not use DalmutiLaunchTheme."
    }
    foreach ($metadataName in @(
        "SPLASH_IMAGE_DRAWABLE",
        "SPLASH_SCREEN_BACKGROUND_COLOR",
        "SPLASH_SCREEN_FADE_OUT_DURATION"
    )) {
        if ($manifestTree.Contains($metadataName)) {
            throw (
                "Compiled manifest still enables Browser Helper splash: " +
                $metadataName
            )
        }
    }
    $v31BlackThemePattern = (
        "(?s)resource $themeId " +
            "lab\.dclab\.dalmuti:style/DalmutiLaunchTheme: <bag>.*?" +
        "Key=0x0101062c[^`n]*$backgroundColorId.*?" +
        "Key=0x0101062d[^`n]*$osBlackId.*?" +
        "Key=0x0101062e[^`n]*0x00000000.*?" +
        "Key=0x01010630[^`n]*0x0106000c"
    )
    if ($resourceValues -notmatch $v31BlackThemePattern) {
        throw (
            "Compiled Android 12+ launch theme does not use the opaque " +
            "black system drawable and black icon background."
        )
    }

    foreach ($obsoleteResource in @(
        "mipmap/ic_launcher",
        "mipmap/ic_maskable",
        "drawable/splash",
        "drawable/splash_glow",
        "drawable/dalmuti_splash_v4",
        "drawable/dalmuti_splash_glow_v4",
        "drawable/dalmuti_splash_transparent",
        "drawable/dalmuti_splash_transparent_v4"
    )) {
        $obsoletePattern = (
            "lab\.dclab\.dalmuti:" +
            [regex]::Escape($obsoleteResource) +
            "(?=[:\s])"
        )
        if ($resourceTable -match $obsoletePattern) {
            throw (
                "APK still packages obsolete native resource: " +
                $obsoleteResource
            )
        }
    }

    Write-Output (
        "Verified DALMUTI Android 1.0.9 (10): immediate unbranded launch, " +
        "black Android 12+ system frame, and customization proof."
    )
} finally {
    $archive.Dispose()
}
