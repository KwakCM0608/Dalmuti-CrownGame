[CmdletBinding()]
param(
    [string]$GeneratedProject = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$generatedRoot = (Resolve-Path -LiteralPath $GeneratedProject).Path
$customRoot = Join-Path $PSScriptRoot "custom"
$resourceSource = Join-Path $customRoot "res"
$resourceTarget = Join-Path $generatedRoot "app\src\main\res"
$assetTarget = Join-Path $generatedRoot "app\src\main\assets"
$manifestPath = Join-Path $generatedRoot "app\src\main\AndroidManifest.xml"
$buildGradlePath = Join-Path $generatedRoot "build.gradle"
$launcherTemplatePath = Join-Path $customRoot "java\LauncherActivity.java"
$launcherPath = Join-Path `
    $generatedRoot `
    "app\src\main\java\lab\dclab\dalmuti\LauncherActivity.java"
$launcherIconResource = "@mipmap/dalmuti_app_icon_v4"
$systemSplashResource = "@drawable/dalmuti_splash_os_black_v4"
$nativeProofFileName = "dalmuti-native-assets-v10.json"

foreach ($requiredPath in @(
    $resourceSource,
    $resourceTarget,
    $manifestPath,
    $buildGradlePath,
    $launcherTemplatePath,
    $launcherPath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required generated Android path is missing: $requiredPath"
    }
}

$obsoleteResourceNames = @(
    "ic_launcher.png",
    "ic_maskable.png",
    "ic_launcher.xml",
    "splash.png",
    "splash_glow.png",
    "dalmuti_splash_v4.png",
    "dalmuti_splash_glow_v4.png",
    "dalmuti_splash_transparent.xml",
    "dalmuti_splash_transparent_v4.xml"
)
Get-ChildItem -LiteralPath $resourceTarget -Recurse -File |
    Where-Object { $_.Name -in $obsoleteResourceNames } |
    Remove-Item -Force

Copy-Item `
    -Path (Join-Path $resourceSource "*") `
    -Destination $resourceTarget `
    -Recurse `
    -Force

Copy-Item `
    -LiteralPath $launcherTemplatePath `
    -Destination $launcherPath `
    -Force
$launcherSourceRawHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $launcherTemplatePath
).Hash.ToLowerInvariant()
$launcherTargetRawHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $launcherPath
).Hash.ToLowerInvariant()
if ($launcherSourceRawHash -ne $launcherTargetRawHash) {
    throw "Native LauncherActivity hash mismatch after copy."
}
$launcherCanonicalText = (
    [System.IO.File]::ReadAllText($launcherTemplatePath)
).Replace("`r`n", "`n").Replace("`r", "`n")
$launcherHashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
try {
    $launcherSourceHash = (
        [System.BitConverter]::ToString(
            $launcherHashAlgorithm.ComputeHash(
                $utf8NoBom.GetBytes($launcherCanonicalText)
            )
        )
    ).Replace("-", "").ToLowerInvariant()
} finally {
    $launcherHashAlgorithm.Dispose()
}

$manifestSource = Get-Content -LiteralPath $manifestPath -Raw
$launcherActivityPattern = '(?s)<activity\s+android:name="LauncherActivity".*?>'
$launcherActivityMatch = [regex]::Match(
    $manifestSource,
    $launcherActivityPattern
)
if (-not $launcherActivityMatch.Success) {
    throw "Could not find the Bubblewrap launcher activity to customize."
}

$launcherActivityTag = $launcherActivityMatch.Value
if ($launcherActivityTag -match 'android:screenOrientation="[^"]*"') {
    $launcherActivityTag = [regex]::Replace(
        $launcherActivityTag,
        'android:screenOrientation="[^"]*"',
        'android:screenOrientation="unspecified"'
    )
} else {
    $launcherActivityTag = $launcherActivityTag.Replace(
        '<activity android:name="LauncherActivity"',
        @'
<activity android:name="LauncherActivity"
            android:screenOrientation="unspecified"
'@
    )
}

if ($launcherActivityTag -match 'android:theme="[^"]*"') {
    $launcherActivityTag = [regex]::Replace(
        $launcherActivityTag,
        'android:theme="[^"]*"',
        'android:theme="@style/DalmutiLaunchTheme"'
    )
} else {
    $launcherActivityTag = $launcherActivityTag.Replace(
        '<activity android:name="LauncherActivity"',
        @'
<activity android:name="LauncherActivity"
            android:theme="@style/DalmutiLaunchTheme"
'@
    )
}

$manifestSource = $manifestSource.Replace(
    $launcherActivityMatch.Value,
    $launcherActivityTag
)

$applicationPattern = '(?s)<application\b[^>]*>'
$applicationMatch = [regex]::Match($manifestSource, $applicationPattern)
if (-not $applicationMatch.Success) {
    throw "Could not find the Android application tag to customize."
}
$applicationTag = $applicationMatch.Value
foreach ($attributeName in @("icon", "roundIcon")) {
    $attributePattern = "android:$attributeName=`"[^`"]*`""
    $attributeValue = "android:$attributeName=`"$launcherIconResource`""
    if ($applicationTag -match $attributePattern) {
        $applicationTag = [regex]::Replace(
            $applicationTag,
            $attributePattern,
            $attributeValue
        )
    } else {
        $applicationTag = $applicationTag.Replace(
            "<application",
            "<application $attributeValue"
        )
    }
}
$manifestSource = $manifestSource.Replace(
    $applicationMatch.Value,
    $applicationTag
)

$browserHelperSplashMetadataNames = @(
    "android.support.customtabs.trusted.SPLASH_IMAGE_DRAWABLE",
    "android.support.customtabs.trusted.SPLASH_SCREEN_BACKGROUND_COLOR",
    "android.support.customtabs.trusted.SPLASH_SCREEN_FADE_OUT_DURATION"
)
foreach ($metadataName in $browserHelperSplashMetadataNames) {
    $metadataPattern = (
        '(?s)\s*<meta-data\b(?=[^>]*android:name="' +
        [regex]::Escape($metadataName) +
        '")[^>]*/>'
    )
    $manifestSource = [regex]::Replace(
        $manifestSource,
        $metadataPattern,
        ""
    )
}

if (
    -not $manifestSource.Contains(
        "android:icon=`"$launcherIconResource`""
    ) -or
    -not $manifestSource.Contains(
        "android:roundIcon=`"$launcherIconResource`""
    )
) {
    throw "Versioned DALMUTI launcher icon references were not installed."
}
foreach ($metadataName in $browserHelperSplashMetadataNames) {
    if ($manifestSource.Contains($metadataName)) {
        throw "Browser Helper splash metadata was not removed: $metadataName"
    }
}
[System.IO.File]::WriteAllText($manifestPath, $manifestSource, $utf8NoBom)

$buildGradleSource = Get-Content -LiteralPath $buildGradlePath -Raw
$buildGradleSource = $buildGradleSource.Replace(
    "com.android.tools.build:gradle:8.9.1",
    "com.android.tools.build:gradle:8.10.1"
)
[System.IO.File]::WriteAllText(
    $buildGradlePath,
    $buildGradleSource,
    $utf8NoBom
)

$resourceFingerprintLines = @()
foreach (
    $sourceFile in (
        Get-ChildItem -LiteralPath $resourceSource -Recurse -File
    )
) {
    $relativePath = $sourceFile.FullName.Substring(
        $resourceSource.Length + 1
    )
    $targetFile = Join-Path $resourceTarget $relativePath
    if (-not (Test-Path -LiteralPath $targetFile)) {
        throw "Native resource was not copied: $relativePath"
    }
    $sourceHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $sourceFile.FullName
    ).Hash.ToLowerInvariant()
    $targetHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $targetFile
    ).Hash.ToLowerInvariant()
    if ($sourceHash -ne $targetHash) {
        throw "Native resource hash mismatch after copy: $relativePath"
    }
    $normalizedPath = $relativePath.Replace("\", "/")
    $resourceFingerprintLines += "$normalizedPath`:$sourceHash"
}
$fingerprintPayload = (
    $resourceFingerprintLines |
        Sort-Object
) -join "`n"
$fingerprintBytes = [System.Text.Encoding]::UTF8.GetBytes(
    $fingerprintPayload
)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    $resourceTreeHash = (
        [System.BitConverter]::ToString(
            $sha256.ComputeHash($fingerprintBytes)
        )
    ).Replace("-", "").ToLowerInvariant()
} finally {
    $sha256.Dispose()
}

New-Item -ItemType Directory -Path $assetTarget -Force | Out-Null
Get-ChildItem `
    -LiteralPath $assetTarget `
    -Filter "dalmuti-native-assets-v*.json" `
    -File `
    -ErrorAction SilentlyContinue |
    Remove-Item -Force
$nativeProof = [ordered]@{
    schemaVersion = 2
    packageId = "lab.dclab.dalmuti"
    appVersion = "1.0.9"
    versionCode = 10
    launcherIconResource = $launcherIconResource
    systemSplashResource = $systemSplashResource
    browserHelperSplashDisabled = $true
    iconSourceSha256 = "80664605da730198b2d59d7d5beb3b1dbf7b837a49e9b560f03b8d73d403081c"
    launcherSourceSha256 = $launcherSourceHash
    resourceTreeSha256 = $resourceTreeHash
}
$nativeProofPath = Join-Path $assetTarget $nativeProofFileName
[System.IO.File]::WriteAllText(
    $nativeProofPath,
    ($nativeProof | ConvertTo-Json),
    $utf8NoBom
)

Write-Output (
    "Applied DALMUTI Android assets v10 with Browser Helper splash disabled: " +
    $nativeProofPath
)
