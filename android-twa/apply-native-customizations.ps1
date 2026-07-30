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
$launcherIconResource = "@mipmap/dalmuti_app_icon_v3"
$splashResource = "@drawable/dalmuti_splash_v4"
$nativeProofFileName = "dalmuti-native-assets-v7.json"

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
    "dalmuti_splash_transparent.xml"
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

$splashMetadataPattern = (
    '(?s)(<meta-data\b(?=[^>]*android:name="' +
    'android\.support\.customtabs\.trusted\.SPLASH_IMAGE_DRAWABLE"' +
    ')[^>]*android:resource=")[^"]+(")'
)
if (-not [regex]::IsMatch($manifestSource, $splashMetadataPattern)) {
    throw "Could not find Browser Helper's splash resource metadata."
}
$manifestSource = [regex]::Replace(
    $manifestSource,
    $splashMetadataPattern,
    "`${1}$splashResource`${2}"
)

if (
    -not $manifestSource.Contains(
        "android:icon=`"$launcherIconResource`""
    ) -or
    -not $manifestSource.Contains(
        "android:roundIcon=`"$launcherIconResource`""
    ) -or
    -not $manifestSource.Contains(
        "android:resource=`"$splashResource`""
    )
) {
    throw "Versioned DALMUTI icon or splash references were not installed."
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
$nativeProof = [ordered]@{
    schemaVersion = 1
    packageId = "lab.dclab.dalmuti"
    appVersion = "1.0.6"
    versionCode = 7
    launcherIconResource = $launcherIconResource
    splashResource = $splashResource
    iconSourceSha256 = "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946"
    splashSourceSha256 = "13fadbea989e85980994d185b44f4a4215f3df59e075d1bdf6056a820756631f"
    resourceTreeSha256 = $resourceTreeHash
}
$nativeProofPath = Join-Path $assetTarget $nativeProofFileName
[System.IO.File]::WriteAllText(
    $nativeProofPath,
    ($nativeProof | ConvertTo-Json),
    $utf8NoBom
)

Write-Output (
    "Applied DALMUTI Android assets v7 and wrote packaging proof: " +
    $nativeProofPath
)
