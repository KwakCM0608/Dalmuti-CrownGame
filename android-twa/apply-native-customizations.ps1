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
$manifestPath = Join-Path $generatedRoot "app\src\main\AndroidManifest.xml"
$buildGradlePath = Join-Path $generatedRoot "build.gradle"
$launcherTemplatePath = Join-Path $customRoot "java\LauncherActivity.java"
$launcherPath = Join-Path `
    $generatedRoot `
    "app\src\main\java\lab\dclab\dalmuti\LauncherActivity.java"

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

Write-Output "Applied the DALMUTI icon, single splash, and system-aware rotation policy."
