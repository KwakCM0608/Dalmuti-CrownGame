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
$launcherPath = Join-Path `
    $generatedRoot `
    "app\src\main\java\lab\dclab\dalmuti\LauncherActivity.java"

foreach ($requiredPath in @(
    $resourceSource,
    $resourceTarget,
    $manifestPath,
    $buildGradlePath,
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

$launcherSource = Get-Content -LiteralPath $launcherPath -Raw
$launcherSource = $launcherSource.Replace(
    "ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED",
    "ActivityInfo.SCREEN_ORIENTATION_FULL_USER"
)
[System.IO.File]::WriteAllText($launcherPath, $launcherSource, $utf8NoBom)

$manifestSource = Get-Content -LiteralPath $manifestPath -Raw
$plainActivity = '<activity android:name="LauncherActivity"'
$brandedActivity = @'
<activity android:name="LauncherActivity"
            android:screenOrientation="fullUser"
            android:theme="@style/DalmutiLaunchTheme"
'@

if (
    $manifestSource.Contains('android:screenOrientation="fullUser"') -and
    $manifestSource.Contains('android:theme="@style/DalmutiLaunchTheme"')
) {
    # Already customized.
} elseif ($manifestSource.Contains($plainActivity)) {
    $manifestSource = $manifestSource.Replace($plainActivity, $brandedActivity)
} else {
    throw "Could not find the Bubblewrap launcher activity to customize."
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

Write-Output "Applied DALMUTI splash branding and user-controlled rotation."
