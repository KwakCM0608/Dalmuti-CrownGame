[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ApkPath
)

$ErrorActionPreference = "Stop"
$resolvedApk = (Resolve-Path -LiteralPath $ApkPath).Path
if ([System.IO.Path]::GetExtension($resolvedApk) -ne ".apk") {
    throw "Expected an APK file: $resolvedApk"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($resolvedApk)
try {
    $entries = @($archive.Entries)
    $proofEntries = @(
        $entries |
            Where-Object {
                $_.FullName -eq "assets/dalmuti-native-assets-v7.json"
            }
    )
    if ($proofEntries.Count -ne 1) {
        throw (
            "APK does not contain exactly one DALMUTI v7 packaging proof. " +
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
        packageId = "lab.dclab.dalmuti"
        appVersion = "1.0.6"
        versionCode = 7
        launcherIconResource = "@mipmap/dalmuti_app_icon_v3"
        splashResource = "@drawable/dalmuti_splash_v4"
        iconSourceSha256 = "5c953737fb31f5a8ed8e2d7f53a75681e5b37a0fcf8db55a743206260f6d7946"
        splashSourceSha256 = "13fadbea989e85980994d185b44f4a4215f3df59e075d1bdf6056a820756631f"
    }
    foreach ($entry in $expectedProof.GetEnumerator()) {
        if ([string]$proof.($entry.Key) -ne [string]$entry.Value) {
            throw "APK packaging proof mismatch: $($entry.Key)"
        }
    }
    if ([string]$proof.resourceTreeSha256 -notmatch '^[0-9a-f]{64}$') {
        throw "APK packaging proof has an invalid resource tree fingerprint."
    }

    $resourceEntryNames = @(
        $entries |
            Where-Object { $_.FullName.StartsWith("res/") } |
            ForEach-Object { [System.IO.Path]::GetFileName($_.FullName) }
    )
    $expectedResourceCounts = [ordered]@{
        "dalmuti_app_icon_v3.png" = 5
        "dalmuti_app_icon_maskable_v3.png" = 5
        "dalmuti_app_icon_v3.xml" = 1
        "dalmuti_splash_v4.png" = 5
        "dalmuti_splash_glow_v4.png" = 5
        "dalmuti_splash_transparent_v4.xml" = 1
    }
    foreach ($entry in $expectedResourceCounts.GetEnumerator()) {
        $actualCount = @(
            $resourceEntryNames |
                Where-Object { $_ -eq $entry.Key }
        ).Count
        if ($actualCount -lt $entry.Value) {
            throw (
                "APK is missing versioned native resource $($entry.Key): " +
                "expected at least $($entry.Value), found $actualCount"
            )
        }
    }

    foreach ($obsoleteName in @(
        "ic_launcher.png",
        "ic_maskable.png",
        "ic_launcher.xml",
        "splash.png",
        "splash_glow.png",
        "dalmuti_splash_transparent.xml"
    )) {
        if ($resourceEntryNames -contains $obsoleteName) {
            throw "APK still packages obsolete native resource: $obsoleteName"
        }
    }

    Write-Output (
        "Verified DALMUTI Android 1.0.6 (7): versioned icon, splash, " +
        "glow, and customization proof are packaged."
    )
} finally {
    $archive.Dispose()
}
