<#
=============================================================================
CREATE CITRA DESKTOP SHORTCUTS
=============================================================================
Puts two shortcuts on the Desktop:
  - "Start Citra"    -> start_citra.ps1 (start-only, no visible window)
  - "Citra Control"  -> start_citra_control.ps1 (opens the Control Panel
                         web app for starting/stopping/restarting either
                         service, and links to Citra's own UI)
Run this once; safe to re-run (it overwrites existing shortcuts rather
than erroring or duplicating).

-WindowStyle Hidden plus -ExecutionPolicy Bypass on each shortcut target:
Bypass is scoped to THIS ONE invocation only — it does not change the
machine's execution policy, and is the standard way to let a local
.ps1 run from a shortcut without requiring the user to loosen their
system-wide policy setting.
=============================================================================
#>

$ProjectRoot = "C:\Users\HP\Desktop\jarvis_room"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell

$Shortcuts = @(
    @{
        Launcher    = "$ProjectRoot\start_citra.ps1"
        Path        = "$DesktopPath\Start Citra.lnk"
        Description = "Start Citra if she isn't already running"
        # shell32.dll,138 is the standard blue/white "power" style icon — a
        # recognizable "turn this on" affordance rather than the generic
        # PowerShell console icon, which would misrepresent what this does.
        Icon        = "shell32.dll,138"
    },
    @{
        Launcher    = "$ProjectRoot\start_citra_control.ps1"
        Path        = "$DesktopPath\Citra Control.lnk"
        Description = "Open the Citra Control Panel (start/stop/restart Citra and her UI, and a link to Citra's own UI)"
        # shell32.dll,317 is a dial/settings-style icon — distinct from the
        # plain power icon above, since this shortcut opens a control
        # surface rather than just starting something.
        Icon        = "shell32.dll,317"
    }
)

foreach ($sc in $Shortcuts) {
    if (-not (Test-Path $sc.Launcher)) {
        Write-Error "Can't find $($sc.Launcher)"
        continue
    }

    $shortcut = $shell.CreateShortcut($sc.Path)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$($sc.Launcher)`""
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.Description = $sc.Description
    $shortcut.IconLocation = $sc.Icon
    $shortcut.Save()

    Write-Host "Created: $($sc.Path)"
}

Write-Host ""
Write-Host "'Start Citra' is safe to click twice - it won't start a second copy."
Write-Host "'Citra Control' opens a web panel to start/stop/restart either service and jump to Citra's UI."
