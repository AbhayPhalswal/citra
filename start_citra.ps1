<#
=============================================================================
START CITRA (manual launcher)
=============================================================================
Starts Citra on this Windows machine — double-clicked from the "Start
Citra" desktop shortcut (created by create_desktop_shortcut.ps1), or run
directly from PowerShell.

This is now the ONLY launch path on Windows. Login-triggered autostart
(setup_autostart.ps1 / AUTOSTART.md) was removed once the always-on
target moved to a Raspberry Pi — see CITRA_ARCHITECTURE.md. Keeping a
Windows box running 24/7 was never the plan, and a Scheduled Task that
only fires at login can't deliver always-on anyway, so it was carrying
maintenance cost for a capability the Pi is meant to provide properly.
Recover it from git history if a Windows always-on host is ever wanted.

Starts nothing if Citra is ALREADY running. That check is the main reason
this script exists rather than the shortcut just launching python
directly: running two copies at once is not a harmless no-op — two
voice-assistant instances fight over the same microphone and both call
the same APIs, which was measured degrading response time by roughly an
order of magnitude (transcription 0.9s -> 8.1s, routing ~1s -> 14.8s)
during a real session where duplicates were running unnoticed. A launcher
a person might double-click twice needs to be safe against exactly that.

Launches each service under watchdog.py with output redirected to a log
file, since pythonw.exe has no console of its own.
=============================================================================
#>

$ProjectRoot = "C:\Users\HP\Desktop\jarvis_room"
$PythonW = "$ProjectRoot\jarvis_venv\Scripts\pythonw.exe"
$Watchdog = "$ProjectRoot\watchdog.py"

$Services = @(
    @{ Name = "Citra Voice Assistant"; Script = "jarvis_voice_assistant.py"; Log = "jarvis_voice_assistant.log" },
    @{ Name = "Citra UI Server";       Script = "citra_ui_server.py";        Log = "citra_ui_server.log" }
)

foreach ($svc in $Services) {
    # Already running? Match on the script filename appearing in any
    # python process's command line — the same signal used to diagnose
    # duplicates by hand, just automated.
    $running = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' or Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -like "*$($svc.Script)*" }

    if ($running) {
        Write-Host "$($svc.Name) is already running - leaving it alone."
        continue
    }

    # Nested quoting: watchdog.py takes the whole inner command as ONE
    # argv element, plus a log path. Each layer was verified end-to-end
    # (constructed, then actually run, then the log checked for correct
    # sys.argv) rather than assumed — it fails silently if wrong, so
    # re-test the same way before hand-editing this.
    $innerCommand = '"' + $PythonW + '" "' + $ProjectRoot + '\' + $svc.Script + '"'
    $escapedInner = $innerCommand.Replace('"', '\"')
    $watchdogArgs = '"' + $Watchdog + '" "' + $escapedInner + '" "' + $ProjectRoot + '\' + $svc.Log + '"'
    Start-Process -FilePath $PythonW -ArgumentList $watchdogArgs -WorkingDirectory $ProjectRoot -WindowStyle Hidden
    Write-Host "Started $($svc.Name)."
}

Write-Host ""
# The old message here claimed "~10-15 seconds", which is wrong in the
# case that actually matters: the log line "Citra is ready. Listening for
# the wake word..." appears ~2s in, but Whisper, the semantic router and
# Silero VAD keep loading in background threads — measured at 49.4s on a
# cold start (10.3s warm, once the model files are in the OS cache). A
# wake word spoken in that window is heard and then goes nowhere, which
# reads exactly like "Citra is broken". Use the Citra Control panel to
# watch the real stage rather than guessing from a fixed number here.
Write-Host "Citra needs up to ~50 seconds on a cold start to finish loading her models"
Write-Host "before she can answer (about 10 seconds if she's been run recently)."
Write-Host "Open the 'Citra Control' desktop shortcut to watch her startup progress."
Start-Sleep -Seconds 3
