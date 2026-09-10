<#
=============================================================================
START CITRA CONTROL (manual launcher)
=============================================================================
Launches citra_control_panel.py — double-clicked from the "Citra Control"
desktop shortcut. Unlike start_citra.ps1, there's no "already running" check
here: citra_control_panel.py handles that itself (it tries to bind its port
and, if something's already listening there, just opens a browser tab at
the existing instance instead of starting a second one). So this script's
only job is to launch it hidden, same pythonw-has-no-console reasoning as
start_citra.ps1.
=============================================================================
#>

$ProjectRoot = "C:\Users\HP\Desktop\jarvis_room"
$PythonW = "$ProjectRoot\jarvis_venv\Scripts\pythonw.exe"
$Script = "$ProjectRoot\citra_control_panel.py"

Start-Process -FilePath $PythonW -ArgumentList "`"$Script`"" -WorkingDirectory $ProjectRoot -WindowStyle Hidden
