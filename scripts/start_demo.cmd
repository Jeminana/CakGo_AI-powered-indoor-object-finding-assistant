@echo off
REM Double-click to start the CakGo demo (Grounding DINO + dev server + phone link).
REM Close this window or press Enter in it to stop everything.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_demo.ps1" %*
pause
