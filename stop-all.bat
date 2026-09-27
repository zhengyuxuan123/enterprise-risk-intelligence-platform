@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-backend.ps1" -Force
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-python-agent.ps1" -Force
echo.
echo [OK] Project application services are stopped.
exit /b 0
