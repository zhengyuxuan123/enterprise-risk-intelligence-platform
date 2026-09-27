@echo off
setlocal
cd /d "%~dp0"

echo.
echo === Enterprise Risk Platform ===
echo.

echo [1/2] Starting Python Agent on port 8081...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-python-agent.ps1"
if errorlevel 1 (
  echo.
  echo [X] Python Agent did not start. Check whether port 8081 is already in use.
  exit /b 1
)

echo [2/2] Starting Spring Boot on port 8080...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-backend.ps1"
if errorlevel 1 (
  echo.
  echo [X] Spring Boot did not start. Check whether port 8080 is already in use.
  exit /b 1
)

echo.
echo [OK] System is ready: http://localhost:8080
echo      Login: admin / Admin@123
start "" "http://localhost:8080"
exit /b 0
