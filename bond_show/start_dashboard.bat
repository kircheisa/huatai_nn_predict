@echo off
setlocal

cd /d "%~dp0"

set "STATIC_PORT=8000"
set "NEWS_PORT=8001"
set "URL=http://localhost:%STATIC_PORT%/html_demo/bond_yield_dashboard.html"
set "STATIC_SERVER_CMD="
set "NEWS_SERVER_CMD="
set "PYTHON_BIN="

rem Python 探测顺序：显式 Anaconda 路径 > py 启动器 > python（避免命中商店stub）
if exist "D:\soft\Anaconda3\python.exe" (
  set "PYTHON_BIN=D:\soft\Anaconda3\python.exe"
  set "STATIC_SERVER_CMD=D:\soft\Anaconda3\python.exe -m http.server %STATIC_PORT% --bind 0.0.0.0"
  set "NEWS_SERVER_CMD=D:\soft\Anaconda3\python.exe news_proxy.py"
) else (
where python >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_BIN=python"
  set "STATIC_SERVER_CMD=python -m http.server %STATIC_PORT% --bind 0.0.0.0"
  set "NEWS_SERVER_CMD=python news_proxy.py"
) else (
  where py >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_BIN=py"
    set "STATIC_SERVER_CMD=py -m http.server %STATIC_PORT% --bind 0.0.0.0"
    set "NEWS_SERVER_CMD=py news_proxy.py"
  ) else (
    echo [ERROR] Python not found.
    echo Install Python and try again.
    echo.
    pause
    exit /b 1
  )
)
)
)

echo [INFO] Starting local static server...
echo [INFO] Working dir: %cd%
echo [INFO] Dashboard URL: %URL%
echo [INFO] News Proxy: http://localhost:%NEWS_PORT%/api/health
echo [INFO] LAN access: use this PC IP, e.g. http://<your-ip>:%STATIC_PORT%/html_demo/bond_yield_dashboard.html
echo [INFO] Static server command: %STATIC_SERVER_CMD%
echo [INFO] News proxy command: %NEWS_SERVER_CMD%
echo.

start "Bond Dashboard Static Server" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; %STATIC_SERVER_CMD%"
start "Bond Dashboard News Proxy" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; %NEWS_SERVER_CMD%"

ping 127.0.0.1 -n 3 >nul
start "" "%URL%"

echo [INFO] Browser open requested.
echo [INFO] If it did not open, visit:
echo %URL%
echo [INFO] If news search still fails, check:
echo http://127.0.0.1:%NEWS_PORT%/api/health
echo.
exit /b 0
