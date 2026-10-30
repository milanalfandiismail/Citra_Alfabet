@echo off
cd /d "%~dp0"
title Evaluasi Citra Gestur BISINDO
echo ====================================================
echo  EVALUASI CITRA TUNGGAL ALFABET BISINDO (EXPLORER)
echo ====================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    set "PY_CMD=.\.venv\Scripts\python.exe"
) else if exist ".\venv\Scripts\python.exe" (
    set "PY_CMD=.\venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

%PY_CMD% test_screenshot.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ada kendala saat menjalankan program evaluasi citra.
    pause
)
