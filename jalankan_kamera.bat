@echo off
cd /d "%~dp0"
title BISINDO Real-Time Camera
echo ====================================================
echo  MEMULAI APLIKASI KAMERA BISINDO (1D-CNN + 2D-CNN)
echo ====================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    set "PY_CMD=.\.venv\Scripts\python.exe"
) else if exist ".\venv\Scripts\python.exe" (
    set "PY_CMD=.\venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

%PY_CMD% test_camera.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ada kendala saat menjalankan program.
    pause
)

