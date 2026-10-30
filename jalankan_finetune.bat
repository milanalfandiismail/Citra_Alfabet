@echo off
cd /d "%~dp0"
title BISINDO - Pelatihan & Fine-Tuning 2D-CNN MobileNetV2
echo ====================================================
echo  PELATIHAN FINE-TUNING MOBILENETV2 (ANTI-NOISE)
echo ====================================================
echo.

if exist ".\.venv\Scripts\python.exe" (
    set "PY_CMD=.\.venv\Scripts\python.exe"
) else if exist ".\venv\Scripts\python.exe" (
    set "PY_CMD=.\venv\Scripts\python.exe"
) else (
    set "PY_CMD=python"
)

%PY_CMD% train_mobilenet_finetune.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Pelatihan terhenti atau terjadi kendala.
    pause
) else (
    echo.
    echo Pelatihan selesai dengan sukses!
    pause
)
