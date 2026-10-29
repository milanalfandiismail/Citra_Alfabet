@echo off
title BISINDO Real-Time Camera
echo ====================================================
echo  MEMULAI APLIKASI KAMERA BISINDO (1D-CNN + 2D-CNN)
echo ====================================================
echo.
.\.venv\Scripts\python.exe test_camera.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ada kendala saat menjalankan program.
    pause
)
