@echo off
cd /d "%~dp0"
title Installer BISINDO (Standar - Tanpa DirectML)
echo ==============================================================================
echo   INSTALLER DEPENDENSI BISINDO (STANDAR / UNIVERSAL - TANPA DIRECTML)
echo   Cocok untuk CPU, NVIDIA GPU (CUDA), atau versi Python terbaru (3.12+)
echo ==============================================================================
echo.

:: 1. Cek ketersediaan Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python tidak terdeteksi pada sistem PATH!
    echo Silakan unduh dan install Python dari https://www.python.org/
    echo Pastikan opsi "Add Python to PATH" dicentang saat instalasi.
    echo.
    pause
    exit /b 1
)

echo [1/3] Memeriksa versi Python...
python --version
echo.

echo [2/3] Meng-upgrade pip...
python -m pip install --upgrade pip
echo.

echo [3/3] Menginstal dependensi standar dari requirements.txt...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Terjadi kegagalan saat menginstal paket dependensi!
    echo.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo [SUKSES] Seluruh dependensi standar berhasil diinstal!
echo Anda dapat menjalankan aplikasi kamera dengan: jalankan_kamera.bat
echo ==============================================================================
echo.
pause
