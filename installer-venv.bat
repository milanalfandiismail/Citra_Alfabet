@echo off
cd /d "%~dp0"
title Installer BISINDO (.venv + DirectML AMD GPU)
echo ==============================================================================
echo   INSTALLER BISINDO VIRTUAL ENVIRONMENT (.venv) + DIRECTML (AMD GPU)
echo   Menggunakan Python <= 3.11 dengan akselerasi GPU via DirectML
echo ==============================================================================
echo.

:: 1. Cek ketersediaan Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python tidak terdeteksi pada sistem PATH!
    echo Silakan install Python (disarankan versi 3.10 atau 3.11) dari https://www.python.org/
    echo Pastikan opsi "Add Python to PATH" dicentang saat instalasi.
    echo.
    pause
    exit /b 1
)

echo [1/4] Memeriksa versi Python...
python --version
python -c "import sys; sys.exit(0 if sys.version_info <= (3, 11) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [CATATAN] Anda menggunakan Python 3.12+.
    echo Rilis resmi library torch-directml dari Microsoft saat ini hanya mendukung Python hingga 3.11.
    echo Jika torch-directml gagal diinstal, sistem otomatis fallback ke mode CPU / CUDA standar.
)
echo.

:: 2. Buat virtual environment jika belum ada
if not exist ".\.venv\Scripts\python.exe" (
    echo [2/4] Membuat virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Gagal membuat virtual environment .venv!
        pause
        exit /b 1
    )
    echo Virtual environment berhasil dibuat di folder .venv.
) else (
    echo [2/4] Virtual environment (.venv) sudah ada, melanjutkan...
)
echo.

:: 3. Upgrade pip di dalam .venv
echo [3/4] Meng-upgrade pip di dalam .venv...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
echo.

:: 4. Install dependensi dengan DirectML
echo [4/4] Menginstal dependensi (termasuk torch-directml) dari requirements-directml.txt...
.\.venv\Scripts\python.exe -m pip install -r requirements-directml.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [PERINGATAN] Terjadi kendala saat instalasi requirements-directml.txt.
    echo Mencoba menginstal dependensi standar terlebih dahulu...
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo ==============================================================================
echo [SUKSES] Penyiapan virtual environment (.venv) selesai!
echo Anda dapat langsung menjalankan aplikasi kamera dengan: jalankan_kamera.bat
echo ==============================================================================
echo.
pause
