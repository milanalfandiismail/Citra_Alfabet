# PANDUAN MENJALANKAN KLASIFIKASI BISINDO CNN

### 1. Menjalankan di Google Colab (Paling Cepat - GPU T4 Gratis)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Upload file `klasifikasi_bisindo_cnn_pytorch.ipynb`.
3. Masuk ke menu **Runtime** > **Change runtime type** > Pilih **T4 GPU**.
4. Upload file `bisindo.zip` ke panel Files (root `/content/`).
5. Jalankan semua sel (**Runtime** > **Run all**). Notebook akan mengekstrak zip otomatis dan melatih kedua model dengan GPU T4.

### 2. Menjalankan di Lokal Windows (Akselerasi GPU AMD Radeon RX 5700 XT)
1. Install PyTorch dengan backend DirectML untuk GPU AMD:
   ```powershell
   pip install torch torchvision torch-directml matplotlib seaborn scikit-learn pillow
   ```
2. Jalankan Jupyter Lab / Notebook:
   ```powershell
   jupyter notebook klasifikasi_bisindo_cnn_pytorch.ipynb
   ```
3. Sel deteksi perangkat akan otomatis memunculkan:
   `🚀 [DEVICE] AMD GPU Terdeteksi via DirectML (AMD Radeon RX 5700 XT)`
   dan seluruh kalkulasi matriks tensor akan diproses oleh GPU AMD Anda.