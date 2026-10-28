# PANDUAN MENJALANKAN KLASIFIKASI BISINDO CNN

## 1. Sumber Dataset (Kaggle)
Dataset resmi yang digunakan dalam proyek ini:
- **Tautan Kaggle**: [Indonesian Sign Language (BISINDO) oleh Agung MRF](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo/data)
- **Download via Kaggle CLI**:
  ```bash
  kaggle datasets download -d agungmrf/indonesian-sign-language-bisindo
  unzip indonesian-sign-language-bisindo.zip -d bisindo/
  ```

---

## 2. Menjalankan di Google Colab (Paling Cepat - GPU T4 Gratis)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file `klasifikasi_bisindo_cnn_pytorch.ipynb`.
3. Masuk ke menu **Runtime** > **Change runtime type** > Pilih **T4 GPU**.
4. Unggah file `bisindo.zip` ke panel Files (root `/content/`).
5. Jalankan semua sel (**Runtime** > **Run all**). Notebook akan mengekstrak zip otomatis dan melatih/memuat model dengan GPU T4.

---

## 3. Menjalankan di Lokal Windows (Akselerasi GPU AMD Radeon RX 5700 XT / NVIDIA)
1. Install dependensi dengan backend DirectML untuk GPU AMD:
   ```powershell
   pip install torch torchvision torch-directml opencv-python mediapipe matplotlib seaborn scikit-learn pillow
   ```
2. Jalankan Jupyter Lab / Notebook:
   ```powershell
   jupyter notebook klasifikasi_bisindo_cnn_pytorch.ipynb
   ```
3. Sel deteksi perangkat akan otomatis memunculkan:
   `🚀 [DEVICE] AMD GPU Terdeteksi via DirectML (AMD Radeon RX 5700 XT)`
   dan seluruh kalkulasi matriks tensor akan diproses oleh GPU AMD Anda.

---

## 4. Pengujian Kamera Real-Time & Screenshot
1. **Pengujian Kamera Real-Time (Webcam)**:
   ```powershell
   python test_camera.py
   ```
   - Mode Default: 1D-CNN Landmark Skeleton (60+ FPS, kebal latar belakang).
   - Tekan `M` untuk beralih antara 1D-CNN (Landmark), MobileNetV2, dan Custom 2D-CNN.
   - Tekan `+` / `-` untuk menyesuaikan ukuran kotak panduan tangan.
   - Tekan `Q` untuk keluar.

2. **Pengujian Screenshot GUI Interaktif**:
   ```powershell
   python test_screenshot.py
   ```
