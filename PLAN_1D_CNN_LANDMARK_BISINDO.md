# Rencana Kerja: Implementasi 1D-CNN Landmark BISINDO (Dual-Mode Presisi Tinggi)

## 1. Analisis & Landasan Keputusan

Berdasarkan pengujian menyeluruh:
- Model 2D-CNN (`best_mobilenetv2.pth`) mencapai **98.1% akurasi** pada data validasi dataset asli (`bisindo/images/val`), namun mengalami *domain shift* dan *background overfitting* ketika diuji pada 8 tangkapan webcam nyata (<20% confidence).
- CNN 2D menghafal piksel dinding studio putih dan seragam aktor asli.
- Solusi terbaik yang tetap **100% berakar pada CNN** (memenuhi silabus Kuliah/Skripsi: *Convolutional Neural Network*) adalah **1D-CNN Spatial Keypoint**.
- Koordinat 21 sendi tangan $(x, y, z)$ dinormalisasi relatif terhadap pergelangan tangan $(0,0,0)$ dan rentang tangan $(1.0)$, menjadikannya kebal 100% terhadap warna dinding merah, kipas angin, dan pencahayaan ruangan.

---

## 2. Arsitektur Model 1D-CNN (`SignLandmarkCNN`)

```text
Input Koordinat Sendi: (Batch, Channels=3, Length=42)
  ├── Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1) + BatchNorm1d + ReLU + Dropout(0.2)
  ├── Conv1d(64, 128, kernel_size=3, padding=1) + BatchNorm1d + ReLU + Dropout(0.2)
  ├── Conv1d(128, 256, kernel_size=3, padding=1) + BatchNorm1d + ReLU + Dropout(0.2)
  ├── AdaptiveAvgPool1d(1) -> Flatten (256)
  ├── Linear(256, 128) + ReLU + Dropout(0.3)
  └── Linear(128, 26) -> Softmax Probabilitas Alfabet A-Z
```

---

## 3. Langkah Eksekusi Teknis

### Langkah 1: Ekstraksi Koordinat Landmark dari Dataset Asli
- Script: `extract_landmarks.py`
- Menjalankan MediaPipe Task pada seluruh gambar `bisindo/images/train/` dan `val/`.
- Menyimpan hasil fitur ke file ringkas:
  - `landmarks_train.npz`
  - `landmarks_val.npz`
- Durasi: ~1–2 menit (otomatis di latar belakang).

### Langkah 2: Pelatihan 1D-CNN (`train_landmark_cnn.py`)
- Melatih model `SignLandmarkCNN` dengan PyTorch (GPU AMD Radeon DirectML / CUDA / CPU).
- Optimizer: AdamW ($lr=0.001$), CrossEntropyLoss, EarlyStopping.
- Karena data berbentuk matriks koordinat numerik (~5MB), pelatihan 40-50 epoch selesai dalam **~20-30 detik** dengan akurasi target **>95%**.
- Bobot disimpan ke: `best_landmark_cnn.pth`.

### Langkah 3: Integrasi Dual-Mode pada `test_camera.py`
- Menjadikan **1D-CNN Landmark** sebagai mode default presisi tinggi di webcam.
- Visualisasi skeleton tangan interaktif (tulang & sendi digambar di atas kamera).
- Tombol `[M]` untuk beralih antara:
  1. `1D-CNN Landmark (Presisi Tinggi & Kebal Background)` [Utama]
  2. `MobileNetV2 2D-CNN (Transfer Learning)` [Bahan Uji Komparasi]
  3. `Custom 2D-CNN (From Scratch)` [Bahan Uji Komparasi]

### Langkah 4: Pengujian Screenshot Kamu (`test_screenshot.py`)
- Script yang mengevaluasi langsung file-file screenshot kamu di folder `.user_uploaded/` untuk membuktikan peningkatan akurasi dari <20% menjadi akurasi tinggi.
