# Klasifikasi Citra Alfabet Bahasa Isyarat Indonesia (BISINDO) Menggunakan Convolutional Neural Network (CNN)

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![DirectML](https://img.shields.io/badge/DirectML-AMD%20Radeon-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C7B7)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-5C3EE8?logo=opencv)

Implementasi komprehensif sistem pengenalan 26 alfabet **Bahasa Isyarat Indonesia (BISINDO)** (A–Z) berbasis *Deep Learning*. Proyek ini menyajikan studi komparatif tiga arsitektur Convolutional Neural Network (CNN) yang terintegrasi penuh dari tahap prapemrosesan citra, ekstraksi fitur sendi 3D, pelatihan terpadu, hingga pengujian nyata melalui kamera webcam *real-time* dan evaluasi *screenshot*.

---

## 📊 Sumber Dataset (Kaggle)
Dataset yang digunakan dalam penelitian ini bersumber dari Kaggle:
- **Judul Dataset**: [Indonesian Sign Language (BISINDO)](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo/data)
- **Author**: **Agung MRF** (`agungmrf`)
- **Format**: Citra RGB resolusi tinggi yang terbagi menjadi:
  - **Data Latih (*Train*)**: 4.160 citra (~160 sampel per kelas)
  - **Data Validasi (*Validation*)**: 910 citra (~35 sampel per kelas)
  - **Jumlah Kelas**: 26 Kelas Alfabet (`A` s.d. `Z`)

### Cara Mengunduh Dataset:
1. **Opsi 1 - Menggunakan Kaggle CLI**:
   ```bash
   kaggle datasets download -d agungmrf/indonesian-sign-language-bisindo
   unzip indonesian-sign-language-bisindo.zip -d bisindo/
   ```
2. **Opsi 2 - Unduh Manual**:
   Kunjungi [Tautan Dataset Kaggle](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo/data), klik tombol **Download**, lalu ekstrak folder `images/` ke dalam direktori `bisindo/`.

---

## 📚 Pemetaan Modul Perkuliahan

| Modul Kuliah | Topik Silabus | Penerapan Konkret dalam Repositori Ini |
| :--- | :--- | :--- |
| **Modul 13** | **Deep Learning & CNN** | **Topik Utama Proyek**: Konvolusi 2D & 1D (`Conv2d`/`Conv1d`), feature maps, pooling layer, receptive field, dan *training loop* konvolusi. |
| **Modul 2** | **Preprocessing Data** | Resize seragam ($224 \times 224$), data augmentasi adaptif (rotasi $\pm 10^\circ$, color jitter tanpa horizontal flip), dan normalisasi z-score ImageNet. |
| **Modul 3** | **Fitur** | Ekstraksi representasi citra visual piksel serta ekstraksi 42 koordinat geometris 3D sendi tangan menggunakan MediaPipe. |
| **Modul 12** | **Neural Networks Dasar** | Fungsi aktivasi (ReLU, Softmax), fungsi rugi (*CrossEntropyLoss*), laju belajar (*learning rate*), serta optimasi *Backpropagation* & AdamW. |
| **Modul 15** | **Transfer Learning** | Pemanfaatan arsitektur *pre-trained* MobileNetV2 (ImageNet), pembekuan *feature extractor*, dan *fine-tuning* pada classifier head 26 kelas. |
| **Modul 6** | **Evaluasi Klasifikasi** | Pengujian komparatif menyeluruh melalui heatmap **Confusion Matrix $26 \times 26$**, akurasi, presisi, recall, dan F1-score. |

---

## 🔬 Tiga Arsitektur CNN yang Dibandingkan

```
                                [Dataset BISINDO (Kaggle)]
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
          [Jalur 2D-CNN Citra Piksel]                 [Jalur 1D-CNN Spasial Landmark]
          - Resize 224x224, Jitter, Rotasi            - MediaPipe HandLandmarker (42 Sendi)
          - Normalisasi ImageNet                      - Translasi Wrist Origin & Skala Bentang
                      │                                           │
          ┌───────────┴───────────┐                               │
          ▼                       ▼                               ▼
    [Model 1]               [Model 2]                       [Model 3]
  Custom 2D-CNN       Transfer Learning 2D-CNN        Spatial Keypoint 1D-CNN
  (From Scratch)            (MobileNetV2)                 (Landmark Skeleton)
```

1. **Model 1: Custom 2D-CNN (From Scratch)**:
   - 3 blok konvolusi berjenjang (`Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d`).
   - Mereduksi dimensi spasial secara adaptif ke $4 \times 4$ sebelum masuk ke *Fully Connected Classifier*.
2. **Model 2: Transfer Learning MobileNetV2**:
   - Memanfaatkan *Inverted Residual Block* dengan *Depthwise Separable Convolution* berbobot awal ImageNet.
   - Backbone dibekukan (*frozen*), lapisan klasifikasi akhir disesuaikan menjadi 26 neuron kelas BISINDO.
3. **Model 3: Spatial Keypoint 1D-CNN (Landmark Skeleton)**:
   - Mengoperasikan konvolusi 1D (`Conv1d`) pada sekuens koordinat 3D sendi tangan $(3, 42)$.
   - **Solusi Ilmiah Latar Belakang**: 100% kebal terhadap warna dinding ruangan, perabot, bayangan, maupun perubahan intensitas pencahayaan.

---

## 📈 Tabel Sintesis Hasil Komparasi

| Metrik Evaluasi | Model 1: Custom 2D-CNN | Model 2: MobileNetV2 | Model 3: Spatial 1D-CNN |
| :--- | :--- | :--- | :--- |
| **Tipe Input Data** | Piksel Citra ($224 \times 224 \times 3$) | Piksel Citra ($224 \times 224 \times 3$) | Koordinat 3D Sendi ($3 \times 42$) |
| **Ukuran Vektor Input** | 150.528 nilai | 150.528 nilai | **126 nilai** |
| **Arsitektur Konvolusi** | 3 Blok `Conv2d` (Scratch) | Inverted Residual `Conv2d` | 3 Blok `Conv1d` Spasial |
| **Jumlah Parameter Dilatih** | ~955.000 parameter | ~335.000 parameter | **~79.000 parameter** |
| **Ukuran File Bobot (.pth)** | 3,65 MB | 13,17 MB | **0,94 MB** |
| **Akurasi Validasi Dataset** | ~92.4% | **~98.1%** | ~97.7% |
| **Ketahanan Latar Belakang** | Lemah (Terkecoh warna dinding/kipas) | Lemah (Overfit pada latar studio) | **Sangat Tinggi (Kebal 100%)** |
| **Latensi Inferensi Webcam** | ~28 ms (35 FPS) | ~20 ms (50 FPS) | **~3 ms (60+ FPS)** |
| **Kelayakan Live Webcam** | Kurang Stabil | Kurang Stabil | **Paling Andal & Direkomendasikan** |

---

## 🚀 Panduan Menjalankan

### 1. Di Google Colab (Gratis GPU T4)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file `klasifikasi_bisindo_cnn_pytorch.ipynb`.
3. Masuk ke menu **Runtime** > **Change runtime type** > Pilih **T4 GPU**.
4. Unggah dataset `bisindo.zip` atau unduh langsung via Kaggle API.
5. Jalankan semua sel (**Runtime** > **Run all**).

### 2. Di Laptop / PC Lokal (Akselerasi AMD Radeon DirectML / NVIDIA CUDA)
1. Siapkan virtual environment dan pasang pustaka yang diperlukan:
   ```powershell
   pip install torch torchvision torch-directml opencv-python mediapipe matplotlib seaborn scikit-learn pillow
   ```
2. Jalankan notebook Jupyter:
   ```powershell
   jupyter notebook klasifikasi_bisindo_cnn_pytorch.ipynb
   ```

### 3. Pengujian Mandiri Kamera Real-Time (Webcam)
Jalankan skrip inferensi real-time:
```powershell
python test_camera.py
```
- **Mode Default**: 1D-CNN Landmark Skeleton (60+ FPS, kebal latar belakang).
- Tekan **`M`** untuk berganti model ke MobileNetV2 atau Custom 2D-CNN.
- Tekan **`+` / `-`** untuk mengatur ukuran kotak panduan tangan.
- Tekan **`X`** untuk beralih mode cermin (*mirror view*).
- Tekan **`Q`** untuk keluar.

### 4. Pengujian Evaluasi Tangkapan Layar (Screenshot GUI)
Jalankan skrip penilai screenshot otomatis:
```powershell
python test_screenshot.py
```
Menampilkan visualisasi skeleton sendi tangan beserta diagram batang probabilitas kelas tertinggi (*Top-5 Bar Chart*).

---

## 📁 Struktur Repositori

```text
├── klasifikasi_bisindo_cnn_pytorch.ipynb # Notebook riset utama (22 sel terpadu)
├── extract_landmarks.py                  # Skrip ekstraktor 42 sendi MediaPipe
├── train_landmark_cnn.py                 # Skrip pelatihan arsitektur 1D-CNN
├── test_camera.py                        # Aplikasi uji kamera webcam real-time
├── test_screenshot.py                    # Aplikasi GUI penguji tangkapan layar
├── hand_landmarker.task                  # Model aset MediaPipe HandLandmarker
├── panduan_alfabet_bisindo.jpg           # Panduan visual peragaan alfabet BISINDO
├── best_landmark_cnn.pth                 # Bobot terlatih Model 3 (0.94 MB)
├── best_mobilenetv2.pth                  # Bobot terlatih Model 2 (13.17 MB)
├── best_custom_cnn.pth                   # Bobot terlatih Model 1 (3.65 MB)
├── landmarks_train.npz                   # Fitur terkompresi data latih sendi (4.160 sampel)
├── landmarks_val.npz                     # Fitur terkompresi data validasi sendi (910 sampel)
├── LICENSE                               # Lisensi MIT (Milan Alfandi Ismail) & Third-Party Notice
├── RUN_INSTRUCTIONS.md                   # Panduan cepat instalasi & eksekusi
└── .gitignore                            # Konfigurasi pengecualian file remote
```

---

## 📜 Lisensi & Atribusi (Open Source Notice)
- **Lisensi Kode Utama**: Repositori ini dilisensikan di bawah [MIT License](LICENSE) - Hak Cipta &copy; 2026 **Milan Alfandi Ismail**. Bebas digunakan, dimodifikasi, dan didistribusikan untuk keperluan akademik maupun pengembangan perangkat lunak.
- **Atribusi Dataset (Third-Party Open Source Notice)**:
  - Dataset: [Indonesian Sign Language (BISINDO)](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo/data) oleh **Agung MRF** via Kaggle.
  - Digunakan untuk tujuan penelitian ilmiah, benchmarking model, dan edukasi sistem pengenalan bahasa isyarat. Seluruh hak cipta data citra mentah adalah milik kontributor dataset.
- **Pustaka & Model Eksternal**:
  - MediaPipe Hand Landmarker (Apache License 2.0 oleh Google LLC).
  - PyTorch & Torchvision (BSD-3-Clause License oleh Meta Platforms, Inc. & PyTorch Contributors).
