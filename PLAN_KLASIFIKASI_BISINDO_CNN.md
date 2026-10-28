# RENCANA KERJA & ARSITEKTUR IMPLEMENTASI (MASTER PLAN)
## Proyek: Klasifikasi Citra Alfabet Bahasa Isyarat Indonesia (BISINDO) Menggunakan Convolutional Neural Network (CNN)
**Platform Akselerasi: PyTorch + AMD DirectML (Lokal) & CUDA (Google Colab)**

---

### 1. RINGKASAN & PRINSIP DESAIN
- **Domain Masalah**: Pengenalan citra gestur tangan untuk 26 alfabet BISINDO (A–Z) berbasis Computer Vision.
- **Dataset**: `bisindo/images` (RGB 640x640), terdiri dari 26 kelas:
  - Data Latih (*Train*): ~350 citra per kelas (Total: ~9.100 citra).
  - Data Validasi (*Val*): ~88 citra per kelas (Total: ~2.300 citra).
- **Mekanisme Akselerasi Hardware GPU**:
  - **Lokal Windows**: Menggunakan **GPU AMD Radeon RX 5700 XT (8 GB VRAM)** via Microsoft DirectML (`torch-directml`).
  - **Google Colab Cloud**: Menggunakan **GPU NVIDIA Tesla T4 (16 GB VRAM)** via `torch.cuda`.
- **Komparasi 2 Model**:
  1. *Custom CNN from Scratch* (Sesuai Materi 13).
  2. *Transfer Learning MobileNetV2* (Sesuai Materi 15).

---

### 2. PEMETAAN MATERI KULIAH (.TXT) KE PIPELINE PYTORCH

| Modul Materi Kuliah | Relevansi Teknis di Proyek | Implementasi dalam PyTorch |
| :--- | :--- | :--- |
| **Modul 2 - Preprocessing Data** | Resizing, normalisasi nilai piksel, data cleaning, dan augmentasi data. | `torchvision.transforms`: Resizing ke 128x128 / 224x224, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`, augmentasi `RandomRotation(degrees=10)`, `RandomAffine(scale=(0.9, 1.1))`, **tanpa RandomHorizontalFlip** (menjaga semantik tangan BISINDO). |
| **Modul 12 - Neural Networks Dasar** | Multi-Layer Perceptron (MLP), forward/backward propagation, fungsi aktivasi & loss, optimasi gradien. | Aktivasi `nn.ReLU()`, loss function `nn.CrossEntropyLoss()`, optimizer `torch.optim.Adam(lr=0.001)`. Eksekusi loop: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. |
| **Modul 13 - Deep Learning & CNN** | Ekstraksi fitur visual spasial via kernel konvolusi, pooling, dan reduksi dimensi. | Arsitektur `nn.Module`: Blok `nn.Conv2d` -> `nn.BatchNorm2d` -> `nn.ReLU` -> `nn.MaxPool2d`, disusul `nn.Dropout(0.5)` dan `nn.Linear` (Fully Connected / Dense). |
| **Modul 15 - Transfer Learning** | Transfer representasi bobot ImageNet ke domain alfabet BISINDO. | `torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)`. Pembekuan layer fitur (`param.requires_grad = False`), modifikasi layer `model.classifier[1] = nn.Linear(1280, 26)`. |
| **Modul 6 - Evaluasi Klasifikasi** | Pengukuran performa model multi-kelas secara kuantitatif dan visual. | Plot grafik Training vs Validation Loss & Accuracy, **Confusion Matrix 26x26** (Seaborn Heatmap), dan *Classification Report* (Precision, Recall, F1-Score) via Scikit-Learn. |

---

### 3. ARSITEKTUR DETEKSI PERANGKAT GPU OTOMATIS (DUAL-MODE)

Kode penentuan perangkat (*device selector*) yang cerdas dan adaptif:
```python
import torch

# Deteksi Perangkat: Prioritas 1 AMD DirectML (Lokal), Prioritas 2 NVIDIA CUDA (Colab)
def get_optimal_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            print("🚀 Terdeteksi GPU AMD Lokal via DirectML: AMD Radeon RX 5700 XT")
            return torch_directml.device()
    except ImportError:
        pass

    if torch.cuda.is_available():
        print(f"🚀 Terdeteksi GPU NVIDIA (Colab/CUDA): {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    
    print("⚠️ GPU tidak terdeteksi, fallback ke CPU")
    return torch.device("cpu")

device = get_optimal_device()
```

---

### 4. DETAIL DESAIN ARSITEKTUR KEDUA MODEL

#### A. Model 1: Custom CNN (From Scratch - Modul 13)
- **Input**: `(batch_size, 3, 128, 128)`
- **Feature Extractor**:
  - Blok 1: `Conv2d(3, 32, kernel_size=3, padding=1)` -> `BatchNorm2d(32)` -> `ReLU()` -> `MaxPool2d(2, 2)`
  - Blok 2: `Conv2d(32, 64, kernel_size=3, padding=1)` -> `BatchNorm2d(64)` -> `ReLU()` -> `MaxPool2d(2, 2)`
  - Blok 3: `Conv2d(64, 128, kernel_size=3, padding=1)` -> `BatchNorm2d(128)` -> `ReLU()` -> `MaxPool2d(2, 2)`
- **Classifier Head**:
  - `AdaptiveAvgPool2d((4, 4))`
  - `Flatten()` -> `Linear(128 * 4 * 4, 256)` -> `ReLU()` -> `Dropout(0.5)` -> `Linear(256, 26)`

#### B. Model 2: Transfer Learning MobileNetV2 (Modul 15)
- **Input**: `(batch_size, 3, 224, 224)`
- **Backbone**: `torchvision.models.mobilenet_v2(pretrained=True)`
- **Layer Freezing**: Seluruh parameter backbone dibekukan (`requires_grad = False`).
- **Classifier Head**:
  - `Dropout(0.2)`
  - `Linear(1280, 26)` (Hanya bobot layer ini yang dilatih).

---

### 5. STRATEGI PELATIHAN & OPTIMASI (TRAINING LOOP)
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `torch.optim.Adam` (lr=1e-3 untuk Custom CNN, lr=1e-4 untuk MobileNetV2).
- **Epochs & Batch Size**: 20 Epochs, Batch Size 32 / 64 (aman untuk 8 GB VRAM RX 5700 XT & 16 GB VRAM T4).
- **Early Stopping & Checkpoint**: Menyimpan model berbobot validasi terbaik (`best_custom_cnn.pth` dan `best_mobilenetv2.pth`).

---

### 6. TAHAPAN EKSEKUSI (ROADMAP IMPLEMENTASI)

1. **Persiapan Lingkungan (Setup)**
   - Setup file notebook `klasifikasi_bisindo_cnn_pytorch.ipynb`.
   - Instruksi instalasi dependency lokal (`pip install torch torchvision torch-directml matplotlib seaborn scikit-learn pillow`).
2. **Data Pipeline & Augmentasi (Modul 2)**
   - Menggunakan `torchvision.datasets.ImageFolder` untuk data `train/` dan `val/`.
   - Wrapping dengan `torch.utils.data.DataLoader(batch_size=32, shuffle=True, num_workers=2, pin_memory=True)`.
3. **Pembangunan & Pelatihan Custom CNN (Modul 12 & 13)**
   - Training loop PyTorch lengkap dengan logging Loss & Akurasi per epoch.
4. **Pembangunan & Pelatihan Transfer Learning (Modul 12 & 15)**
   - Fine-tuning MobileNetV2 pada dataset BISINDO.
5. **Evaluasi & Komparasi Kinerja (Modul 6)**
   - Plot kurva konvergensi (Loss & Accuracy).
   - Matriks Konfusi 26x26 kelas alfabet BISINDO.
   - Classification Report (Precision, Recall, F1-Score).
6. **Modul Pengujian Inferensi Citra Tunggal**
   - Fungsi `predict_single_image(image_path, model, device)` yang memvisualisasikan citra input beserta teks prediksi huruf dan probabilitas keyakinan model.