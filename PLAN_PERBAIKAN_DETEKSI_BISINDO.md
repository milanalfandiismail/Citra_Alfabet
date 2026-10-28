# Rencana Tindakan: Perbaikan Deteksi Huruf BISINDO (Segmentasi Background Netral & Penyesuaian Pose)

## 1. Analisis Akar Masalah (Root Cause Analysis)

Berdasarkan investigasi komparatif antara dataset asli (`bisindo/images/train/`) dan tangkapan webcam kamu (`media_1793193951091.png`):

| Parameter | Dataset Latih Asli (`train_sample_L.jpg`) | Tangkapan Webcam Nyata | Dampak pada CNN |
| :--- | :--- | :--- | :--- |
| **Latar Belakang** | Dinding polos abu-abu netral studio (RGB `~[175, 175, 175]`) | Kipas angin ungu pekat, dinding merah bertekstur | **Domain Gap Parah**: 70% pixel di dalam kotak adalah kipas angin & dinding merah. CNN melihat fitur kipas sebagai objek utama. |
| **Orientasi Tangan** | **Telapak tangan (Palm)** menghadap ke kamera | **Punggung tangan** menghadap ke kamera (buku jari terlihat) | CNN dilatih mengenali kontur telapak & garis tangan, bukan punggung tangan. |
| **Skala Tangan** | Tangan & pergelangan mengisi **75% - 85%** area gambar | Tangan hanya mengisi **25% - 30%** area kotak | Tangan terlalu kecil sehingga detail jari hilang saat di-resize ke 224x224. |
| **Pencahayaan** | Cahaya putih studio merata, bayangan lembut | Pencahayaan remang kemerahan dari samping | Histogram warna RGB bergeser jauh dari distribusi data latih. |

---

## 2. Keputusan Desain yang Disepakati (/grill-me)

1. **Strategi Utama**: Solusi Preprocessing & Pose (mengoptimalkan data input ke model yang sudah terlatih tanpa download data baru terlebih dahulu).
2. **Teknik Pra-pengolahan**: **Segmentasi Tangan + Background Netralizer**
   - MediaPipe Hand Landmark mengisolasi area tangan secara akurat.
   - Area di luar tangan (kipas angin, dinding merah) diganti dengan warna kanvas abu-abu studio (RGB `180, 180, 180`).
   - Melakukan *tight bounding crop* dengan padding 25% agar tangan memenuhi 75-80% frame (sesuai proporsi dataset).
3. **Thresholding & UI**:
   - Jika tangan tidak terdeteksi atau confidence < 40%, sistem menampilkan status `[Menunggu Gestur Tangan...]` (mencegah prediksi acak).
   - Menampilkan inset mini **"Input CNN (Preview)"** di sudut layar agar pengguna bisa melihat apakah tangan terpotong rapi dengan latar netral.
4. **Verifikasi & Evaluasi**:
   - Menjalankan pengujian langsung dengan pose telapak menghadap kamera untuk huruf C, L, I, V (1 tangan) dan A, B (2 tangan).
   - Jika akurasi mencapai target (>75-85%), model dinyatakan siap dipakai untuk laporan/skripsi.

---

## 3. Langkah Implementasi Teknis

### Fase 1: Pembaruan `test_camera.py`
- Mengintegrasikan modul segmentasi tangan berbasis MediaPipe Task (`hand_landmarker.task` yang sudah tersedia lokal).
- Membuat fungsi `segment_and_neutralize_background(frame, hand_landmarks)` yang menghasilkan citra tangan berlatar belakang abu-abu terang.
- Menambahkan filter ambang batas keyakinan (confidence thresholding).
- Menampilkan visual HUD informatif:
  - Kotak deteksi dinamis mengikuti tangan.
  - Inset citra terproses (menunjukkan apa yang dilihat CNN).
  - Indikator peringatan jika pose/skala belum optimal.

### Fase 2: Pengujian Langsung & Panduan Pose
- Menjalankan script webcam yang telah diperbarui.
- Melakukan verifikasi untuk huruf yang sebelumnya gagal (**L** dan **C**):
  - **Huruf L**: Buka telapak tangan menghadap kamera, jempol ke samping kiri, telunjuk tegak ke atas.
  - **Huruf C**: Buka telapak tangan agak menyamping, lengkungkan telunjuk & jempol membentuk huruf C.
- Mengevaluasi apakah probabilitas melonjak dari ~10-20% ke >75-90%.
