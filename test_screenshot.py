import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(filename):
    """Mencari file di direktori script terlebih dahulu, lalu di working directory."""
    path_in_base = os.path.join(BASE_DIR, filename)
    if os.path.exists(path_in_base):
        return path_in_base
    if os.path.exists(filename):
        return os.path.abspath(filename)
    return path_in_base

# 1. Dialog Pemilih File Windows Explorer
def choose_image_file():
    """Membuka dialog file explorer Windows untuk memilih satu gambar."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Pilih Citra Gestur BISINDO untuk Diuji",
            filetypes=[
                ("Citra Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Semua File", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"⚠️ Dialog Explorer gagal dibuka ({e}). Silakan input via terminal.")
        return None

# 2. Definisi Arsitektur Model
# A. Model 1D-CNN Landmark (Model Utama - Kebal Background Noise)
class SignLandmarkCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=26):
        super(SignLandmarkCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# B. Model 2D-CNN MobileNetV2 (Model Pembanding Citra)
def build_mobilenet_v2(num_classes=26):
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return m

# 3. Deteksi Perangkat & Load Model
def get_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Perangkat aktif evaluasi citra: {device}")

def safe_load_model(model, file_path, target_device):
    try:
        state = torch.load(file_path, map_location=target_device, weights_only=False)
        model.load_state_dict(state)
        model.to(target_device)
        model.eval()
        return model, target_device
    except Exception as err_target:
        try:
            cpu_dev = torch.device("cpu")
            state = torch.load(file_path, map_location=cpu_dev, weights_only=False)
            model.load_state_dict(state)
            model.to(cpu_dev)
            model.eval()
            print(f"   ℹ️ Dimuat ke CPU sebagai fallback: {os.path.basename(file_path)}")
            return model, cpu_dev
        except Exception as err_cpu:
            raise RuntimeError(f"Gagal memuat {file_path}: {err_cpu}")

# Pemuatan Model 1D-CNN
lnd_model = None
lnd_path = resolve_path("best_landmark_cnn.pth")
if os.path.exists(lnd_path):
    try:
        m_lnd = SignLandmarkCNN(in_channels=3, num_classes=26)
        lnd_model, _ = safe_load_model(m_lnd, lnd_path, device)
        print(f"✅ Berhasil memuat: {os.path.basename(lnd_path)} (1D-CNN Landmark)")
    except Exception as e:
        print(f"❌ Gagal memuat 1D-CNN: {e}")

# Pemuatan Model 2D-CNN MobileNetV2
m2_model = None
m2_path = resolve_path("best_mobilenetv2.pth")
if os.path.exists(m2_path):
    try:
        m_m2 = build_mobilenet_v2(26)
        m2_model, _ = safe_load_model(m_m2, m2_path, device)
        print(f"✅ Berhasil memuat: {os.path.basename(m2_path)} (2D-CNN MobileNetV2)")
    except Exception as e:
        print(f"❌ Gagal memuat 2D-CNN: {e}")

# Inisialisasi MediaPipe Hand Landmarker
detector = None
task_path = resolve_path("hand_landmarker.task")
if os.path.exists(task_path):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path=task_path)
        detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.18,
            min_hand_presence_confidence=0.18
        ))
        print(f"✅ MediaPipe Hand Landmarker siap: {os.path.basename(task_path)}")
    except Exception as e:
        print(f"⚠️ MediaPipe gagal diinisialisasi: {e}")
else:
    print(f"⚠️ File '{task_path}' tidak ditemukan.")

# Pipeline Transformasi 2D-CNN
tf_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = [chr(65 + i) for i in range(26)]

def normalize_hand(landmarks):
    """Normalisasi sendi MediaPipe: translasi ke wrist (0) dan skala terhadap MCP jari tengah (9)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0].copy()
    scale = np.linalg.norm(pts[9])
    if scale > 1e-4:
        pts /= scale
    return pts

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# -------------------------------------------------------------------------
# Helper Rendering UI Modern (HUD, Badges, Corner Reticles)
# -------------------------------------------------------------------------
def draw_corner_brackets(img, x1, y1, x2, y2, color=(240, 180, 0), length=28, thickness=2, label="SMART ROI"):
    """Menggambar reticle sudut kamera pintar (corner brackets) pada area ROI."""
    h_img, w_img = img.shape[:2]
    # Inset 4px agar reticle tidak terpotong tepi citra
    x1, y1 = max(4, x1), max(4, y1)
    x2, y2 = min(w_img - 5, x2), min(h_img - 5, y2)
    length = min(length, max(6, (x2 - x1) // 4), max(6, (y2 - y1) // 4))

    # 4 Corner brackets dengan anti-aliasing cv2.LINE_AA
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)

    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    if label and (x2 - x1) > 90 and (y2 - y1) > 50:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        if y1 < 26:
            # Letakkan badge di dalam kotak jika terlalu dekat dengan tepi atas
            bx1 = x1 + 4
            by1 = y1 + 4
            bx2 = min(w_img - 4, bx1 + tw + 14)
            by2 = by1 + th + 10
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 16, 14), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
            cv2.putText(img, label, (bx1 + 7, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            # Letakkan badge tepat di atas garis sudut atas
            bx1 = x1
            by1 = y1 - th - 10
            bx2 = min(w_img - 4, x1 + tw + 14)
            by2 = y1
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (20, 16, 14), -1)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 1, cv2.LINE_AA)
            cv2.putText(img, label, (bx1 + 7, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

def build_modern_hud_panel(filename, crop_roi_info, hand_count, top3_lnd, top3_2d, target_h=720, panel_w=500):
    """Menghasilkan panel instrumen HUD modern bergaya dark theme dengan cards dan progress bars."""
    panel = np.full((target_h, panel_w, 3), (28, 21, 18), dtype=np.uint8)  # Slate dark #12151c

    # Garis aksen atas dual-color (Emerald - Cyan)
    cv2.line(panel, (0, 0), (panel_w // 2, 0), (129, 185, 16), 3)
    cv2.line(panel, (panel_w // 2, 0), (panel_w, 0), (212, 182, 6), 3)

    # Header Panel
    cv2.putText(panel, "EVALUASI CITRA BISINDO", (22, 34), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "Dual-Model AI Recognition Pipeline", (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 175, 185), 1, cv2.LINE_AA)

    # Baris Informasi Metadata (Pill Badges)
    fname_display = (filename[:26] + "...") if len(filename) > 28 else filename
    cv2.rectangle(panel, (22, 66), (250, 90), (43, 31, 26), -1)
    cv2.rectangle(panel, (22, 66), (250, 90), (65, 50, 42), 1, cv2.LINE_AA)
    cv2.putText(panel, f"File: {fname_display}", (30, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

    # Status Deteksi Tangan
    if hand_count >= 2:
        status_bg = (24, 52, 22)
        status_border = (50, 160, 40)
        status_text = f"[●] {hand_count} Tangan Terdeteksi"
        status_color = (110, 255, 130)
    elif hand_count == 1:
        status_bg = (30, 48, 48)
        status_border = (80, 160, 160)
        status_text = "[●] 1 Tangan Terdeteksi"
        status_color = (100, 240, 240)
    else:
        status_bg = (20, 25, 55)
        status_border = (40, 60, 180)
        status_text = "[!] Tangan Tidak Terdeteksi"
        status_color = (90, 140, 255)

    cv2.rectangle(panel, (260, 66), (478, 90), status_bg, -1)
    cv2.rectangle(panel, (260, 66), (478, 90), status_border, 1, cv2.LINE_AA)
    cv2.putText(panel, status_text, (270, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.36, status_color, 1, cv2.LINE_AA)

    # Info Area ROI
    cv2.putText(panel, f"Area Analisis: {crop_roi_info}", (24, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150, 150, 160), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # CARD 1: MODEL 1 - 1D-CNN LANDMARK SKELETON (Emerald Theme)
    # -------------------------------------------------------------
    c1_y1, c1_y2 = 120, 372
    cv2.rectangle(panel, (18, c1_y1), (482, c1_y2), (40, 29, 24), -1)
    cv2.rectangle(panel, (18, c1_y1), (482, c1_y2), (65, 50, 42), 1, cv2.LINE_AA)
    # Stripe aksen kiri (Emerald)
    cv2.rectangle(panel, (18, c1_y1), (22, c1_y2), (129, 185, 16), -1)

    cv2.putText(panel, "[MODEL 1] 1D-CNN LANDMARK SKELETON", (32, 145), cv2.FONT_HERSHEY_DUPLEX, 0.50, (120, 245, 130), 1, cv2.LINE_AA)
    cv2.putText(panel, "Model Utama · Kebal Background Noise (Presisi 97%+)", (32, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 210, 160), 1, cv2.LINE_AA)

    if top3_lnd:
        best_char, best_conf = top3_lnd[0]
        # Kotak badge huruf besar
        cv2.rectangle(panel, (32, 178), (96, 242), (26, 18, 15), -1)
        cv2.rectangle(panel, (32, 178), (96, 242), (129, 185, 16), 1, cv2.LINE_AA)

        (cw, ch), _ = cv2.getTextSize(best_char, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)
        lx = 32 + (64 - cw) // 2
        ly = 178 + (64 + ch) // 2 - 2
        cv2.putText(panel, best_char, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(panel, "PREDIKSI UTAMA", (110, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 165, 175), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Huruf {best_char}", (110, 220), cv2.FONT_HERSHEY_DUPLEX, 0.68, (120, 255, 140), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Keyakinan: {best_conf:.1f}%", (110, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.putText(panel, "DISTRIBUSI TOP-3 PROBABILITAS:", (32, 264), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 145, 155), 1, cv2.LINE_AA)

        bar_y = 282
        for rank, (char, pct) in enumerate(top3_lnd):
            cv2.putText(panel, f"#{rank+1}", (32, bar_y + 11), cv2.FONT_HERSHEY_DUPLEX, 0.38, (170, 175, 185), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Huruf {char}", (62, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

            track_x, track_w, track_h = 145, 240, 13
            cv2.rectangle(panel, (track_x, bar_y), (track_x + track_w, bar_y + track_h), (30, 22, 18), -1)
            cv2.rectangle(panel, (track_x, bar_y), (track_x + track_w, bar_y + track_h), (55, 42, 34), 1, cv2.LINE_AA)

            fill_w = int((pct / 100.0) * track_w)
            if fill_w > 0:
                bar_col = (110, 235, 100) if rank == 0 and pct >= 50 else ((70, 190, 80) if rank == 0 else (95, 80, 72))
                cv2.rectangle(panel, (track_x + 1, bar_y + 1), (track_x + fill_w, bar_y + track_h - 1), bar_col, -1)

            cv2.putText(panel, f"{pct:.1f}%", (track_x + track_w + 12, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (225, 225, 225), 1, cv2.LINE_AA)
            bar_y += 26
    else:
        cv2.rectangle(panel, (32, 178), (96, 242), (26, 18, 15), -1)
        cv2.rectangle(panel, (32, 178), (96, 242), (40, 60, 180), 1, cv2.LINE_AA)
        cv2.putText(panel, "?", (50, 224), cv2.FONT_HERSHEY_DUPLEX, 1.4, (90, 140, 255), 2, cv2.LINE_AA)

        cv2.putText(panel, "SENDI TIDAK TERDETEKSI", (110, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "MediaPipe Tidak Menemukan Tangan", (110, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, "Pastikan kedua tangan terlihat jelas di dalam frame.", (32, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 160), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # CARD 2: MODEL 2 - 2D-CNN MOBILENETV2 (Electric Cyan Theme)
    # -------------------------------------------------------------
    c2_y1, c2_y2 = 386, 638
    cv2.rectangle(panel, (18, c2_y1), (482, c2_y2), (40, 29, 24), -1)
    cv2.rectangle(panel, (18, c2_y1), (482, c2_y2), (65, 50, 42), 1, cv2.LINE_AA)
    # Stripe aksen kiri (Electric Cyan)
    cv2.rectangle(panel, (18, c2_y1), (22, c2_y2), (212, 182, 6), -1)

    cv2.putText(panel, "[MODEL 2] 2D-CNN MOBILENETV2 (CITRA)", (32, 411), cv2.FONT_HERSHEY_DUPLEX, 0.50, (230, 210, 60), 1, cv2.LINE_AA)
    cv2.putText(panel, "Model Pembanding · Evaluasi Tekstur Gestur Tangan", (32, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 200, 150), 1, cv2.LINE_AA)

    if top3_2d:
        best_2d_char, best_2d_conf = top3_2d[0]
        # Kotak badge huruf besar
        cv2.rectangle(panel, (32, 444), (96, 508), (26, 18, 15), -1)
        cv2.rectangle(panel, (32, 444), (96, 508), (212, 182, 6), 1, cv2.LINE_AA)

        (cw, ch), _ = cv2.getTextSize(best_2d_char, cv2.FONT_HERSHEY_DUPLEX, 1.4, 2)
        lx = 32 + (64 - cw) // 2
        ly = 444 + (64 + ch) // 2 - 2
        cv2.putText(panel, best_2d_char, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(panel, "PREDIKSI CITRA", (110, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 165, 175), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Huruf {best_2d_char}", (110, 486), cv2.FONT_HERSHEY_DUPLEX, 0.68, (240, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Keyakinan: {best_2d_conf:.1f}%", (110, 504), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.putText(panel, "DISTRIBUSI TOP-3 PROBABILITAS:", (32, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 145, 155), 1, cv2.LINE_AA)

        bar_y = 548
        for rank, (char, pct) in enumerate(top3_2d):
            cv2.putText(panel, f"#{rank+1}", (32, bar_y + 11), cv2.FONT_HERSHEY_DUPLEX, 0.38, (170, 175, 185), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Huruf {char}", (62, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

            track_x, track_w, track_h = 145, 240, 13
            cv2.rectangle(panel, (track_x, bar_y), (track_x + track_w, bar_y + track_h), (30, 22, 18), -1)
            cv2.rectangle(panel, (track_x, bar_y), (track_x + track_w, bar_y + track_h), (55, 42, 34), 1, cv2.LINE_AA)

            fill_w = int((pct / 100.0) * track_w)
            if fill_w > 0:
                bar_col = (235, 200, 30) if rank == 0 and pct >= 50 else ((190, 160, 25) if rank == 0 else (95, 80, 72))
                cv2.rectangle(panel, (track_x + 1, bar_y + 1), (track_x + fill_w, bar_y + track_h - 1), bar_col, -1)

            cv2.putText(panel, f"{pct:.1f}%", (track_x + track_w + 12, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (225, 225, 225), 1, cv2.LINE_AA)
            bar_y += 26
    else:
        cv2.rectangle(panel, (32, 444), (96, 508), (26, 18, 15), -1)
        cv2.rectangle(panel, (32, 444), (96, 508), (55, 42, 34), 1, cv2.LINE_AA)
        cv2.putText(panel, "-", (52, 490), cv2.FONT_HERSHEY_DUPLEX, 1.4, (140, 140, 140), 2, cv2.LINE_AA)
        cv2.putText(panel, "Model 2D-CNN Tidak Aktif", (110, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 180), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # FOOTER SECTION (Panduan Navigasi Keyboard)
    # -------------------------------------------------------------
    cv2.rectangle(panel, (0, target_h - 66), (panel_w, target_h), (20, 15, 12), -1)
    cv2.line(panel, (0, target_h - 66), (panel_w, target_h - 66), (55, 42, 35), 1, cv2.LINE_AA)

    # Tombol [ O ]
    cv2.rectangle(panel, (20, target_h - 52), (54, target_h - 22), (32, 24, 18), -1)
    cv2.rectangle(panel, (20, target_h - 52), (54, target_h - 22), (129, 185, 16), 1, cv2.LINE_AA)
    cv2.putText(panel, "O", (30, target_h - 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (120, 255, 140), 1, cv2.LINE_AA)
    cv2.putText(panel, "Buka File Baru (Explorer)", (64, target_h - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    # Tombol [ Q ]
    cv2.rectangle(panel, (300, target_h - 52), (334, target_h - 22), (32, 24, 18), -1)
    cv2.rectangle(panel, (300, target_h - 52), (334, target_h - 22), (75, 60, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "Q", (310, target_h - 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, (120, 140, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "Keluar", (344, target_h - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1, cv2.LINE_AA)

    return panel

# 4. Fungsi Utama Evaluasi Citra Tunggal
def evaluate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membaca gambar: {image_path}")
        return None

    h, w, _ = img.shape
    display_img = img.copy()

    # 1. Jalankan deteksi MediaPipe pada citra penuh (agar tidak terpotong)
    res = None
    hand_count = 0
    hand_found = False
    if detector is not None:
        rgb_display = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_display))

    # Jika tangan terdeteksi, gambar visualisasi skeleton sendi dengan efek glowing anti-aliased
    if res and res.hand_landmarks:
        hand_found = True
        hand_count = len(res.hand_landmarks)
        for lms in res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            # Garis tulang sendi (anti-aliased)
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(display_img, pts[p1], pts[p2], (0, 245, 180), 2, cv2.LINE_AA)
            # Titik landmark dengan dual-glow
            for idx, p in enumerate(pts):
                if idx == 0:  # Titik pergelangan tangan (wrist) aksen emas
                    cv2.circle(display_img, p, 6, (0, 220, 255), -1, cv2.LINE_AA)
                    cv2.circle(display_img, p, 2, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(display_img, p, 5, (0, 200, 100), -1, cv2.LINE_AA)
                    cv2.circle(display_img, p, 2, (255, 255, 255), -1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # A. PREDIKSI MODEL 1: 1D-CNN LANDMARK SKELETON (MODEL UTAMA)
    # -------------------------------------------------------------
    top3_lnd = []
    if lnd_model is not None and hand_found and res and res.hand_landmarks:
        if hand_count == 1:
            h1 = normalize_hand(res.hand_landmarks[0])
            h2 = np.zeros((21, 3), dtype=np.float32)
            inp = torch.from_numpy(np.vstack([h1, h2]).T).unsqueeze(0).to(device)
            with torch.no_grad():
                out = lnd_model(inp)
                probs_lnd = torch.softmax(out[0], dim=0)
        else:
            # Dual-Hand: Coba permutasi [H1, H2] dan [H2, H1]
            hA = normalize_hand(res.hand_landmarks[0])
            hB = normalize_hand(res.hand_landmarks[1])
            p1 = np.vstack([hA, hB]).T
            p2 = np.vstack([hB, hA]).T
            batch_inp = torch.from_numpy(np.stack([p1, p2])).to(device)
            with torch.no_grad():
                out = lnd_model(batch_inp)
                probs_b = torch.softmax(out, dim=1)
                max_conf, _ = torch.max(probs_b, dim=1)
                best_idx = torch.argmax(max_conf).item()
                probs_lnd = probs_b[best_idx]

        top_p, top_idx = torch.topk(probs_lnd, 3)
        for p, idx in zip(top_p, top_idx):
            top3_lnd.append((classes[idx.item()], p.item() * 100))

    # -------------------------------------------------------------
    # B. PREDIKSI MODEL 2: 2D-CNN MOBILENETV2 DENGAN SMART CROP
    # -------------------------------------------------------------
    top3_2d = []
    crop_roi_info = "Seluruh Citra (Full Image)"
    if m2_model is not None:
        target_crop = img.copy()
        if hand_found and res and res.hand_landmarks:
            all_x = [int(lm.x * w) for hand in res.hand_landmarks for lm in hand]
            all_y = [int(lm.y * h) for hand in res.hand_landmarks for lm in hand]
            min_x, max_x = max(0, min(all_x)), min(w, max(all_x))
            min_y, max_y = max(0, min(all_y)), min(h, max(all_y))
            # Tambahkan margin padding 40% agar proporsi tangan pas seperti dataset
            box_w = max_x - min_x
            box_h = max_y - min_y
            size = max(box_w, box_h, 50)
            pad = int(size * 0.40)
            cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
            r_size = size + pad
            x1, x2 = max(0, cx - r_size // 2), min(w, cx + r_size // 2)
            y1, y2 = max(0, cy - r_size // 2), min(h, cy + r_size // 2)
            if y2 > y1 and x2 > x1:
                target_crop = img[y1:y2, x1:x2]
                crop_roi_info = f"Smart Hand-Crop ({r_size}x{r_size})"
                # Gambar reticle targeting futuristik di sekeliling area tangan
                draw_corner_brackets(display_img, x1, y1, x2, y2, color=(240, 180, 0), length=28, thickness=2, label=f"SMART ROI ({r_size}x{r_size})")

        if target_crop is not None and target_crop.size > 0:
            rgb_2d = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
            tensor_2d = tf_2d(Image.fromarray(rgb_2d)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_2d = torch.softmax(m2_model(tensor_2d)[0], dim=0)
                top_p, top_idx = torch.topk(probs_2d, 3)
                for p, idx in zip(top_p, top_idx):
                    top3_2d.append((classes[idx.item()], p.item() * 100))

    # -------------------------------------------------------------
    # C. BUAT PANEL INFORMASI DAN HASIL (HUD) DI SISI KANAN
    # -------------------------------------------------------------
    # Skala citra secara proporsional agar tingginya pas 720p tanpa margin hitam kosong
    TARGET_H = 720
    PANEL_W = 500
    scale = TARGET_H / float(h)
    new_w = max(1, int(round(w * scale)))
    scaled_img = cv2.resize(display_img, (new_w, TARGET_H), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    panel = build_modern_hud_panel(os.path.basename(image_path), crop_roi_info, hand_count, top3_lnd, top3_2d, TARGET_H, PANEL_W)
    combined = np.hstack([scaled_img, panel])
    
    # Cetak hasil ke terminal
    print("\n" + "=" * 65)
    print(f"📊 HASIL KLASIFIKASI CITRA: {os.path.basename(image_path)}")
    print("=" * 65)
    if top3_lnd:
        print(f"🟢 [1D-CNN Landmark] Prediksi: Huruf {top3_lnd[0][0]} ({top3_lnd[0][1]:.2f}%)")
        for i, (c, p) in enumerate(top3_lnd):
            print(f"    Rank {i+1}: Huruf {c} ({p:.2f}%)")
    else:
        print("🟡 [1D-CNN Landmark] MediaPipe tidak mendeteksi sendi tangan.")

    if top3_2d:
        print(f"🔵 [2D-CNN MobileNet] Prediksi: Huruf {top3_2d[0][0]} ({top3_2d[0][1]:.2f}%)")
        for i, (c, p) in enumerate(top3_2d):
            print(f"    Rank {i+1}: Huruf {c} ({p:.2f}%)")
    print("=" * 65)

    return combined

def main():
    print("\n" + "=" * 65)
    print("🎯 PENGUJIAN CITRA TUNGGAL ALFABET BISINDO (GUI & EXPLORER)")
    print("=" * 65)

    current_file = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        current_file = sys.argv[1]
    else:
        print("Membuka File Explorer untuk memilih gambar...")
        current_file = choose_image_file()

    if not current_file or not os.path.exists(current_file):
        print("❌ Tidak ada file gambar yang dipilih. Program selesai.")
        return

    win_title = "Evaluasi Citra BISINDO (Dual Model: 1D-CNN + 2D-CNN)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while True:
        combined_view = evaluate_image(current_file)
        if combined_view is None:
            break

        # Skala window seragam agar rasio aspek 1:1 original terjaga (tidak gepeng)
        vh, vw = combined_view.shape[:2]
        max_screen_w, max_screen_h = 1366, 768
        scale = min(max_screen_w / vw, max_screen_h / vh, 1.0)
        win_w = max(int(vw * scale), 400)
        win_h = max(int(vh * scale), 300)
        cv2.resizeWindow(win_title, win_w, win_h)
        cv2.imshow(win_title, combined_view)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break
        elif key == ord("o") or key == ord("O"):
            print("\nMembuka File Explorer untuk memilih gambar baru...")
            new_file = choose_image_file()
            if new_file and os.path.exists(new_file):
                current_file = new_file

    cv2.destroyAllWindows()
    print("\nProgram evaluasi citra selesai. Sampai jumpa!")

if __name__ == "__main__":
    main()
