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

    # Jika tangan terdeteksi, gambar visualisasi skeleton sendi
    if res and res.hand_landmarks:
        hand_found = True
        hand_count = len(res.hand_landmarks)
        for lms in res.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for p in pts:
                cv2.circle(display_img, p, 4, (0, 255, 127), -1)
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(display_img, pts[p1], pts[p2], (0, 255, 255), 2)

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
    # SOLUSI BACKGROUND NOISE TINGGI:
    # Potong kotak bujur sangkar terpusat hanya pada tangan untuk hilangkan latar belakang
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
                # Gambar kotak penanda crop pada display image
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 200, 0), 2)

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
    panel_w = 460
    target_h = max(h, 560)
    panel = np.full((target_h, panel_w, 3), 24, dtype=np.uint8)

    # Header Panel
    cv2.putText(panel, "EVALUASI CITRA BISINDO", (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 215, 255), 2)
    cv2.putText(panel, f"File : {os.path.basename(image_path)[:30]}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1)
    cv2.putText(panel, f"Area : {crop_roi_info}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1)
    
    status_tangan = f"{hand_count} Tangan Terdeteksi" if hand_count > 0 else "Tangan Tidak Terdeteksi"
    color_tangan = (0, 255, 127) if hand_count > 0 else (0, 140, 255)
    cv2.putText(panel, f"Status: {status_tangan}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color_tangan, 1)

    cv2.line(panel, (20, 126), (panel_w - 20, 126), (60, 60, 60), 1)

    # 1. HASIL MODEL 1: 1D-CNN LANDMARK SKELETON
    cv2.putText(panel, "[MODEL 1] 1D-CNN LANDMARK SKELETON", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 127), 1)
    cv2.putText(panel, "(Kebal Noise Background, Presisi 97%+)", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 220, 140), 1)

    y_pos = 198
    if top3_lnd:
        best_char, best_conf = top3_lnd[0]
        cv2.putText(panel, f"Prediksi Utama: {best_char} ({best_conf:.1f}%)", (25, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.70, (0, 255, 0), 2)
        y_pos += 26
        for rank, (char, pct) in enumerate(top3_lnd):
            cv2.putText(panel, f"#{rank+1} Huruf {char}: {pct:.1f}%", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            bar_len = int((pct / 100.0) * 220)
            bar_col = (0, 255, 127) if rank == 0 and pct >= 50 else ((0, 215, 255) if rank == 0 else (100, 100, 100))
            cv2.rectangle(panel, (170, y_pos - 12), (170 + 220, y_pos - 2), (40, 40, 40), -1)
            if bar_len > 0:
                cv2.rectangle(panel, (170, y_pos - 12), (170 + bar_len, y_pos - 2), bar_col, -1)
            y_pos += 26
    else:
        cv2.putText(panel, "Sendi tangan tidak terdeteksi oleh MediaPipe", (25, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 140, 255), 1)
        y_pos += 45

    y_pos += 12
    cv2.line(panel, (20, y_pos), (panel_w - 20, y_pos), (60, 60, 60), 1)
    y_pos += 24

    # 2. HASIL MODEL 2: 2D-CNN MOBILENETV2
    cv2.putText(panel, "[MODEL 2] 2D-CNN MOBILENETV2 (Citra)", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 215, 255), 1)
    cv2.putText(panel, "(Menggunakan Smart Hand-Crop Tanpa Noise)", (20, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 230), 1)
    y_pos += 44

    if top3_2d:
        best_2d_char, best_2d_conf = top3_2d[0]
        cv2.putText(panel, f"Prediksi Utama: {best_2d_char} ({best_2d_conf:.1f}%)", (25, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 215, 255), 2)
        y_pos += 26
        for rank, (char, pct) in enumerate(top3_2d):
            cv2.putText(panel, f"#{rank+1} Huruf {char}: {pct:.1f}%", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            bar_len = int((pct / 100.0) * 220)
            bar_col = (0, 255, 127) if rank == 0 and pct >= 50 else ((0, 215, 255) if rank == 0 else (100, 100, 100))
            cv2.rectangle(panel, (170, y_pos - 12), (170 + 220, y_pos - 2), (40, 40, 40), -1)
            if bar_len > 0:
                cv2.rectangle(panel, (170, y_pos - 12), (170 + bar_len, y_pos - 2), bar_col, -1)
            y_pos += 26
    else:
        cv2.putText(panel, "Model 2D-CNN tidak aktif", (25, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 140, 255), 1)

    # Footer Navigasi
    cv2.rectangle(panel, (0, target_h - 60), (panel_w, target_h), (16, 16, 16), -1)
    cv2.putText(panel, "[O] Buka File Baru (File Explorer)", (20, target_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 127), 1)
    cv2.putText(panel, "[Q] Keluar Aplikasi", (20, target_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    # Jaga rasio aspek citra asli tanpa melar / gepeng
    if h != target_h:
        canvas_img = np.full((target_h, w, 3), 20, dtype=np.uint8)
        y_offset = (target_h - h) // 2
        canvas_img[y_offset:y_offset + h, 0:w] = display_img
        combined = np.hstack([canvas_img, panel])
    else:
        combined = np.hstack([display_img, panel])
    
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
