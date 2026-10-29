import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Deteksi Perangkat GPU
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
print(f"Perangkat aktif inferensi kamera: {device}")

class_names = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

# 2. Definisi Arsitektur Model
# A. 1D-CNN Landmark (Model Utama Presisi Tinggi)
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

# B. 2D-CNN MobileNetV2
def build_mobilenet_v2(num_classes=26):
    m = models.mobilenet_v2(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return m

# C. Custom 2D-CNN
class CustomSignCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(CustomSignCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.adaptive_pool(self.block3(self.block2(self.block1(x)))))

# 3. Load Seluruh Model
models_dict = {}

# Model 1: 1D-CNN Landmark
if os.path.exists("best_landmark_cnn.pth"):
    try:
        m_lnd = SignLandmarkCNN(in_channels=3, num_classes=26).to(device)
        m_lnd.load_state_dict(torch.load("best_landmark_cnn.pth", map_location=device))
        m_lnd.eval()
        models_dict["1D-CNN Landmark Skeleton (Presisi 97%+) [Utama]"] = {
            "type": "landmark",
            "model": m_lnd
        }
        print("✅ Berhasil memuat: best_landmark_cnn.pth (Model 1D-CNN Utama)")
    except Exception as e:
        print("Gagal memuat 1D-CNN:", e)

# Model 2: 2D-CNN MobileNetV2
if os.path.exists("best_mobilenetv2.pth"):
    try:
        m_m2 = build_mobilenet_v2(26).to(device)
        m_m2.load_state_dict(torch.load("best_mobilenetv2.pth", map_location=device))
        m_m2.eval()
        models_dict["2D-CNN MobileNetV2 (Transfer Learning)"] = {
            "type": "image",
            "model": m_m2,
            "size": (224, 224)
        }
        print("✅ Berhasil memuat: best_mobilenetv2.pth")
    except Exception as e:
        print("Gagal memuat MobileNetV2:", e)

# Model 3: Custom 2D-CNN
if os.path.exists("best_custom_cnn.pth"):
    try:
        m_c = CustomSignCNN(26).to(device)
        m_c.load_state_dict(torch.load("best_custom_cnn.pth", map_location=device))
        m_c.eval()
        models_dict["2D-CNN Custom Scratch (3-Block CNN)"] = {
            "type": "image",
            "model": m_c,
            "size": (128, 128)
        }
        print("✅ Berhasil memuat: best_custom_cnn.pth")
    except Exception as e:
        print("Gagal memuat Custom CNN:", e)

model_keys = list(models_dict.keys())
current_model_idx = 0

# 4. Inisialisasi MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.18,
    min_hand_presence_confidence=0.18
))

def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts -= pts[0].copy()
    scale = np.linalg.norm(pts[9])
    if scale > 1e-4:
        pts /= scale
    return pts

tf_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Pindai Kamera
def scan_cameras(max_check=6):
    available = []
    for i in range(max_check):
        temp_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not temp_cap.isOpened():
            temp_cap = cv2.VideoCapture(i)
        if temp_cap.isOpened():
            available.append(i)
            temp_cap.release()
    return available

available_cams = scan_cameras()
initial_cam = 2 if 2 in available_cams else (available_cams[0] if available_cams else 0)
if len(sys.argv) > 1 and sys.argv[1].isdigit():
    val = int(sys.argv[1])
    if val in available_cams or len(available_cams) == 0:
        initial_cam = val

cam_idx_pointer = available_cams.index(initial_cam) if initial_cam in available_cams else 0

def init_capture(cam_index):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap

current_cam_id = available_cams[cam_idx_pointer] if available_cams else 0
cap = init_capture(current_cam_id)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mirror_view = True

# Ukuran kotak tengah: disesuaikan persis dengan format dataset latih (Bujur Sangkar 640x640)
roi_size = 640
if roi_size > int(min(actual_w, actual_h) * 0.90):
    roi_size = int(min(actual_w, actual_h) * 0.85)

print("\n" + "="*65)
print("🎯 BISINDO CLASSIFIER - DUAL MODE (1D-CNN LANDMARK & 2D-CNN):")
print(" - Model Utama : 1D-CNN Landmark Skeleton (Presisi 97%+ Kebal Dinding/Kipas)")
print(" - Tekan 'M' : Ganti Model (1D-CNN <-> MobileNetV2 <-> Custom CNN)")
print(" - Tekan '+' : Perbesar kotak tengah")
print(" - Tekan '-' : Perkecil kotak tengah")
print(" - Tekan 'X' : Toggle Mirror View (Cermin ON/OFF)")
print(" - Tekan 'C' : Ganti Kamera (Index 0/1/2)")
print(" - Tekan 'Q' : Keluar")
print("="*65 + "\n")

window_name = "Pengenalan Alfabet BISINDO - Center Frame"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

while True:
    ret, frame = cap.read()
    if not ret:
        cv2.putText(frame, f"Gagal membaca Kamera {current_cam_id}. Tekan 'C' ganti kamera.",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c") or key == ord("C"):
            if available_cams:
                cam_idx_pointer = (cam_idx_pointer + 1) % len(available_cams)
                current_cam_id = available_cams[cam_idx_pointer]
                cap.release()
                cap = init_capture(current_cam_id)
        continue

    if mirror_view:
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Posisi Kotak Pas di Tengah (Bujur Sangkar 1:1 Sesuai Dataset)
    start_x = max(0, (w - roi_size) // 2)
    start_y = max(0, (h - roi_size) // 2)
    end_x = min(w, start_x + roi_size)
    end_y = min(h, start_y + roi_size)

    roi = frame[start_y:end_y, start_x:end_x]
    active_key = model_keys[current_model_idx]
    active_meta = models_dict[active_key]

    top3_results = []
    hand_found = False
    hand_count = 0

    # 1. Jalankan Deteksi MediaPipe pada Kotak ROI (Format 640x640 Identik Dataset)
    if roi.size > 0 and roi.shape[0] == roi_size and roi.shape[1] == roi_size:
        # Resize ke 640x640 agar densitas dan koordinat geometris 100% identik dengan dataset train
        roi_640 = cv2.resize(roi, (640, 640))
        rgb_roi = cv2.cvtColor(roi_640, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
        res = detector.detect(mp_img)

        # Gambar Skeleton pada ROI
        if res.hand_landmarks:
            hand_found = True
            hand_count = len(res.hand_landmarks)
            for lms in res.hand_landmarks:
                pts = [(start_x + int(lm.x * roi_size), start_y + int(lm.y * roi_size)) for lm in lms]
                for p in pts:
                    cv2.circle(frame, p, 5, (0, 255, 127), -1)
                for p1, p2 in HAND_CONNECTIONS:
                    cv2.line(frame, pts[p1], pts[p2], (0, 255, 255), 2)

        # 2. Inferensi Sesuai Model Aktif
        if active_meta["type"] == "landmark":
            # Model 1D-CNN Landmark
            if res.hand_landmarks:
                if hand_count == 1:
                    h1 = normalize_hand(res.hand_landmarks[0])
                    h2 = np.zeros((21, 3), dtype=np.float32)
                    inp_lnd = torch.from_numpy(np.vstack([h1, h2]).T).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = active_meta["model"](inp_lnd)
                        probs = torch.nn.functional.softmax(out[0], dim=0)
                else:
                    # Evaluasi kedua kombinasi permutasi [H1, H2] dan [H2, H1] secara instan dalam 1 batch tensor
                    hA = normalize_hand(res.hand_landmarks[0])
                    hB = normalize_hand(res.hand_landmarks[1])
                    p1 = np.vstack([hA, hB]).T
                    p2 = np.vstack([hB, hA]).T
                    batch_inp = torch.from_numpy(np.stack([p1, p2])).to(device)
                    with torch.no_grad():
                        out = active_meta["model"](batch_inp)
                        probs_b = torch.nn.functional.softmax(out, dim=1)
                        max_conf, _ = torch.max(probs_b, dim=1)
                        best_idx = torch.argmax(max_conf).item()
                        probs = probs_b[best_idx]

                top_probs, top_indices = torch.topk(probs, 3)
                for p, idx in zip(top_probs, top_indices):
                    top3_results.append((class_names[idx.item()], p.item() * 100))
        else:
            # Model 2D-CNN Citra (MobileNetV2 / Custom)
            target_size = active_meta["size"]
            pil_img = Image.fromarray(rgb_roi).resize(target_size)
            tensor_2d = tf_2d(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = active_meta["model"](tensor_2d)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                top_probs, top_indices = torch.topk(probs, 3)

            for p, idx in zip(top_probs, top_indices):
                top3_results.append((class_names[idx.item()], p.item() * 100))

    # Redupkan area luar kotak tengah
    dimmed = frame.copy()
    dimmed[:start_y, :] = (dimmed[:start_y, :] * 0.55).astype(np.uint8)
    dimmed[end_y:, :] = (dimmed[end_y:, :] * 0.55).astype(np.uint8)
    dimmed[start_y:end_y, :start_x] = (dimmed[start_y:end_y, :start_x] * 0.55).astype(np.uint8)
    dimmed[start_y:end_y, end_x:] = (dimmed[start_y:end_y, end_x:] * 0.55).astype(np.uint8)
    frame = dimmed

    best_letter = "-"
    best_conf = 0.0
    if top3_results:
        best_letter, best_conf = top3_results[0]

    # Border Kotak
    is_confident = (best_conf >= 50.0)
    box_color = (0, 255, 127) if is_confident else ((0, 215, 255) if best_conf >= 25.0 else (0, 140, 255))
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 3)

    cv2.putText(frame, f"KOTAK TENGAH ({roi_size}x{roi_size}) - FORMAT DATASET", (start_x, start_y - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, box_color, 2)

    # Panel HUD Prediksi (Kiri Atas)
    cv2.rectangle(frame, (30, 25), (650, 240), (20, 20, 20), -1)
    cv2.rectangle(frame, (30, 25), (650, 240), (70, 70, 70), 2)

    cv2.putText(frame, f"Kamera: Index {current_cam_id} | Model [M]: {active_key[:38]}", 
                (45, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1)

    if hand_count >= 2:
        hand_text = "Tangan: 2 Terdeteksi [Dual-Hand Aktif]"
        hand_col = (0, 255, 127)
    elif hand_count == 1:
        hand_text = "Tangan: 1 Terdeteksi"
        hand_col = (0, 215, 255)
    else:
        hand_text = "Tangan: Belum Terdeteksi"
        hand_col = (0, 165, 255)

    cv2.putText(frame, f"{hand_text} | Mirror [X]: {'ON' if mirror_view else 'OFF'}", (45, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, hand_col, 1)

    if top3_results and is_confident:
        cv2.putText(frame, f"Prediksi: {best_letter}", (45, 142),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 3)
        cv2.putText(frame, f"Keyakinan: {best_conf:.1f}%", (45, 178),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        alt_text = f"Top-3: 1.{best_letter} ({best_conf:.0f}%) | 2.{top3_results[1][0]} ({top3_results[1][1]:.0f}%) | 3.{top3_results[2][0]} ({top3_results[2][1]:.0f}%)"
        cv2.putText(frame, alt_text, (45, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    else:
        status_msg = "Posisikan Tangan di Kotak..." if not hand_found else f"Gestur: {best_letter} ({best_conf:.0f}%)"
        cv2.putText(frame, status_msg, (45, 150), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 165, 255), 2)
        cv2.putText(frame, "Hadapkan telapak tangan & lakukan gestur BISINDO", (45, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Banner Tombol Pintas
    banner_text = "[M] Ganti Model | [+/-] Ukuran Kotak | [X] Mirror | [C] Kamera | [Q] Keluar"
    cv2.putText(frame, banner_text, (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break
    elif key == ord("m") or key == ord("M"):
        current_model_idx = (current_model_idx + 1) % len(model_keys)
        print(f"🔄 Berpindah ke model: {model_keys[current_model_idx]}")
    elif key == ord("+") or key == ord("="):
        roi_size = min(int(min(w, h) * 0.95), roi_size + 30)
        print(f"➕ Kotak tengah diperbesar: {roi_size}x{roi_size}")
    elif key == ord("-") or key == ord("_"):
        roi_size = max(200, roi_size - 30)
        print(f"➖ Kotak tengah diperkecil: {roi_size}x{roi_size}")
    elif key == ord("x") or key == ord("X"):
        mirror_view = not mirror_view
        print(f"🪞 Cermin Kamera: {'ON' if mirror_view else 'OFF'}")
    elif key == ord("c") or key == ord("C"):
        if available_cams:
            cam_idx_pointer = (cam_idx_pointer + 1) % len(available_cams)
            current_cam_id = available_cams[cam_idx_pointer]
            print(f"🔄 Beralih ke Kamera Indeks: {current_cam_id}")
            cap.release()
            cap = init_capture(current_cam_id)
    elif ord("0") <= key <= ord("9"):
        req_id = key - ord("0")
        if req_id in available_cams:
            cam_idx_pointer = available_cams.index(req_id)
            current_cam_id = req_id
            print(f"🎯 Memilih langsung Kamera Indeks: {current_cam_id}")
            cap.release()
            cap = init_capture(current_cam_id)

cap.release()
cv2.destroyAllWindows()
