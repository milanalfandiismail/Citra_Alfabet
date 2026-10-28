import os
import sys
import glob
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# 1. Load Model 2D-CNN MobileNetV2
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 26)

ckpt_path = "best_mobilenetv2.pth"
if not os.path.exists(ckpt_path):
    print(f"Error: {ckpt_path} tidak ditemukan!")
    sys.exit(1)

model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

classes = [chr(65 + i) for i in range(26)]

# Inisialisasi MediaPipe jika ada
has_mp = False
detector = None
if os.path.exists("hand_landmarker.task"):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.15
        )
        detector = vision.HandLandmarker.create_from_options(options)
        has_mp = True
    except Exception:
        pass

def evaluate_and_visualize(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    h, w, _ = img.shape
    display_img = img.copy()

    # Cari kotak kuning ROI jika ada
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (20, 140, 140), (40, 255, 255))
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cnt = None
    max_a = 0
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw * bh > max_a and bw > 150:
            max_a = bw * bh
            best_cnt = (bx, by, bw, bh)

    crop_to_eval = img
    roi_info = "Full Image"
    if best_cnt:
        bx, by, bw, bh = best_cnt
        crop_to_eval = img[max(0, by+4):min(h, by+bh-4), max(0, bx+4):min(w, bx+bw-4)]
        roi_info = f"Kotak ROI ({bw}x{bh})"

    # Prediksi 2D-CNN
    rgb_crop = cv2.cvtColor(crop_to_eval, cv2.COLOR_BGR2RGB)
    tensor = tf(Image.fromarray(rgb_crop)).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].numpy()
    top5_idx = np.argsort(probs)[::-1][:5]

    # Deteksi skeleton tangan jika ada
    if has_mp and detector is not None:
        mp_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_rgb))
        if res.hand_landmarks:
            for lms in res.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                for p in pts:
                    cv2.circle(display_img, p, 4, (0, 255, 0), -1)
                # Garis jari
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                for p1, p2 in connections:
                    cv2.line(display_img, pts[p1], pts[p2], (0, 255, 255), 2)

    # Buat Panel Hasil di Samping Kanan
    panel_w = 420
    panel = np.full((h, panel_w, 3), 25, dtype=np.uint8)

    cv2.putText(panel, "EVALUASI SCREENSHOT", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 215, 255), 2)
    cv2.putText(panel, f"Target: {roi_info}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, f"File: {os.path.basename(image_path)[:25]}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.putText(panel, "Prediksi Top-5 2D-CNN:", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    y_bar = 190
    for rank, idx in enumerate(top5_idx):
        cls_name = classes[idx]
        pct = probs[idx] * 100
        # Label teks
        cv2.putText(panel, f"{rank+1}. Huruf {cls_name}: {pct:.1f}%", (25, y_bar), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        # Bar grafik
        bar_len = int((pct / 100.0) * 250)
        bar_color = (0, 255, 127) if rank == 0 and pct > 50 else ((0, 215, 255) if rank == 0 else (120, 120, 120))
        cv2.rectangle(panel, (25, y_bar + 8), (25 + 250, y_bar + 22), (40, 40, 40), -1)
        if bar_len > 0:
            cv2.rectangle(panel, (25, y_bar + 8), (25 + bar_len, y_bar + 22), bar_color, -1)
        y_bar += 48

    cv2.putText(panel, "[Q] Keluar | [N] Gambar Berikutnya", (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

    # Gabung gambar dan panel
    combined = np.hstack([display_img, panel])
    win_title = "Evaluasi Screenshot BISINDO"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, min(combined.shape[1], 1280), min(combined.shape[0], 720))
    cv2.imshow(win_title, combined)
    print(f"\nEvaluasi: {os.path.basename(image_path)}")
    print(f"Top-1: Huruf {classes[top5_idx[0]]} ({probs[top5_idx[0]]*100:.2f}%)")

def main():
    user_dir = r"C:\Users\lannnn\.gemini\antigravity-ide\brain\696f81b7-5bca-4acf-b59a-9a5dcb0dba08\.user_uploaded"
    uploaded = sorted(glob.glob(os.path.join(user_dir, "*.*")), key=os.path.getmtime)
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        evaluate_and_visualize(sys.argv[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if not uploaded:
        print("Tidak ada screenshot di folder user_uploaded.")
        return

    idx = len(uploaded) - 1
    while True:
        evaluate_and_visualize(uploaded[idx])
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q") or k == 27:
            break
        elif k == ord("n") or k == ord("N"):
            idx = (idx + 1) % len(uploaded)
        elif k == ord("p") or k == ord("P"):
            idx = (idx - 1) % len(uploaded)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
