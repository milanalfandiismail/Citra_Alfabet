import os
import glob
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
if not os.path.exists(TASK_PATH):
    print("Error: hand_landmarker.task tidak ditemukan!")
    exit(1)

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.18,
    min_hand_presence_confidence=0.18
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = pts[0].copy()
    pts -= wrist
    scale = np.linalg.norm(pts[9])
    if scale > 1e-4:
        pts /= scale
    return pts

def extract_from_split(split_name, max_per_class=180):
    base_dir = os.path.join(BASE_DIR, "bisindo", "images", split_name)
    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    X_list = []
    y_list = []
    
    t0 = time.time()
    print(f"\nMemulai ekstraksi split '{split_name}' (Target max {max_per_class} per kelas)...")
    
    total_imgs = 0
    extracted_imgs = 0
    
    for cls_idx, cls_name in enumerate(classes):
        folder = os.path.join(base_dir, cls_name)
        img_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))
        
        cls_count = 0
        for p in img_paths:
            if cls_count >= max_per_class:
                break
            total_imgs += 1
            
            img = cv2.imread(p)
            if img is None:
                continue
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)
            
            if not res.hand_landmarks:
                continue
                
            # Hand 1
            h1 = normalize_hand(res.hand_landmarks[0])
            # Hand 2 (jika ada, jika tidak isi 0)
            if len(res.hand_landmarks) > 1:
                h2 = normalize_hand(res.hand_landmarks[1])
            else:
                h2 = np.zeros((21, 3), dtype=np.float32)
                
            # Gabungkan jadi (42, 3) lalu transpose ke (3, 42)
            combined = np.vstack([h1, h2]).T  # shape: (3, 42)
            
            X_list.append(combined)
            y_list.append(cls_idx)
            cls_count += 1
            extracted_imgs += 1
            
        print(f" - [{split_name.upper()}] Kelas {cls_name} ({cls_idx+1}/26): {cls_count} sampel berhasil diekstrak")
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.int64)
    out_file = os.path.join(BASE_DIR, f"landmarks_{split_name}.npz")
    np.savez_compressed(out_file, X=X_arr, y=y_arr, classes=classes)
    
    elapsed = time.time() - t0
    print(f"\n✅ Selesai '{split_name}': {extracted_imgs}/{total_imgs} sampel ({extracted_imgs/max(1, total_imgs)*100:.1f}%) dalam {elapsed:.1f} detik -> Disimpan ke: {out_file}")
    print(f"   Shape X: {X_arr.shape}, Shape y: {y_arr.shape}")

if __name__ == "__main__":
    extract_from_split("val", max_per_class=35)
