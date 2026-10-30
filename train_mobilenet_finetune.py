"""
Script Pelatihan & Fine-Tuning Model 2D-CNN (MobileNetV2)
Alfabet Bahasa Isyarat Indonesia (BISINDO)

Fitur Utama:
1. Akselerasi Dual-Mode: AMD DirectML (GPU Lokal) / NVIDIA CUDA / CPU Fallback.
2. Two-Stage Fine-Tuning:
   - Stage 1: Warmup Classifier Head (Freeze Backbone)
   - Stage 2: Unfreeze 4 Blok Konvolusi Teratas (features[14:]) dengan Differential Learning Rate.
3. Augmentasi Anti-Background-Noise: RandomErasing, RandomAffine, ColorJitter, GaussianBlur.
4. Auto-Backup bobot lama ke best_mobilenetv2_backup.pth dan simpan hasil terbaik ke best_mobilenetv2.pth.
"""

import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Pastikan output teks terminal mendukung UTF-8 di Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# -------------------------------------------------------------------------
# 1. SETUP LINGKUNGAN & PERANGKAT GPU
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "bisindo", "images", "train")
VAL_DIR = os.path.join(BASE_DIR, "bisindo", "images", "val")
MODEL_OUT_PATH = os.path.join(BASE_DIR, "best_mobilenetv2.pth")
BACKUP_PATH = os.path.join(BASE_DIR, "best_mobilenetv2_backup.pth")

def get_optimal_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            dev = torch_directml.device()
            print(f"🚀 Terdeteksi Akselerasi AMD DirectML GPU Lokal: {dev}")
            return dev
    except Exception:
        pass

    if torch.cuda.is_available():
        print(f"🚀 Terdeteksi GPU NVIDIA (CUDA): {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    print("⚠️ GPU tidak ditemukan, menggunakan CPU.")
    return torch.device("cpu")

def build_mobilenet_v2(num_classes=26):
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        print("✅ Bobot awal ImageNet pre-trained berhasil dimuat.")
    except Exception:
        print("⚠️ Menggunakan arsitektur MobileNetV2 lokal (tanpa download ImageNet).")
        model = models.mobilenet_v2(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model

def main():
    device = get_optimal_device()

    print("\n" + "=" * 65)
    print("📦 MEMPERSIAPKAN DATASET & AUGMENTASI DATA (ANTI-NOISE)")
    print("=" * 65)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.30, scale=(0.02, 0.20), value='random')
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"❌ Direktori dataset tidak ditemukan di:\n  Train: {TRAIN_DIR}\n  Val: {VAL_DIR}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val)

    BATCH_SIZE = 32
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    classes = train_dataset.classes
    NUM_CLASSES = len(classes)
    print(f"✅ Data Latih : {len(train_dataset)} citra ({NUM_CLASSES} kelas: {', '.join(classes[:5])} ... {', '.join(classes[-3:])})")
    print(f"✅ Data Val   : {len(val_dataset)} citra")

    # Inisialisasi model
    model = build_mobilenet_v2(num_classes=NUM_CLASSES)

    # Backup bobot lama jika ada
    if os.path.exists(MODEL_OUT_PATH):
        try:
            shutil.copy2(MODEL_OUT_PATH, BACKUP_PATH)
            print(f"💾 Cadangan model lama disimpan ke: {os.path.basename(BACKUP_PATH)}")
            state = torch.load(MODEL_OUT_PATH, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            print(f"🔄 Berhasil memuat bobot awal dari {os.path.basename(MODEL_OUT_PATH)}.")
        except Exception as e:
            print(f"⚠️ Gagal memuat bobot lama ({e}), memulai dengan bobot default.")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    def evaluate(m, loader):
        m.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = m(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return total_loss / max(1, total), (correct / max(1, total)) * 100.0

    def train_epoch(m, loader, opt):
        m.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            opt.zero_grad()
            outputs = m(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return total_loss / max(1, total), (correct / max(1, total)) * 100.0

    # Evaluasi baseline sebelum pelatihan
    baseline_loss, baseline_acc = evaluate(model, val_loader)
    print(f"\n📊 Baseline Akurasi Validasi Saat Ini: {baseline_acc:.2f}% (Loss: {baseline_loss:.4f})")
    best_acc = baseline_acc

    # STAGE 1: Warmup Classifier Head
    print("\n" + "=" * 65)
    print("🚀 STAGE 1: WARMUP CLASSIFIER HEAD (Backbone Beku)")
    print("=" * 65)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter dilatih pada Stage 1: {trainable_p1:,} params")

    optimizer_stage1 = optim.Adam(model.classifier.parameters(), lr=1e-3, foreach=False)
    EPOCHS_STAGE1 = 4

    for ep in range(1, EPOCHS_STAGE1 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_stage1)
        val_loss, val_acc = evaluate(model, val_loader)
        dur = time.time() - t0
        
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            improved = "⭐ [BEST CHECKPOINT SAVED]"
            
        print(f"Epoch [{ep}/{EPOCHS_STAGE1}] ({dur:.1f}s) | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% {improved}")

    # STAGE 2: Deep Fine-Tuning (Unfreeze features[14:])
    print("\n" + "=" * 65)
    print("🚀 STAGE 2: DEEP FINE-TUNING (Unfreeze features[14:] dengan Differential LR)")
    print("=" * 65)

    for i, layer in enumerate(model.features):
        if i >= 14:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            for param in layer.parameters():
                param.requires_grad = False

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter dilatih pada Stage 2: {trainable_p2:,} params")

    unfrozen_backbone_params = [p for p in model.features[14:].parameters() if p.requires_grad]
    optimizer_stage2 = optim.AdamW([
        {'params': unfrozen_backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}
    ], foreach=False)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode='max', factor=0.5, patience=2)

    EPOCHS_STAGE2 = 12
    patience = 4
    patience_counter = 0

    for ep in range(1, EPOCHS_STAGE2 + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer_stage2)
        val_loss, val_acc = evaluate(model, val_loader)
        dur = time.time() - t0
        scheduler.step(val_acc)
        
        improved = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            improved = "⭐ [BEST CHECKPOINT SAVED]"
            patience_counter = 0
        else:
            patience_counter += 1

        current_lrs = [f"{g['lr']:.1e}" for g in optimizer_stage2.param_groups]
        print(f"Epoch [{ep}/{EPOCHS_STAGE2}] ({dur:.1f}s) LRs: {current_lrs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% {improved}")

        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping aktif setelah {patience} epoch tanpa peningkatan validasi.")
            break

    print("\n" + "=" * 65)
    print("🎉 PELATIHAN FINE-TUNING SELESAI!")
    print(f"🏆 Akurasi Validasi Terbaik: {best_acc:.2f}%")
    print(f"💾 File Model Tersimpan     : {MODEL_OUT_PATH}")
    print(f"💾 File Cadangan Lama       : {BACKUP_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    main()
