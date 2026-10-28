import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. Pilih Device
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
print(f"Perangkat pelatihan 1D-CNN: {device}")

# 2. Load Dataset Koordinat Landmark
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data = np.load(os.path.join(BASE_DIR, "landmarks_train.npz"))
val_data = np.load(os.path.join(BASE_DIR, "landmarks_val.npz"))

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
classes = train_data["classes"]

print(f"Data latih : {X_train.shape[0]} sampel (Shape: {X_train.shape})")
print(f"Data validasi: {X_val.shape[0]} sampel (Shape: {X_val.shape})")

# Konversi ke PyTorch Tensor
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Arsitektur 1D-CNN Landmark
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
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SignLandmarkCNN(in_channels=3, num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

# 4. Loop Pelatihan
epochs = 40
best_val_acc = 0.0
best_epoch = 0
out_model_path = os.path.join(BASE_DIR, "best_landmark_cnn.pth")

t_start = time.time()
print("\n" + "="*60)
print("🚀 MEMULAI PELATIHAN 1D-CNN LANDMARK BISINDO")
print("="*60)

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Data Augmentation on-the-fly: jitter halus koordinat
        noise = torch.randn_like(inputs) * 0.008
        inputs = inputs + noise
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data).item()
        total_train += labels.size(0)
        
    train_loss = running_loss / total_train
    train_acc = (correct_train / total_train) * 100.0
    
    # Validasi
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)
            
    val_loss = val_loss / total_val
    val_acc = (correct_val / total_val) * 100.0
    scheduler.step()
    
    saved_str = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), out_model_path)
        saved_str = "⭐ [BEST MODEL DISIMPAN]"
        
    if epoch % 5 == 0 or epoch == epochs or "BEST" in saved_str:
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% {saved_str}")

total_time = time.time() - t_start
print("\n" + "="*60)
print(f"✅ PELATIHAN SELESAI dalam {total_time:.1f} detik!")
print(f"🏆 Akurasi Validasi Terbaik: {best_val_acc:.2f}% (Epoch {best_epoch})")
print(f"💾 File Model Tersimpan: {out_model_path} ({os.path.getsize(out_model_path)/(1024*1024):.2f} MB)")
print("="*60)
