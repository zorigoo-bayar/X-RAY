# train_efficientnet_b0.py
"""
Train EfficientNet-B0 multi-label chest X-ray.
Use subset.csv if USE_SUBSET_FROM_FILE=True else use full CSV.
"""

import os, time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score

# ----------------- USER SETTINGS -----------------
root_dir = r"C:/Users/hitech/OneDrive/Documents/X ray"
csv_path = os.path.join(root_dir, "Data_Entry_2017.csv")
subset_csv = os.path.join(root_dir, "subset.csv")  # created by build_subset.py

USE_SUBSET_FROM_FILE = True   # if True read subset.csv (must exist)
BATCH_SIZE = 24              # try 24/32 on GPU, reduce on CPU
IMG_SIZE = 224               # or 160/128 to speed up
EPOCHS = 10
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 0              # Windows safe
USE_AMP = True               # mixed precision on GPU
# -------------------------------------------------

# choose CSV
if USE_SUBSET_FROM_FILE and os.path.exists(subset_csv):
    df = pd.read_csv(subset_csv)
    print("Using subset CSV:", subset_csv)
else:
    df = pd.read_csv(csv_path)
    print("Using full CSV:", csv_path)

# parse labels
def parse_labels(x):
    labs = [p.strip() for p in str(x).split("|") if p.strip()!=""]
    return [] if "No Finding" in labs else labs

df["labels_list"] = df["Finding Labels"].apply(parse_labels)

all_labels = sorted({lab for labs in df["labels_list"] for lab in labs})
print("Detected labels:", all_labels)
NUM_CLASSES = len(all_labels)

mlb = MultiLabelBinarizer(classes=all_labels)
Y = mlb.fit_transform(df["labels_list"])

# transforms
train_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# build file index
def build_file_index(root_dir, exts=(".png",".jpg",".jpeg")):
    file_map = {}
    entries = sorted(os.listdir(root_dir))
    subdirs = [os.path.join(root_dir,d) for d in entries if d.startswith("images_") and os.path.isdir(os.path.join(root_dir,d))]
    if len(subdirs)==0:
        subdirs = [os.path.join(root_dir,d) for d in entries if os.path.isdir(os.path.join(root_dir,d))]
    for sd in subdirs:
        for base,_,files in os.walk(sd):
            for f in files:
                if f.lower().endswith(exts) and f not in file_map:
                    file_map[f] = os.path.join(base,f)
    return file_map

FILE_INDEX = build_file_index(root_dir)
print("Indexed images:", len(FILE_INDEX))

def find_file(fname):
    if fname in FILE_INDEX:
        return FILE_INDEX[fname]
    return FILE_INDEX.get(os.path.basename(fname), None)

# Dataset
class XrayDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row["Image Index"]).strip()
        p = find_file(fname)
        if p is None:
            raise FileNotFoundError(fname)
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        label = torch.tensor(mlb.transform([row["labels_list"]])[0], dtype=torch.float32)
        return img, label

dataset = XrayDataset(df, train_tf)
n = len(dataset)
n_val = int(VAL_SPLIT * n)
n_train = n - n_val
print(f"Samples: {n} | Train: {n_train} | Val: {n_val}")

train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

# model (EfficientNet-B0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# classifier: EfficientNet_B0 has classifier = Sequential(Dropout, Linear)
in_ft = eff.classifier[1].in_features
eff.classifier[1] = nn.Linear(in_ft, NUM_CLASSES)
model = eff.to(device)

# loss + optimizer (pos_weight)
label_counts = Y.sum(axis=0)
neg_counts = len(Y) - label_counts
pos_weight = (neg_counts / (label_counts + 1e-6)).astype(np.float32)
pos_weight_tensor = torch.tensor(pos_weight).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))

# evaluation
def evaluate(model, loader):
    model.eval()
    ys, preds = [], []
    tot = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            tot += loss.item() * imgs.size(0)
            probs = torch.sigmoid(out).cpu().numpy()
            ys.append(labels.cpu().numpy()); preds.append(probs)
    if len(ys)==0: return float('nan'), float('nan')
    ys = np.vstack(ys); preds = np.vstack(preds)
    aucs=[]
    for i in range(ys.shape[1]):
        if len(np.unique(ys[:,i]))>1:
            try: aucs.append(roc_auc_score(ys[:,i], preds[:,i]))
            except: aucs.append(np.nan)
        else: aucs.append(np.nan)
    return tot/len(loader.dataset), np.nanmean(aucs)

# training loop
best_auc = 0.0
start = time.time()
for epoch in range(1, EPOCHS+1):
    model.train(); running = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        if USE_AMP and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                out = model(imgs); loss = criterion(out, labels)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            out = model(imgs); loss = criterion(out, labels); loss.backward(); optimizer.step()
        running += loss.item() * imgs.size(0)
        pbar.set_postfix(train_loss=running / ((pbar.n+1) * BATCH_SIZE))
    train_loss = running / len(train_loader.dataset)
    val_loss, val_auc = evaluate(model, val_loader)
    print(f"\nEpoch {epoch}: Train={train_loss:.4f} Val={val_loss:.4f} AUC={val_auc:.4f}")
    scheduler.step(val_auc)
    if not np.isnan(val_auc) and val_auc > best_auc:
        best_auc = val_auc
        torch.save({"model_state_dict": model.state_dict(), "all_labels": all_labels},
                   os.path.join(root_dir, "efficientnet_b0_best.pth"))
        print("Saved best checkpoint.")
# final save
torch.save({"model_state_dict": model.state_dict(), "all_labels": all_labels},
           os.path.join(root_dir, "efficientnet_b0_final.pth"))
end = time.time()
print("Finished. Best AUC:", best_auc, "Elapsed (s):", end-start)
