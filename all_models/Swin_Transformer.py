# ============================================================
# IMAGE-ONLY 6-CLASS FAKE-NEWS CLASSIFIER (PAPER-ALIGNED)
# ============================================================

import os, math, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from transformers import SwinModel, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# ------------------------------------------------------------
# 0. Reproducibility + device
# ------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# 1. Load data + resolve image paths
# ------------------------------------------------------------
DATASET_PATH = "/kaggle/input/datasets/charumaurya15/multimodal-fake-news-processed"
CSV_PATH = os.path.join(DATASET_PATH, "master_dataset.csv")
if not os.path.exists(CSV_PATH):
    for root, _, files in os.walk("/kaggle/input"):
        if "master_dataset.csv" in files:
            CSV_PATH = os.path.join(root, "master_dataset.csv")
            DATASET_PATH = root
            break

df = pd.read_csv(CSV_PATH)
print("CSV:", CSV_PATH, "| Shape:", df.shape)

LABEL_NAMES = {
    0: "True", 1: "Satire / Parody", 2: "Misleading Content",
    3: "Imposter Content", 4: "False Connection", 5: "Manipulated Content"
}

def get_actual_image_path(row):
    if pd.isna(row["image_path"]):
        return None
    filename = os.path.basename(str(row["image_path"]))
    source = str(row["source"]).lower()
    if source == "fakenewsnet":
        folder = os.path.join(DATASET_PATH, "fakenewsnet", "fakenewsnet", "images")
    elif source == "reddit":
        folder = os.path.join(DATASET_PATH, "reddit", "reddit", "images")
    elif source == "twitter":
        folder = os.path.join(DATASET_PATH, "twitter", "twitter", "images")
    else:
        return None
    path = os.path.join(folder, filename)
    return path if os.path.exists(path) else None

df["actual_image_path"] = df.apply(get_actual_image_path, axis=1)
df = df[df["actual_image_path"].notna()].reset_index(drop=True)
print("With images:", len(df))
print(df["label"].value_counts().sort_index())

# ------------------------------------------------------------
# 2. Split
# ------------------------------------------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)
print("Split -> train/val/test:", len(train_df), len(val_df), len(test_df))

# ------------------------------------------------------------
# 3. Transforms
# ------------------------------------------------------------
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class ImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["actual_image_path"]).convert("RGB")
        return self.transform(img), int(row["label"])

train_ds = ImageDataset(train_df, train_tfms)
val_ds   = ImageDataset(val_df,   eval_tfms)
test_ds  = ImageDataset(test_df,  eval_tfms)

# ------------------------------------------------------------
# 4. DataLoaders (Standard Shuffling)
# ------------------------------------------------------------
BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# ------------------------------------------------------------
# 5. Model Architecture
# ------------------------------------------------------------
class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        
        # Load Swin Transformer matching the 1024-d pooled feature extraction
        self.backbone = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224", 
            use_safetensors=True
        )
        
        # Strictly freeze the entire visual backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Projection layer: 1024-d native features mapped to 768-d space
        self.projection_head = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 768)
        )
        
        # Linear classifier on top of the projected features
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, pixel_values):
        # Extract features without gradients for backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output # Shape: (batch_size, 1024)
            
        # Project and classify
        projected_features = self.projection_head(pooled_output)
        logits = self.classifier(projected_features)
        
        return logits

model = ImageOnlyModel(num_classes=6).to(device)

# Count trainable params (Should only include projection + classification layers)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# ------------------------------------------------------------
# 6. Optimizer & Loss
# ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# ------------------------------------------------------------
# 7. Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    loss_sum = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels_d = labels.to(device, non_blocking=True)

        logits = model(pixel_values=imgs)
        loss_sum += criterion(logits, labels_d).item() * labels.size(0)
        
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels); all_preds = np.array(all_preds)
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {
        "loss": loss_sum / len(all_labels),
        "acc": acc, "prec": p, "rec": r, "f1": f1,
        "labels": all_labels, "preds": all_preds,
    }

# ------------------------------------------------------------
# 8. Training loop
# ------------------------------------------------------------
best_f1 = -1.0
best_state = None
patience, patience_counter = 4, 0

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values=imgs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct/total:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    train_loss = running_loss / total
    train_acc  = correct / total

    val = evaluate(model, val_loader)

    print(
        f"\nEpoch {epoch+1}: "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
        f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
        f"val_macroF1={val['f1']:.4f}"
    )

    if val["f1"] > best_f1:
        best_f1 = val["f1"]
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, "/kaggle/working/best_image_only_ablation.pt")
        print("*** New best saved ***")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered by validation loss patience.")
            break

# ------------------------------------------------------------
# 9. Final test evaluation
# ------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)

test = evaluate(model, test_loader)

print("\n" + "=" * 60)
print("FINAL TEST (PAPER-ALIGNED IMAGE-ONLY ABLATION)")
print("=" * 60)
print(f"Accuracy       : {test['acc']:.4f}")
print(f"Macro Precision: {test['prec']:.4f}")
print(f"Macro Recall   : {test['rec']:.4f}")
print(f"Macro F1       : {test['f1']:.4f}")

print("\nClassification report:")
print(classification_report(
    test["labels"], test["preds"],
    labels=list(range(6)),
    target_names=[LABEL_NAMES[i] for i in range(6)],
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(test["labels"], test["preds"], labels=list(range(6))))

# ============================================================
# EXTRACT RAW SWIN FEATURES (1024-dim) FOR TWOFUSE-W FUSION
# ============================================================
def extract_raw_swin_features(loader, model):
    model.eval()
    feats, lbls = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Swin raw features"):
            imgs = imgs.to(device)
            outputs = model.backbone(pixel_values=imgs)
            feats.append(outputs.pooler_output.cpu())   # (B, 1024)
            lbls.append(labels)
    return torch.cat(feats), torch.cat(lbls)

# make sure best weights are loaded
if best_state is not None:
    model.load_state_dict(best_state)

tr_f, tr_l = extract_raw_swin_features(train_loader, model)
va_f, va_l = extract_raw_swin_features(val_loader,   model)
te_f, te_l = extract_raw_swin_features(test_loader,  model)

torch.save({"features": tr_f, "labels": tr_l}, "/kaggle/working/train_image_raw_features.pt")
torch.save({"features": va_f, "labels": va_l}, "/kaggle/working/val_image_raw_features.pt")
torch.save({"features": te_f, "labels": te_l}, "/kaggle/working/test_image_raw_features.pt")

print("Raw Swin features saved:", tr_f.shape, va_f.shape, te_f.shape)

# Re-create loaders with shuffle=False and drop_last=False JUST for feature extraction
fix_train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
fix_val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
fix_test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# Re-extract
tr_f, tr_l = extract_raw_swin_features(fix_train_loader, model)
va_f, va_l = extract_raw_swin_features(fix_val_loader, model)
te_f, te_l = extract_raw_swin_features(fix_test_loader, model)

# Overwrite the files
torch.save({"features": tr_f, "labels": tr_l}, "/kaggle/working/train_image_raw_features.pt")
torch.save({"features": va_f, "labels": va_l}, "/kaggle/working/val_image_raw_features.pt")
torch.save({"features": te_f, "labels": te_l}, "/kaggle/working/test_image_raw_features.pt")

print("Fixed Swin features saved:", tr_f.shape, va_f.shape, te_f.shape)
