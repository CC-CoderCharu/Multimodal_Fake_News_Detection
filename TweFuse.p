import os, copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1. Load Features (Auto-find)
def find_file(filename):
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None

txt_dir = os.path.dirname(find_file("train_text_features.pt"))
img_dir = os.path.dirname(find_file("train_image_raw_features.pt"))

# Use the fine-tuned text features
txt_tr = torch.load(os.path.join(txt_dir, "train_text_features.pt"))
txt_va = torch.load(os.path.join(txt_dir, "val_text_features.pt"))
txt_te = torch.load(os.path.join(txt_dir, "test_text_features.pt"))

img_tr = torch.load(os.path.join(img_dir, "train_image_raw_features.pt"))
img_va = torch.load(os.path.join(img_dir, "val_image_raw_features.pt"))
img_te = torch.load(os.path.join(img_dir, "test_image_raw_features.pt"))

class FusionDataset(Dataset):
    def __init__(self, t, i, y):
        self.t, self.i, self.y = t, i, y
    def __len__(self): return len(self.y)
    def __getitem__(self, k): return self.t[k], self.i[k], self.y[k]

train_loader = DataLoader(FusionDataset(txt_tr["features"], img_tr["features"], img_tr["labels"]), batch_size=32, shuffle=True)
val_loader   = DataLoader(FusionDataset(txt_va["features"], img_va["features"], img_va["labels"]), batch_size=32, shuffle=False)
test_loader  = DataLoader(FusionDataset(txt_te["features"], img_te["features"], img_te["labels"]), batch_size=32, shuffle=False)

# 2. Model with LayerNorm (Prevents scale bias)
class TwoFuseW(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, shared=768, n_classes=6):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, shared), nn.ReLU(), nn.Linear(shared, shared), nn.LayerNorm(shared)
        )
        self.img_proj = nn.Sequential(
            nn.Linear(image_dim, shared), nn.ReLU(), nn.Linear(shared, shared), nn.LayerNorm(shared)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * shared, 128), nn.ReLU(), nn.Linear(128, 2)
        )
        nn.init.zeros_(self.gate[2].weight)
        nn.init.zeros_(self.gate[2].bias)
        self.classifier = nn.Linear(shared, n_classes)

    def forward(self, text_feats, img_feats):
        T = self.text_proj(text_feats)
        V = self.img_proj(img_feats)
        a = torch.softmax(self.gate(torch.cat([T, V], dim=1)), dim=1)
        aT, aV = a[:, 0:1], a[:, 1:2]
        F = aT * T + aV * V
        return self.classifier(F), aT, aV

model = TwoFuseW().to(device)

# 3. Training Loop with Balance Regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
EPOCHS = 20
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)
scheduler = LambdaLR(optimizer, lambda s: s/warmup_steps if s < warmup_steps else max(0.0, (total_steps-s)/(total_steps-warmup_steps)))

@torch.no_grad()
def evaluate(loader, ret_weights=False):
    model.eval()
    preds, labels, aTs = [], [], []
    for t, i, y in loader:
        logits, aT, _ = model(t.to(device), i.to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        if ret_weights: aTs.extend(aT.squeeze(1).cpu().numpy())
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {"acc": accuracy_score(labels, preds), "f1": f1, "labels": np.array(labels), "preds": np.array(preds)}
    if ret_weights: out["alpha_T"] = np.array(aTs)
    return out

best_f1, best_state, patience = -1, None, 0
for ep in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")
    for t, i, y in pbar:
        t, i, y = t.to(device), i.to(device), y.to(device)
        optimizer.zero_grad()
        logits, aT, aV = model(t, i)
        
        # ADDING BALANCE REGULARIZATION (Forces aT towards 0.5)
        balance_penalty = 0.5 * ((aT - 0.5)**2).mean() 
        loss = criterion(logits, y) + balance_penalty
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    val = evaluate(val_loader)
    print(f"Epoch {ep+1}: val_acc={val['acc']:.4f} val_macroF1={val['f1']:.4f}")
    
    if val["f1"] > best_f1:
        best_f1 = val["f1"]
        best_state = copy.deepcopy(model.state_dict())
        patience = 0
    else:
        patience += 1
        if patience >= 4:
            print("Early stopping triggered.")
            break

model.load_state_dict(best_state)
test = evaluate(test_loader, ret_weights=True)

print("\n" + "="*60)
print(f"FINAL TEST MACRO F1: {test['f1']:.4f}")
print("="*60)

LABEL_NAMES = {0: "True", 1: "Satire", 2: "Misleading", 3: "Imposter", 4: "False Connection", 5: "Manipulated"}
print("\n=== Table IV: Learned α_T / α_V per class ===")
for c in range(6):
    m = test["labels"] == c
    if m.sum() == 0: continue
    print(f"{LABEL_NAMES[c]:18s} α_T={test['alpha_T'][m].mean():.3f}  α_V={1-test['alpha_T'][m].mean():.3f}")

# Plot
classes = list(range(6))
mT = [test["alpha_T"][test["labels"] == c].mean() for c in classes]
mV = [1 - x for x in mT]
plt.figure(figsize=(10, 4))
x = np.arange(6); w = 0.35
plt.bar(x-w/2, mT, w, label="Text α_T", color="#2b9fd8")
plt.bar(x+w/2, mV, w, label="Image α_V", color="#f2c200")
plt.xticks(x, [LABEL_NAMES[i] for i in classes], rotation=20)
plt.ylim(0, 1); plt.title("Fig. 4 — Modality Weights"); plt.legend(); plt.show()
