from pathlib import Path

# Dataset root
ROOT = Path("C:/Users/kmand/Case-a-thon/classification_dataset")

# Artifact dir
SAVE_DIR = ROOT.parent / "benthic_artifacts"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# File paths for checkpoints/exports
CKPT_LAST   = SAVE_DIR / "convnext_tiny_last.pth"      # latest checkpoint
CKPT_BEST   = SAVE_DIR / "convnext_tiny_best.pth"      # best-by-val checkpoint
TS_EXPORT   = SAVE_DIR / "convnext_tiny_scripted.pt"   # TorchScript export
CLASSES_JSON= SAVE_DIR / "classes.json"                # class names

# -------------------- Config & Repro --------------------
import torch, random, numpy as np

CLASS_NAMES = ["Scallop", "roundfish", "crab", "whelk", "skate", "flatfish", "Eel"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

BATCH = 64
EPOCHS = 10
EXTRA_EPOCHS = 10
LR = 3e-4
WD = 1e-2
SEED = 12345

RESUME = True
RESUME_FROM = "last"  # "last" or "best"
RESUME_PATH = CKPT_LAST if RESUME_FROM == "last" else CKPT_BEST

# Colab-friendly DataLoader settings
NUM_WORKERS = 3
PIN_MEMORY = True
PERSISTENT = False
PREFETCH = 2

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------- Imports & Utils --------------------
from typing import List, Tuple
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import json

# Parse labels.txt and build a list of (image_path, class_id, year) samples.
def read_labels(root: str):
    root = Path(root)
    labels = root / "labels.txt"; images = root / "images"
    samples: List[Tuple[Path, int, str]] = []
    with labels.open("r", encoding="utf-8") as f:
        for line in f:
            filename, label = line.split()
            path = images / filename
            cid = CLASS_TO_ID[label]
            year = filename.split("_", 1)[0]
            samples.append((path, cid, year))
    return samples

# Create a class-stratified 70/15/15 train/val/test split for IID evaluation.
def stratified_iid_split(samples, seed=12345):
    y = np.array([cid for _, cid, _ in samples])
    trainval, test = train_test_split(samples, test_size=0.15, stratify=y, random_state=seed)
    y_trainval = np.array([cid for _, cid, _ in trainval])
    train, val = train_test_split(trainval, test_size=0.1765, stratify=y_trainval, random_state=seed)
    return train, val, test

# Minimal Dataset that loads an image (with optional transform) and its class id.
class BenthicDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items; self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, cid, _ = self.items[i]
        with Image.open(path) as im: img = im.convert("RGB")
        if self.transform: img = self.transform(img)
        return img, cid

# Build train (augmented) and val/test (deterministic) transforms for 224×224 inputs.
def build_transforms():
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    val_tf = weights.transforms(antialias=True)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0),
                                     interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR, fill=0),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf

# Evaluate top-1 accuracy over a DataLoader with no grad in eval mode.
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / total if total else 0.0

# -------------------- Checkpoint / Resume Helpers --------------------

# Save a full training snapshot (model + epoch/metrics + optimizer/scaler + hparams).
def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_acc: float, *,
                    optimizer=None, scaler=None, extra: dict = None):
    payload = {
        "epoch": int(epoch),
        "val_acc": float(val_acc),
        "state_dict": model.state_dict(),
        "arch": "convnext_tiny",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "hparams": {"lr": LR, "weight_decay": WD, "batch": BATCH, "label_smoothing": 0.1},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if extra: payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)

# Export a portable TorchScript model for inference without the Python class.
def export_torchscript(model: nn.Module, path: Path, device: torch.device):
    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.trace(model_cpu, torch.randn(1, 3, 224, 224))
    scripted.save(str(path))
    model.to(device)

# Rebuild model/optimizer/scaler from a checkpoint and return resume state.
def resume_from_checkpoint(ckpt_path: Path, device, device_type):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"]); model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("val_acc", 0.0))
    return model, optimizer, scaler, start_epoch, best_val

# -------------------- Build loaders --------------------
samples = read_labels(ROOT)
train_items, val_items, test_items = stratified_iid_split(samples, seed=SEED)
train_tf, val_tf = build_transforms()

train_loader = DataLoader(
    BenthicDataset(train_items, transform=train_tf),
    batch_size=BATCH, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT, prefetch_factor=PREFETCH,
    drop_last=True)

val_loader = DataLoader(
    BenthicDataset(val_items, transform=val_tf),
    batch_size=BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT, prefetch_factor=PREFETCH,
    drop_last=False)

test_loader = DataLoader(
    BenthicDataset(test_items, transform=val_tf),
    batch_size=BATCH, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT, prefetch_factor=PREFETCH,
    drop_last=False)

# -------------------- Model + AMP (Init or Resume) --------------------

# Select device (GPU if available) and set up AMP (automatic mixed precision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if torch.cuda.is_available() else "cpu"
autocast_cm = torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda"))
scaler = torch.amp.GradScaler(device_type) if device_type == "cuda" else None

# Persist class names for inference scripts (map logits -> labels without importing this notebook)
with CLASSES_JSON.open("w", encoding="utf-8") as f:
    json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)

# Resume path: load a saved checkpoint if requested and available
if RESUME and RESUME_PATH.exists():
    model, opt, scaler, start_epoch, best_val = resume_from_checkpoint(RESUME_PATH, device, device_type)
    print(f"Resumed from {RESUME_PATH} at epoch {start_epoch} (best val_acc={best_val:.3f})")
    end_epoch = (start_epoch - 1) + EXTRA_EPOCHS

# Fresh start: initialize from ImageNet weights and create optimizer
else:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    start_epoch, best_val = 1, 0.0
    end_epoch = EPOCHS


# -------------------- Training Loop --------------------
for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    running = 0.0; n_seen = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}"):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with autocast_cm:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        running += loss.item() * xb.size(0); n_seen += xb.size(0)

    train_loss = running / max(1, n_seen)
    val_acc = evaluate(model, val_loader, device)
    print(f"loss={train_loss:.4f}  val_acc={val_acc:.3f}")

    # save last each epoch
    save_checkpoint(CKPT_LAST, model, epoch, val_acc, optimizer=opt, scaler=scaler, extra={"split": "iid"})

    # save best and export torchscript
    if val_acc > best_val:
        best_val = val_acc
        save_checkpoint(CKPT_BEST, model, epoch, val_acc, optimizer=opt, scaler=scaler, extra={"split": "iid"})
        export_torchscript(model, TS_EXPORT, device)
        print(f"  ↳ saved BEST: {CKPT_BEST}")
        print(f"  ↳ exported TorchScript: {TS_EXPORT}")

    # --- Load best checkpoint and eval on test_loader ---
import torch, torch.nn as nn
from torchvision.models import convnext_tiny

ckpt_path = "/content/drive/MyDrive/data/benthic_artifacts/convnext_tiny_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(ckpt_path, map_location="cpu")
model = convnext_tiny(weights=None)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, ckpt["num_classes"])
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

# assumes you still have `evaluate()` and `test_loader` defined
test_acc = evaluate(model, test_loader, device)
print(f"Test acc: {test_acc:.3f}")


# -------------------- Final test --------------------
test_acc = evaluate(model, test_loader, device)
print(f"Test acc: {test_acc:.3f}")