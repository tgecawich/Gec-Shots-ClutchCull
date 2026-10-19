"""Train the jersey digit classifier from labeled crops. GROUNDWORK - this is
step 3 of the pipeline (extract_jersey_crops.py -> label_crops.html -> here)
and is unusable until a labels.csv exists, so it has not been run end-to-end.
Review the TODOs before the first real training run.

Why a classifier instead of the CRNN jersey.py uses today: measured recall on
real game frames is ~29% because the CRNN was trained on documents/signage, not
stylised athletic numerals. A small classifier trained on actual jersey crops
turns "read arbitrary text" into "which of 0-99 (or nothing) is this", a far
easier problem with exactly the data this tool chain produces.

Usage (after labeling):
    python tools/train_digits.py tools/crops tools/labels.csv

Needs torch + torchvision (local only, never a server dependency):
    pip install torch torchvision

Outputs tools/jersey_digits.onnx, which jersey.py can load with cv2.dnn like
every other model in this project.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# Classes: "none" (no number visible) + 0..99. Labels "unreadable" and "bad"
# are excluded from training entirely: "unreadable even to a human" carries no
# signal, and "bad crop" means the detector failed, not the classifier.
NONE_CLASS = 0
NUM_CLASSES = 101  # none + 00..99
INPUT_SIZE = 96    # square crop fed to the net; matches a chest-crop aspect
VAL_FRACTION = 0.15
EPOCHS = 30
BATCH = 64
LR = 3e-4
SEED = 22


def class_of(label: str) -> int | None:
    if label == "none":
        return NONE_CLASS
    if label in ("unreadable", "bad"):
        return None
    if label.isdigit() and 0 <= int(label) <= 99:
        return 1 + int(label)
    return None


def load_rows(crops_root: Path, labels_csv: Path):
    rows = []
    with open(labels_csv, newline="") as f:
        for name, label in list(csv.reader(f))[1:]:
            cls = class_of(label.strip())
            if cls is None:
                continue
            p = crops_root / name
            if p.exists():
                rows.append((p, cls))
    return rows


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    crops_root, labels_csv = Path(sys.argv[1]), Path(sys.argv[2])

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from PIL import Image
    except ImportError:
        print("Install training deps first: pip install torch torchvision")
        sys.exit(1)

    rows = load_rows(crops_root, labels_csv)
    if len(rows) < 200:
        print(f"Only {len(rows)} usable labels; aim for 500+ before training.")
        if len(rows) < 50:
            sys.exit(1)

    random.Random(SEED).shuffle(rows)
    n_val = max(1, int(len(rows) * VAL_FRACTION))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"{len(train_rows)} train / {len(val_rows)} val")

    # Augmentation mirrors what game frames actually do to numbers: perspective
    # (turned torsos), blur (motion), color jitter (jersey/lighting variety).
    # No horizontal flip: it would turn readable digits into garbage.
    train_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomPerspective(0.25, p=0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.GaussianBlur(3, (0.1, 1.5)),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
    ])

    class Crops(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            p, cls = self.rows[i]
            return self.tf(Image.open(p).convert("RGB")), cls

    # TODO before first real run: pretrain on SVHN (torchvision.datasets.SVHN)
    # and fine-tune here; with only ~500 jersey crops that will matter a lot.
    net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(128, NUM_CLASSES),
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    train_dl = DataLoader(Crops(train_rows, train_tf), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(Crops(val_rows, val_tf), batch_size=BATCH)

    best = 0.0
    out = Path(__file__).parent / "jersey_digits.onnx"
    for epoch in range(EPOCHS):
        net.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = net(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {epoch + 1:2d}/{EPOCHS}  val acc {acc:.3f}")
        if acc > best:
            best = acc
            torch.onnx.export(
                net.cpu(), torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE), str(out),
                input_names=["input"], output_names=["logits"], opset_version=13,
            )
            net.to(device)
    print(f"best val acc {best:.3f}; exported {out}")
    print("Next: wire into jersey.py behind an env flag and re-measure recall "
          "on the same frames that produced the 29% baseline.")


if __name__ == "__main__":
    main()
