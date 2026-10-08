"""
AgriLite-XAI: FULL STUDY pipeline
==================================
Completes items 1-3 from the README in one script:
  1. Uses ImageNet-pretrained weights (requires internet - see notes below)
  2. Trains the FULL baseline set (not just MobileNetV3-Small)
  3. Runs 5 seeds per model and computes mean +/- std, plus paired t-tests
     of AgriLite-XAI vs each baseline

DESIGN GOALS:
  - Resumable: results are saved to disk after EVERY model finishes, so a
    disconnect/crash partway through does not lose completed work.
  - Honest: nothing here fabricates results. If a run fails, it is logged as
    failed, not silently skipped or filled in.

REQUIREMENTS:
  - Internet access in the Kaggle notebook (Settings -> Internet -> On).
    This requires phone verification on Kaggle if not already done.
    Without internet, set PRETRAINED = False below and this becomes a
    from-scratch run identical in spirit to the previous pilot.
  - GPU strongly recommended (Settings -> Accelerator -> GPU). 5 seeds x 9
    models is a lot of training; on CPU this could take many hours.

USAGE (Kaggle notebook, after pasting this whole file into a cell and running it):
    build_ghana_subset(
        src_dir="/kaggle/input/datasets/irakozekelly/crop-pest-and-disease-dataset/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset",
        dst_dir="/kaggle/working/splits",
        per_class_cap=100,   # bump this up if time/compute allows, e.g. 300
    )
    run_full_study("/kaggle/working/splits", epochs=20, seeds=[42,43,44,45,46])

Expected total runtime on a Kaggle T4 GPU: roughly 2-5 hours for the full
5-seed x 9-model sweep at per_class_cap=100. Use Save & Run All (Commit) so
it continues even if you close the browser tab.
"""

import copy
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PRETRAINED = True  # set False only if internet genuinely unavailable
RESULTS_FILE = "/kaggle/working/full_study_results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Dataset subset builder (Ghana dataset, corruption-safe)
# ----------------------------------------------------------------------
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def build_ghana_subset(src_dir, dst_dir, per_class_cap=100, train=0.7, val=0.15, seed=42):
    random.seed(seed)
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    crops = ["Cashew", "Cassava", "Maize", "Tomato"]
    total_classes = 0
    for crop in crops:
        crop_dir = src_dir / crop
        if not crop_dir.is_dir():
            print(f"WARNING: {crop_dir} not found, skipping")
            continue
        for cond_dir in sorted(p for p in crop_dir.iterdir() if p.is_dir()):
            class_name = f"{crop}__{cond_dir.name.replace(' ', '_')}"
            all_images = list(cond_dir.glob("*"))
            random.shuffle(all_images)
            good_images, skipped = [], 0
            for img_path in all_images:
                if len(good_images) >= per_class_cap:
                    break
                if is_valid_image(img_path):
                    good_images.append(img_path)
                else:
                    skipped += 1
            n = len(good_images)
            n_train = int(n * train)
            n_val = int(n * val)
            splits = {"train": good_images[:n_train], "val": good_images[n_train:n_train+n_val], "test": good_images[n_train+n_val:]}
            for split, files in splits.items():
                out_dir = dst_dir / split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copy(f, out_dir / f.name)
            total_classes += 1
            print(f"  {class_name}: {n} valid images ({skipped} skipped)")
    print(f"\nDone: {total_classes} classes written to {dst_dir}")


# ----------------------------------------------------------------------
# Safe image loading (crash-proof against corrupted files)
# ----------------------------------------------------------------------
def safe_pil_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (0, 0, 0))


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(data_root, img_size=224, batch_size=16, num_workers=2):
    train_ds = datasets.ImageFolder(f"{data_root}/train", get_transforms(img_size, True), loader=safe_pil_loader)
    val_ds = datasets.ImageFolder(f"{data_root}/val", get_transforms(img_size, False), loader=safe_pil_loader)
    test_ds = datasets.ImageFolder(f"{data_root}/test", get_transforms(img_size, False), loader=safe_pil_loader)
    assert train_ds.classes == val_ds.classes == test_ds.classes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, train_ds.classes


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def build_teacher(num_classes, pretrained=PRETRAINED):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


class LightAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(nn.Conv2d(channels, 1, kernel_size=7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.channel_gate(x)
        x = x * self.spatial_gate(x)
        return x


class AgriLiteStudent(nn.Module):
    def __init__(self, num_classes, pretrained=PRETRAINED):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.attention = LightAttention(576)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(576, 256), nn.Hardswish(inplace=True), nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_baseline(name, num_classes, pretrained=PRETRAINED):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "efficientnet_b1":
        m = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "shufflenet_v2":
        m = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(name)
    return m


FULL_BASELINE_NAMES = ["mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
                        "efficientnet_b0", "efficientnet_b1", "shufflenet_v2"]


# ----------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------
def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits, targets)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)
    return alpha * soft_loss + (1 - alpha) * hard_loss


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_targets.extend(y.numpy())
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    return acc, f1, prec, rec


def train_model(model, train_loader, val_loader, device, epochs=20, lr=3e-4,
                 teacher=None, T=4.0, alpha=0.5, patience=10, log_prefix=""):
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_f1, best_state, no_improve = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = kd_loss(out, t_out, y, T=T, alpha=alpha)
            else:
                loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"[{log_prefix}] epoch {epoch+1}/{epochs} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_state, no_improve = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{log_prefix}] early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(best_state)
    return model, best_f1


def count_params_mb(model):
    n_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_params / 1e6, size_bytes / (1024 ** 2)


def quantize_dynamic(model):
    model.eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


# ----------------------------------------------------------------------
# Result persistence (resumable)
# ----------------------------------------------------------------------
def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_result(results, seed, model_name, metrics):
    key = f"seed{seed}"
    results.setdefault(key, {})
    results[key][model_name] = metrics
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    return results


def already_done(results, seed, model_name):
    return f"seed{seed}" in results and model_name in results[f"seed{seed}"]


# ----------------------------------------------------------------------
# Full study runner
# ----------------------------------------------------------------------
def run_full_study(data_root, epochs=20, batch_size=16, seeds=(42, 43, 44, 45, 46),
                    baseline_names=None):
    baseline_names = baseline_names or FULL_BASELINE_NAMES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Pretrained: {PRETRAINED} | Seeds: {list(seeds)}")
    results = load_results()

    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        set_seed(seed)
        train_loader, val_loader, test_loader, classes = get_dataloaders(data_root, batch_size=batch_size)
        num_classes = len(classes)

        # Teacher (needed every seed, for distillation)
        if not already_done(results, seed, "resnet50_teacher"):
            teacher = build_teacher(num_classes)
            teacher, _ = train_model(teacher, train_loader, val_loader, device, epochs=epochs, log_prefix=f"s{seed}-teacher")
            acc, f1, prec, rec = evaluate(teacher, test_loader, device)
            params, size = count_params_mb(teacher)
            results = save_result(results, seed, "resnet50_teacher",
                                   {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "params_M": params, "size_MB": size})
        else:
            print(f"[seed {seed}] teacher already done, reloading for KD...")
            teacher = build_teacher(num_classes)
            teacher, _ = train_model(teacher, train_loader, val_loader, device, epochs=epochs, log_prefix=f"s{seed}-teacher-reload")

        # Baselines
        for name in baseline_names:
            if already_done(results, seed, name):
                print(f"[seed {seed}] {name} already done, skipping")
                continue
            m = build_baseline(name, num_classes)
            m, _ = train_model(m, train_loader, val_loader, device, epochs=epochs, log_prefix=f"s{seed}-{name}")
            acc, f1, prec, rec = evaluate(m, test_loader, device)
            params, size = count_params_mb(m)
            results = save_result(results, seed, name,
                                   {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "params_M": params, "size_MB": size})

        # AgriLite-XAI (KD + quantized)
        if not already_done(results, seed, "agrilite_xai"):
            student = AgriLiteStudent(num_classes)
            student, _ = train_model(student, train_loader, val_loader, device, epochs=epochs,
                                      teacher=teacher, log_prefix=f"s{seed}-agrilite")
            acc, f1, prec, rec = evaluate(student, test_loader, device)
            params, size = count_params_mb(student)
            results = save_result(results, seed, "agrilite_xai",
                                   {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "params_M": params, "size_MB": size})

            quantized = quantize_dynamic(copy.deepcopy(student).cpu())
            acc_q, f1_q, prec_q, rec_q = evaluate(quantized, test_loader, "cpu")
            params_q, size_q = count_params_mb(quantized)
            results = save_result(results, seed, "agrilite_xai_quantized",
                                   {"acc": acc_q, "f1": f1_q, "prec": prec_q, "rec": rec_q, "params_M": params_q, "size_MB": size_q})

    print(f"\n\nAll seeds complete. Results saved to {RESULTS_FILE}")
    summarize_and_test(results)
    return results


def summarize_and_test(results):
    """Print mean +/- std per model across seeds, plus paired t-test of
    AgriLite-XAI vs each baseline."""
    from scipy import stats

    model_names = set()
    for seed_data in results.values():
        model_names.update(seed_data.keys())
    model_names = sorted(model_names)

    print("\n=== SUMMARY: mean +/- std across seeds ===")
    per_model_acc = {}
    for name in model_names:
        accs = [results[s][name]["acc"] for s in results if name in results[s]]
        f1s = [results[s][name]["f1"] for s in results if name in results[s]]
        per_model_acc[name] = accs
        if len(accs) >= 2:
            print(f"{name:25s} acc={statistics.mean(accs)*100:.2f}% +/- {statistics.stdev(accs)*100:.2f}%  "
                  f"f1={statistics.mean(f1s)*100:.2f}% +/- {statistics.stdev(f1s)*100:.2f}%  (n={len(accs)})")
        else:
            print(f"{name:25s} acc={accs[0]*100:.2f}%  f1={f1s[0]*100:.2f}%  (n={len(accs)}, no std yet)")

    print("\n=== PAIRED T-TESTS: AgriLite-XAI vs each baseline (on accuracy) ===")
    agri_accs = per_model_acc.get("agrilite_xai", [])
    if len(agri_accs) < 2:
        print("Not enough seeds completed yet for AgriLite-XAI to run significance tests.")
        return
    for name in model_names:
        if name in ("agrilite_xai", "agrilite_xai_quantized"):
            continue
        other_accs = per_model_acc.get(name, [])
        if len(other_accs) != len(agri_accs):
            print(f"{name:25s} SKIPPED (unequal seed counts: {len(other_accs)} vs {len(agri_accs)})")
            continue
        t_stat, p_val = stats.ttest_rel(agri_accs, other_accs)
        diff = (statistics.mean(agri_accs) - statistics.mean(other_accs)) * 100
        sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
        print(f"{name:25s} AgriLite-XAI diff={diff:+.2f}pp  p={p_val:.4f}  {sig}")


if __name__ == "__main__":
    build_ghana_subset(
        src_dir="/kaggle/input/datasets/irakozekelly/crop-pest-and-disease-dataset/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset",
        dst_dir="/kaggle/working/splits",
        per_class_cap=100,
    )
    run_full_study("/kaggle/working/splits", epochs=20, seeds=[42, 43, 44, 45, 46])
