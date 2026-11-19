"""
AgriLite ablation study (Step 1)
================================
Isolates what produces the gain of AgriLite over stock MobileNetV3-Small.
AgriLite differs from the stock backbone in THREE things at once:
    (a) narrower classifier head  (576->256 instead of 576->1024)
    (b) LightAttention block
    (c) knowledge distillation from the ResNet50 teacher

This script trains all six combinations on the SAME split and the SAME seeds
as the main study, so the numbers are directly comparable with Table 2.

    A_stock_ce            stock MobileNetV3-Small, cross-entropy   (= existing baseline)
    B_stock_kd            stock MobileNetV3-Small + KD             -> effect of KD alone
    C_head256_ce          256-unit head, no attention, CE          -> effect of narrow head alone
    D_head256_kd          256-unit head, no attention, + KD
    E_att_head256_ce      attention + 256 head, CE (no KD)         -> effect of attention (no KD)
    F_full_att_head256_kd attention + 256 head + KD                = AgriLite (full model)

SETUP (Kaggle notebook)
    1. Rename "full_study_pipeline (1).py" to  full_study_pipeline.py  and put it
       in the same folder as this file (or paste both into the notebook working dir).
    2. Make sure /kaggle/working/splits already exists (built by build_ghana_subset
       in the main study). Do NOT rebuild it with a different seed, or results will
       not be comparable with Table 2.
    3. Run:   python ablation_study.py
       or in a notebook cell:
            from ablation_study import run_ablation
            run_ablation("/kaggle/working/splits", epochs=20, seeds=[42,43,44,45,46])

Resumable: results are written after every model; trained teachers are saved
to disk so a restart does not retrain them. Expected time on a Kaggle T4:
about 2 hours for 5 seeds (1 teacher + 6 small models per seed).
"""

import copy
import json
import os
import statistics

import torch
import torch.nn as nn
from torchvision import models

from full_study_pipeline import (
    PRETRAINED, LightAttention, build_baseline, build_teacher,
    count_params_mb, evaluate, get_dataloaders, set_seed, train_model,
)

ABL_FILE = "/kaggle/working/ablation_results.json"
TEACHER_DIR = "/kaggle/working/teachers"


# ----------------------------------------------------------------------
# Configurable student: head width and optional attention
# ----------------------------------------------------------------------
class SmallNet(nn.Module):
    """MobileNetV3-Small features + optional LightAttention + 2-layer head.
    With attention=True and head_dim=256 this is architecturally identical to
    AgriLiteStudent in the main pipeline."""

    def __init__(self, num_classes, head_dim=256, attention=False, pretrained=PRETRAINED):
        super().__init__()
        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.attention = LightAttention(576) if attention else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, head_dim), nn.Hardswish(inplace=True),
            nn.Dropout(0.2), nn.Linear(head_dim, num_classes))

    def forward(self, x):
        x = self.attention(self.features(x))
        return self.classifier(torch.flatten(self.avgpool(x), 1))


VARIANTS = {
    # name: (builder(num_classes) -> model, use_kd)
    "A_stock_ce":            (lambda n: build_baseline("mobilenet_v3_small", n), False),
    "B_stock_kd":            (lambda n: build_baseline("mobilenet_v3_small", n), True),
    "C_head256_ce":          (lambda n: SmallNet(n, 256, attention=False), False),
    "D_head256_kd":          (lambda n: SmallNet(n, 256, attention=False), True),
    "E_att_head256_ce":      (lambda n: SmallNet(n, 256, attention=True), False),
    "F_full_att_head256_kd": (lambda n: SmallNet(n, 256, attention=True), True),
}

# (label, variant_1, variant_2)  -> reports mean(variant_1 - variant_2)
CONTRASTS = [
    ("KD effect, stock head           (B - A)", "B_stock_kd", "A_stock_ce"),
    ("Narrow head effect, no KD       (C - A)", "C_head256_ce", "A_stock_ce"),
    ("Attention effect, no KD         (E - C)", "E_att_head256_ce", "C_head256_ce"),
    ("KD effect, narrow head          (D - C)", "D_head256_kd", "C_head256_ce"),
    ("KD effect, with attention       (F - E)", "F_full_att_head256_kd", "E_att_head256_ce"),
    ("Attention effect, with KD       (F - D)", "F_full_att_head256_kd", "D_head256_kd"),
    ("Full AgriLite vs stock          (F - A)", "F_full_att_head256_kd", "A_stock_ce"),
]


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def load_results():
    if os.path.exists(ABL_FILE):
        with open(ABL_FILE) as f:
            return json.load(f)
    return {}


def save_result(results, seed, name, metrics):
    results.setdefault(f"seed{seed}", {})[name] = metrics
    with open(ABL_FILE, "w") as f:
        json.dump(results, f, indent=2)
    return results


def get_teacher(seed, num_classes, loaders, device, epochs):
    """Train the ResNet50 teacher once per seed and cache it on disk."""
    os.makedirs(TEACHER_DIR, exist_ok=True)
    path = f"{TEACHER_DIR}/teacher_seed{seed}.pt"
    teacher = build_teacher(num_classes)
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[seed {seed}] loaded cached teacher")
    else:
        train_loader, val_loader, _ = loaders
        teacher, _ = train_model(teacher, train_loader, val_loader, device,
                                 epochs=epochs, log_prefix=f"s{seed}-teacher")
        torch.save(teacher.state_dict(), path)
    return teacher.to(device).eval()


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run_ablation(data_root, epochs=20, batch_size=16, seeds=(42, 43, 44, 45, 46)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Seeds: {list(seeds)} | Variants: {list(VARIANTS)}")
    results = load_results()

    for seed in seeds:
        pending = [v for v in VARIANTS if v not in results.get(f"seed{seed}", {})]
        if not pending:
            print(f"[seed {seed}] all variants done, skipping")
            continue
        print(f"\n{'=' * 60}\nSEED {seed}  (pending: {pending})\n{'=' * 60}")
        set_seed(seed)
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            data_root, batch_size=batch_size)
        num_classes = len(classes)
        teacher = None
        if any(VARIANTS[v][1] for v in pending):
            teacher = get_teacher(seed, num_classes, (train_loader, val_loader, test_loader),
                                  device, epochs)

        for name in pending:
            build, use_kd = VARIANTS[name]
            set_seed(seed)  # same head initialisation across runs of the same seed
            model = build(num_classes)
            model, _ = train_model(model, train_loader, val_loader, device, epochs=epochs,
                                   teacher=teacher if use_kd else None,
                                   log_prefix=f"s{seed}-{name}")
            acc, f1, prec, rec = evaluate(model, test_loader, device)
            params, size = count_params_mb(model)
            results = save_result(results, seed, name, {
                "acc": acc, "f1": f1, "prec": prec, "rec": rec,
                "params_M": params, "size_MB": size, "kd": use_kd})
            print(f"[seed {seed}] {name}: acc={acc:.4f} f1={f1:.4f} params={params:.3f}M")

    summarize(results)
    return results


# ----------------------------------------------------------------------
# Statistics: mean +/- SD, paired differences with 95% CI, Holm correction
# ----------------------------------------------------------------------
def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def summarize(results):
    from scipy import stats
    seeds = sorted(results)
    print("\n=== ABLATION: test accuracy / macro-F1, mean +/- SD over seeds ===")
    acc = {}
    for v in VARIANTS:
        a = [results[s][v]["acc"] for s in seeds if v in results[s]]
        f = [results[s][v]["f1"] for s in seeds if v in results[s]]
        acc[v] = a
        if len(a) >= 2:
            p = results[seeds[0]][v]["params_M"]
            print(f"{v:24s} acc={statistics.mean(a)*100:6.2f} +/- {statistics.stdev(a)*100:4.2f}  "
                  f"f1={statistics.mean(f)*100:6.2f} +/- {statistics.stdev(f)*100:4.2f}  "
                  f"params={p:.3f}M  n={len(a)}")

    print("\n=== PAIRED CONTRASTS (accuracy, percentage points) ===")
    rows, pvals = [], []
    for label, v1, v2 in CONTRASTS:
        a1, a2 = acc.get(v1, []), acc.get(v2, [])
        if len(a1) != len(a2) or len(a1) < 3:
            continue
        diffs = [(x - y) * 100 for x, y in zip(a1, a2)]
        n = len(diffs)
        m, sd = statistics.mean(diffs), statistics.stdev(diffs)
        half = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
        t, p = stats.ttest_rel(a1, a2)
        d = m / sd if sd > 0 else float("nan")
        wins = sum(x > 0 for x in diffs)
        rows.append((label, m, m - half, m + half, p, d, wins, n))
        pvals.append(p)
    adj = holm(pvals) if pvals else []
    print(f"{'contrast':42s} {'diff':>7s} {'95% CI':>17s} {'p':>7s} {'p_Holm':>7s} {'d':>6s} {'wins':>5s}")
    for (label, m, lo, hi, p, d, w, n), pa in zip(rows, adj):
        print(f"{label:42s} {m:+7.2f} [{lo:+6.2f},{hi:+6.2f}] {p:7.3f} {pa:7.3f} {d:+6.2f} {w:>2d}/{n}")
    print("\nHow to read it: a contrast whose 95% CI excludes 0 AND p_Holm < 0.05 is a "
          "supported effect; otherwise report it as 'not detected'.")


if __name__ == "__main__":
    run_ablation("/kaggle/working/splits", epochs=20, seeds=[42, 43, 44, 45, 46])
