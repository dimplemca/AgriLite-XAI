# AgriLite-XAI — Manuscript Package (v5: COMPLETE Statistically Validated Study)

## This is the real deal now
You ran the full study: 9 models × 5 seeds = 45 training runs, on GPU, with ImageNet-pretrained
weights, on the real Ghana field dataset. The paper is now built entirely around real,
statistically tested results — no placeholders, no single-run caveats, no "TBD" anywhere
in the results.

## The headline result (genuinely good news)
**AgriLite-XAI statistically significantly outperforms its own MobileNetV3-Small backbone:**
83.52% ± 0.92% vs 82.30% ± 1.79% (+1.21 percentage points, p = 0.041, Cohen's d = 1.33 —
a large effect size). It won in every single one of the 5 seeds, with no exceptions.

It's also statistically indistinguishable from four baselines that are all larger than it
(MobileNetV2, MobileNetV3-Large, EfficientNet-B1, ShuffleNetV2) — meaning comparable accuracy
at meaningfully smaller size. It trails the ResNet50 teacher by a modest, expected 2.42 points
despite being 19.8x smaller. Quantization costs no significant accuracy (p = 0.62). The one
baseline it doesn't match is EfficientNet-B0, by a small but real 0.55-point margin.

This is a legitimate, defensible, positive result — and importantly, it's honest: the paper
also reports that an earlier pilot phase (before pretrained weights were available) showed
the opposite result, and explains why, rather than hiding that history.

## Why this reversed so completely from the earlier pilot
The earlier no-internet pilot trained everything from random initialization — no ImageNet
pretrained weights. That's a much harder starting point than virtually anything in the
literature you're competing against. Once pretrained weights were restored, the entire
picture changed: 47.9% → 83.5% for AgriLite-XAI, and it went from losing to its own backbone
to significantly beating it. This confirms what I flagged as the single most likely fix,
several versions back — and it worked.

## Files
1. **AgriLite-XAI_Research_Paper.docx** — the complete, rebuilt manuscript
2. **full_study_results.png** — Figure 2, the real 5-seed results chart with error bars
3. **efficiency_comparison.png** — Figure 1, real architecture comparison
4. **full_study_raw_results.json** — your raw per-seed, per-model numbers (source of truth)
5. **full_study_pipeline.py** — the script that produced everything
6. Other pipeline variants (fast/no-internet versions) — kept for reference, no longer the
   primary path since the full study superseded them

## Honest remaining gaps (Section 7.1 in the paper — read this before submitting)
- **Single dataset, capped at 100 images/class** — not the dataset's full scale
- **Explainability module implemented but not evaluated** — Grad-CAM++/LayerCAM quality
  assessment is still unexecuted; this is now the single biggest remaining piece of the
  framework's three-part claim (accuracy ✓ done, deployability ✓ done, explainability —
  not yet)
- **No architecture-only ablation** — can't yet separate how much of AgriLite-XAI's win
  over MobileNetV3-Small comes from the attention module vs. from knowledge distillation
  itself, since both were changed together
- **Edge-hardware latency not measured** — Section 6.1's numbers are CPU/general-hardware,
  not actual Raspberry Pi/Jetson devices
- **Quantization only partial** (dynamic, Linear-layers only) — full static quantization
  would likely compress further

## On "ready for a Scopus-indexed journal" — now genuinely closer
This version is substantially stronger than any previous one. It has what reviewers actually
look for: a complete protocol, multiple baselines, statistical testing, honest limitations,
and a real, defensible, moderately-positive result rather than an overclaimed one. The most
valuable remaining piece before submission, if you have time/compute for one more push, would
be the Grad-CAM explainability evaluation (Section 4.6) — that's the one part of the paper's
three-part framing (efficient + accurate + explainable) that's still unvalidated.

## Before submitting
- Consider executing the explainability evaluation (Section 4.6) if time allows — biggest
  remaining gap
- Resolve preprint/grey-literature reference flags (refs [4], [6], [10], [17])
- Verify DOIs for refs [7] and [11] directly on publisher pages
- Author details, ethics statement, funding/conflict-of-interest declarations
- Run through your institution's plagiarism/similarity checker




