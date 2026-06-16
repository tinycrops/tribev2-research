# Phase 2 — Cross-modal ASCII / image experiments on TRIBE v2

How does the ASCII-cat LoRA (a Llama-3.2-3B text finetune) change TRIBE v2's predicted
brain maps when ASCII is presented as an **image** (V-JEPA video branch) vs as **text**
(Llama branch)? TRIBE's three encoders here: text = Llama-3.2-3B, audio = w2v-bert-2.0,
video = V-JEPA 2 (`vjepa2-vitg-fpc64-256`). The LoRA lives in the **text** branch only.

All maps are fsaverage5 (20484 vertices); "visual enrichment" = share of the top-5%
most-active vertices falling in the Destrieux visual ROI, over its prior (>1× = biased
toward visual cortex). Branches are driven in isolation (video-only vs word-only events)
unless stated. Scripts: `probe_visual.py`, `probe_text_ascii.py`, `experiment_phase2.py`,
`experiment_substantiality.py`, `experiment_combined.py`.

---

## Result 1 — Visual branch (V-JEPA): a real cat lands in visual cortex; an ASCII cat only partly
`probe_visual.py` · video-only events.

| stimulus | visual | language | cos to real cat |
|---|---|---|---|
| real cat photo | **5.48×** | 0.29× | — (anchor) |
| ASCII-cat bitmap | 3.64× | 2.76× | **0.435** |

A real photo (even a static frame) reads cleanly into visual cortex. An ASCII-cat bitmap
is half-visual / half-language (it's glyphs), and only **cos 0.44** similar to the real
cat — an ASCII cat does not *look* like a cat to the brain model. That gap is the
optimization headroom for "make ASCII a model reads as a cat."

## Result 2 — Text branch probe (n=1 iconic cat): finetune pushes ASCII-text toward vision
`probe_text_ascii.py` · word-only events, ASCII injected via the `context` field (no TTS).

One clean hand-drawn ASCII cat, base Llama vs finetune: rel-change **34%** (vs ~8% for
ordinary speech), and visual enrichment **0.85× → 2.05×**. The finetune moved the map
toward visual cortex — the opposite of the phase-1 speech result (which avoided visual).
**This was a single, in-distribution stimulus — a hint, not a result.**

## Result 3 — Scaled with img2ascii (n=6 photos): the hint does NOT transfer to photo-conversions
`experiment_phase2.py` · 6 real cats → real image (anchor) + full/reduced img2ascii.

| metric (mean, n=6) | full-res | reduced-res |
|---|---|---|
| cos(anchor, ascii **bitmap**) | 0.55 | 0.43 |
| bitmap visual enrichment | 3.98× | 2.53× |
| cos(anchor, ascii **text**) | 0.13 | 0.08 |
| finetune convergence gain | −0.006 (3/6) | −0.080 (1/6) |

**Visual realism ladder confirmed:** higher-resolution img2ascii bitmaps read as more
cat-like to V-JEPA (0.55 > 0.43). **But** ASCII-as-text stays near-orthogonal to the
image anchor (cos ≈ 0.1) and the finetune does not pull it closer. *At the time I called
the convergence "refuted" — that was too strong.* This run changed the **stimulus
distribution** (hand-drawn line-art → img2ascii brightness-ramp mush), not just the sample
size, so it could not actually test Result 2's claim. See Result 4.

## Result 4 — Substantiality test (n=86 line-art ASCII, apehex): SUBSTANTIAL but GENERIC
`experiment_substantiality.py` · `apehex/ascii-art` animals shard, line-drawn ASCII as
text, base vs finetune. Cats vs Dogs vs Horses = the phase-1 minimal pair, at scale.

| stimulus | n | visual: base → ft | shift (ft−base) | % positive | rel-change |
|---|---|---|---|---|---|
| **cats** | 40 | 0.92 → **2.15×** | +1.23 ± 0.98 | **100%** | 36.8% |
| **dogs** | 30 | 1.22 → **2.47×** | +1.24 ± 1.05 | **100%** | 27.1% |
| **horses** | 16 | 0.89 → **1.57×** | +0.68 ± 0.70 | 81% | 36.0% |

cat-specificity contrast: **+0.18** (cat +1.23 vs control +1.05), Mann-Whitney **p = 0.30**.

**The probe effect is real** — across 40 distinct iconic ASCII cats the finetune pushes
the map toward visual cortex **every single time** (100%), 0.92× → 2.15×, crossing from
"not visual" to "clearly visual." Result 2 was not noise; my "refuted" was wrong.

**But it is not about cats.** Dogs shift +1.24 — identical to cats — also 100% positive.
The cat-vs-control contrast is tiny and non-significant. The finetune pushes *any*
line-drawn ASCII animal toward visual cortex.

## Result 5 — Fusion: the same ASCII through BOTH branches at once (n=20 cats)
`experiment_combined.py` · one events frame with the ASCII bitmap (V-JEPA) + the same
ASCII as text (Llama), time-aligned so the brain model fuses them. Every prior test
isolated a single branch; this is the first fused stimulus.

| quantity | value | reading |
|---|---|---|
| finetune ‖Δ‖/‖base‖: text-only → fused | 38% → **16%** | visual context **dampens** the finetune effect (halved) |
| visual enrichment: text-only / fused-base / visual-only | 0.90× / **1.68×** / 2.96× | fused sits between text & image — the bitmap pulls it visual |
| visual enrichment: fused-base → fused-ft | 1.68× → 1.67× | finetune adds ~nothing visual on top of the bitmap |
| cos(fused-base → visual / text / blend) | 0.74 / 0.55 / **0.88** | **fusion ≈ additive**, visual-dominant |
| fusion convergence gain (cos→visual: ft − base) | **+0.056 ± 0.058, 20/20** | finetune still nudges fused map toward visual — small, but every cat |

A fused ASCII stimulus reads as a weighted blend of its image and text maps (cos 0.88 to
the normalized sum), leaning toward the image (cos 0.74 visual vs 0.55 text; enrichment
1.68×). The real bitmap does the heavy lifting on visual localization. The text finetune's
effect is **diluted** by the visual input (38%→16%) and adds essentially no extra
visual-ness on top of the bitmap (1.68×≈1.67×) — yet it still nudges the fused map toward
the visual rep in **every one of the 20 cats** (+0.056). The style-detector signal survives
fusion, but the actual image dominates: the most visual you make an ASCII cat is by showing
the picture, not by the finetune.

---

## Unifying interpretation: the LoRA is a line-art-ASCII **style** detector

| condition | ASCII-art style present? | finetune effect |
|---|---|---|
| Phase 1: TTS speech | no | ~8%, stays in *language*, avoids visual |
| Result 3: img2ascii (gradients) | no (OOD format) | null convergence |
| Result 2/4: hand-drawn line-art | **yes** | big, 100%-consistent push to *visual* |

The finetune did not teach Llama "cat." It taught Llama "**this glyph-block is a
picture**," and that picture-ness surfaces as visual cortex. When the input matches the
iconic line-art format the LoRA trained on, the representation routes toward visual cortex
regardless of subject (cat = dog). When it doesn't (speech, or brightness-ramp img2ascii),
nothing visual happens. This is exactly the phase-1 headline — **TRIBE is sensitive to the
extractor's distribution/style, not its semantics** — now demonstrated at the
representational level.

## Methodological note
A single in-distribution probe (Result 2) badly over-promised, and a distribution-shifted
scale-up (Result 3) badly under-claimed. Only scaling *within the probe's own regime* with
a matched control (Result 4) gave the honest answer: real effect, wrong reason. Always
scale in-regime, with a control, before believing or dismissing a cross-modal effect.

## Artifacts
`subst_summary.json` / `subst_rows.json` (Result 4), `scaled_summary.json` (Result 3),
`visual_maps.npz` / `text_ascii_maps.npz` (Results 1–2), `showcase.html` (self-contained
visual report). See repo-root `AGENTS.md` for setup/branch-isolation gotchas.
