# Does an ASCII-cat LoRA finetune change TRIBE v2's predicted brain maps?

**Question.** TRIBE v2's text branch uses `meta-llama/Llama-3.2-3B` as a *frozen
feature extractor* (reads 20 hidden-state layers, group-mean pooled, into the brain
encoder). We swap in an ASCII-cat LoRA finetune of that same Llama
(reproduced in bf16 from the real dataset `pookie3000/ascii-cats`, 201 cats, 10
epochs — the published adapter is GGUF-only and won't load for hidden states) and
ask whether — and where — TRIBE's predictions change, and whether the change is
*cat-specific*.

**Design.** Text-only. Minimal-pair stimuli: identical 10-sentence templates with
the creature word swapped (cat / dog / horse) + a neutral weather passage. Events
(text -> TTS -> Parakeet -> words) are computed ONCE and reused across extractors, so
the audio (w2v-bert) branch is common-mode and Delta isolates the Llama text features.
Finetune predictions use a freshly-built TRIBE config + separate feature cache.

## Verdict

**1. The finetune is real but the brain-map change is NOT cat-specific.**

| category | cosine(base,ft) | \|\|Delta\|\| | rel. change |
|---|---|---|---|
| cat | 0.9966 | 1.90 | 8.40% |
| dog | 0.9964 | 2.06 | 8.72% |
| horse | 0.9967 | 1.77 | 8.38% |
| neutral | 0.9957 | 1.80 | 9.43% |

The LoRA shifts every category by ~8–9%. Cat is not the largest — **dog moves more.**
A cat-overfit adapter induces a broad, roughly *uniform* representational drift, not a
"cat fingerprint." (Hypothesis "cat moves most" — **refuted.**)

**2. The drift lands in the LANGUAGE network, and avoids visual cortex.**
Raw cat delta, top-5% most-changed vertices: **62.9% language (5.4x enrichment),
0.0% visual, ~0 auditory.** Changing the LLM text features perturbs the
language-responsive cortex the text branch drives — not visual areas. (Hypothesis
"teaching Llama to draw cats pushes cat-talk toward visual cortex" — **refuted**;
visual cortex is if anything *de*-enriched.)

**3. Rigor control (validates the pipeline).** Running STOCK Llama through the exact
finetune code path (fresh load + separate cache) reproduces the base maps to
**Delta = 0.004% (cosine 1.00000)**. So the 8% is genuinely the LoRA, not a dtype /
cache / code-path artifact. (An earlier run that reported Delta = 0 exactly was a
feature-cache reuse bug, now fixed and guarded.)

**4. The cat-specific contrast (cat − dog) is small and noisy.** Isolating the part
of the shift unique to "cat" vs "dog" (a low-power 2nd-order difference, deltas ~0.005)
leans into **auditory / superior-temporal cortex** (planum temporale, Heschl's gyrus,
lateral STG/STS; auditory enrichment 4.3x) with a whisper of occipito-temporal
(lingual, lateral occipitotemporal). Visual is still de-enriched (0.66x). Suggestive
only — do not over-read.

## Interpretation

A tiny LoRA (24M params, r=16; 201 examples) overfit to *generating* ASCII cats mostly
shifts the model's overall activation statistics — a broad stylistic/distributional
drift — rather than rewriting the *semantic content* of "cat" for ordinary
natural-language input. Because TRIBE reads representations of the input (not the
model's generations), that drift surfaces as a uniform ~8% perturbation concentrated
where the text branch has the most cortical leverage: the language network. The
headline takeaway:

> **TRIBE's predicted brain maps are sensitive to the feature-extractor's distribution
> but robust to its behavioral/semantic specialization.** A cat-art finetune moves cat,
> dog, horse, and neutral speech almost identically, and into language — not visual —
> cortex.

## Caveats
- The cat−dog contrast is low-power (single block, ~36 TRs/condition); its localization
  is suggestive, not established.
- Stimuli reach Llama as TTS-then-transcribed words; we did NOT push raw ASCII art
  through the text branch (TTS can't pronounce it). The maximal-effect probe — inject
  ASCII glyphs straight into the Llama extractor as word-events, bypassing TTS — is the
  natural follow-up.
- Reproduced LoRA hyperparameters are sensible defaults, not the original's exact recipe;
  the *direction* of the finetune is faithful (same dataset), magnitudes may differ.

## Files
- `report.md`, `metrics.json` — auto-generated metrics
- `base_maps.npz`, `ft_maps.npz`, `nullctl_maps.npz`, `condition_maps.npz` — raw maps
- `delta_cat.png`, `cat_specific.png`, `cat_minus_horse.png` — cortical surface maps
- scripts: `../train_asciicat_lora.py`, `../experiment_asciicat.py`, `../control_nullswap.py`
