# TRIBE v2 on GB10 (DGX Spark / Blackwell)

Running [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) — Meta's tri-modal
(video + audio + text) foundation model that predicts human fMRI brain responses — on this
NVIDIA **GB10 Grace-Blackwell** machine (aarch64, sm_121, CUDA 13).

## Quick start

```bash
source env.sh                       # activates venv, sets HF_HOME, etc.
cd work
python run_tribev2.py               # smoke test on a synthesized clip
python run_tribev2.py --text story.txt
python demo_language_localizer.py   # in-silico neuroscience evaluation
```

Inference helper `work/run_tribev2.py` accepts `--video`, `--audio`, or `--text` and prints
the predicted cortical response shape `(n_timesteps, 20484)` on the fsaverage5 surface,
timing, and peak GPU memory.

## How it was set up (and the non-obvious bits)

| Component | Choice | Why |
|---|---|---|
| PyTorch | **2.12.0+cu130** (`download.pytorch.org/whl/cu130`) | The repo pins `torch<2.7`, too old for Blackwell **sm_121**. Pins relaxed in `tribev2/pyproject.toml`. |
| Env | `.venv` (system Python 3.12) | meets `>=3.11` |
| Speed | TF32 + `set_float32_matmul_precision("high")` | big speedup on the fp32 model, negligible accuracy cost |
| HF cache | `HF_HOME=$TRIBE_ROOT/hf_cache` | `~/.cache/huggingface` is root-owned on this box |
| Transcription | **NVIDIA Parakeet on GPU** (`transformers.ParakeetForCTC`) | replaced `uvx`/whisperx, which is CPU-only on aarch64 (ctranslate2 has no ARM CUDA build). See `tribev2/parakeet_asr.py`. |
| Run dir | `work/` (no `./tribev2` folder) | avoids the local repo dir shadowing the installed `tribev2` package |

Feature extractors pulled by the checkpoint: video = `vjepa2-vitg` + `dinov2-large`,
audio = `w2v-bert-2.0` (open), text = **`meta-llama/Llama-3.2-3B`** (gated — `HF_TOKEN`
must have accepted the license).

### Parakeet transcription

`transformers`' CTC pipeline word-timestamp postprocess is buggy for Parakeet's tokenizer,
so `parakeet_asr.py` runs the model directly and derives word timestamps from a greedy CTC
alignment. Override the model with `TRIBE_PARAKEET_MODEL` (default `nvidia/parakeet-ctc-1.1b`).

## Demonstration: in-silico neuroscience (`work/demo_language_localizer.py`)

Replicates the method of the paper's Fig. 5: feed controlled linguistic stimuli, read out
the model's *predicted* fsaverage5 responses, and analyse them like a localizer fMRI
experiment. Outputs to `work/demo_out/`.

**Result — heard speech drives auditory / superior-temporal cortex** (zero-shot):

- Auditory cortex share of the top-5% activated vertices: **47.5%** (vs 7.2% prior) → **6.6× enrichment**
- Strongest regions: transverse-temporal sulcus, **Heschl's gyrus (primary auditory)**,
  lateral superior temporal gyrus, planum temporale, posterior lateral fissure.

i.e. with no real brain data, the model recovers the textbook localization of speech to
primary + secondary auditory cortex. See `demo_out/speech_evoked_response.png` and
`evaluation_report.md`.

(The subtler `sentences > word-lists` syntactic contrast is included but weak here: the
TTS audio is near-identical across conditions and a single 8-TR block has low power.)

## Verified inference

- video clip → `(8, 20484)`, ~10.9 GB peak GPU
- text input → `(8, 20484)`, ~9.3 GB peak GPU, full tri-modal path (Parakeet → Llama)
