# AGENTS.md — TRIBE v2 research box (read this first)

Hard-won setup gotchas for working in `/home/ath/tribev2-research`. Every item below
is something that actually bit me and cost time. Skim this before running anything.

## 0. Always start a shell session like this
```bash
cd /home/ath/tribev2-research/work
source /home/ath/tribev2-research/env.sh     # ALWAYS the absolute path
```
`env.sh` activates `.venv`, sets `HF_HOME`, `TOKENIZERS_PARALLELISM`, and assumes the
gated-model `HF_TOKEN` (already in `~/.bashrc`). Nothing tribev2 imports correctly
without it.

## 1. The Bash tool's working directory drifts — use absolute paths
The shell cwd **persists across tool calls** and silently moved on me mid-session
(e.g. into `work/`), after which `source env.sh` failed with *"No such file or
directory"* and `python script.py` hit *"file not found"*. Symptoms look like broken
code but it's just cwd.
- Source env via the **absolute** path: `source /home/ath/tribev2-research/env.sh`.
- `cd /home/ath/tribev2-research/work` explicitly at the top of commands that need it.
- Don't trust that you're where you think you are.

## 2. `tribev2` is a namespace package — import only works under the venv, from the right dir
`import tribev2` gives `__file__ == None` and `dir(tribev2) == []`. `TribeModel` only
resolves via `from tribev2 import TribeModel`, and **only** when (a) the venv is active
and (b) cwd has **no `./tribev2` folder shadowing the package**. The real source lives at
`tribev2/tribev2/demo_utils.py`. Run scripts from `work/` (no shadowing there).
- Failure mode: `ImportError: cannot import name 'TribeModel' from 'tribev2' (unknown
  location)` — almost always means env.sh wasn't sourced (wrong cwd) or you're in a
  shadowing dir.

## 3. HF cache + gated model
- `~/.cache/huggingface` is **root-owned** → `PermissionError` if you forget env.sh
  (which redirects `HF_HOME=/home/ath/tribev2-research/hf_cache`).
- Text branch `meta-llama/Llama-3.2-3B` is **gated**; needs `HF_TOKEN` (already set). Do
  not log/echo the token.
- torch is **2.12.0+cu130** (Blackwell sm_121). The repo's `torch<2.7` pin is too old for
  this GPU; pins were relaxed in `tribev2/pyproject.toml`. Don't "fix" them back.
- `datasets` and `peft` are not always present — `pip install -q datasets peft` if needed.

## 4. Swapping the text feature-extractor (the Δ=0 trap)
TRIBE's text branch is a **frozen exca/pydantic** config, and **exca memoizes the
feature-cache uid**. Mutating the live extractor's `model_name` in place (even with
`object.__setattr__` + popping private attrs) silently serves the **cached base
features** → you get a beautiful, totally fake **Δ = exactly 0**.
- Correct way: build a **fresh** model pointed at the new weights from the start, with a
  **separate cache folder**:
  ```python
  TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE+"_ft",
      device="auto", config_update={"data.text_feature.model_name": MERGED})
  ```
- **Always run the null-swap control**: stock Llama through the same fresh-load code path
  must reproduce base to ~0.004% (cosine 1.00000). A bit-for-bit Δ=0 means the *cache*,
  not the model, is talking. (See `work/control_nullswap.py`.)

## 5. Local model dirs fail neuralset's validator
`neuralset` validates `model_name` against `extractors/data/huggingface-repos.txt` + a
hub `repo_exists()` call, and **rejects local paths**. The base model load also populates
an in-memory `HuggingFaceText._REPOS` ClassVar that then ignores the file. So you must
append the local path to **both** before the first extractor builds:
```python
import neuralset.extractors.base as _nb
wl = Path(_nb.__file__).with_name("data")/"huggingface-repos.txt"
wl.write_text(wl.read_text()+ "\n" + MERGED)          # file
from neuralset.extractors.text import HuggingFaceText
HuggingFaceText._REPOS.append(MERGED)                  # in-memory ClassVar
```

## 6. The published ASCII-cat adapter is GGUF-only
`pookie3000/Llama-3.2-3B-ascii-cats-lora-GGUF` won't load for hidden-state extraction.
Reproduce the LoRA in bf16 (PEFT) from `pookie3000/ascii-cats` and `merge_and_unload()`
to a standalone HF dir → loads identically to stock. See `work/train_asciicat_lora.py`.

## 7. Driving a single branch in isolation (phase-2 mechanics)
- **VIDEO (V-JEPA 2)**: `get_events_dataframe(video_path=...)` takes **video files only**
  (`.mp4/.avi/...`), so a still image must be a **static mp4**. It *also* extracts +
  transcribes the audio track → spurious `Audio`/`Word` events that **contaminate the map
  into temporal/auditory cortex**. Encode the mp4 with a silent `anullsrc` track (so
  `ExtractAudioFromVideo` doesn't error), then **keep only Video rows**:
  `ev = ev[ev.type=="Video"]` to isolate V-JEPA.
- **TEXT, bypassing TTS**: `HuggingFaceText` (contextualized=True) tokenizes each `Word`
  event's **`context`** field (`neuralset/extractors/text.py:220,405`). Inject arbitrary
  text by overwriting `text`/`context`/`sentence` on a valid Word-events skeleton
  (reuse `work/demo_out/asciicat/events.pkl`) and keep only Word rows. `context` must be
  non-empty or it raises.

## 8. Misc
- Cat images: Wikimedia upload often **429s** (rate limit); use `cataas.com/cat` for N
  distinct cats.
- TTS can't speak ASCII art (it pronounces punctuation) — that's *why* §7's text-injection
  path exists.
- A single **in-distribution** stimulus can badly over-promise: the ascii-text→visual
  "convergence" hint from one iconic cat vanished when scaled with realistic img2ascii.
  Scale with OOD inputs before believing a cross-modal effect.
- **Background-watcher self-match (cost me 3h).** A wait loop like
  `until ! pgrep -f experiment_x.py; do sleep 30; done` never exits: `pgrep -f` matches the
  **watcher's own command line**, which contains `experiment_x.py`. Wait on a **PID**
  instead — capture `$!` at launch and `until ! kill -0 "$PID"; do sleep 20; done`. Never
  `pgrep -f <name>` when `<name>` appears in the watcher command itself.

---
Deeper context lives in agent memory: `tribev2-env`, `tribev2-asciicat-experiment`,
`tribev2-phase2-crossmodal`, `tribev2-demo`. Key scripts in `work/`:
`train_asciicat_lora.py`, `experiment_asciicat.py`, `control_nullswap.py`,
`probe_visual.py`, `probe_text_ascii.py`, `experiment_phase2.py`, `make_showcase.py`.
