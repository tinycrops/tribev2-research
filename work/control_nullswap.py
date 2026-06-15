#!/usr/bin/env python
"""Rigor control: re-run the *finetune code path* (fresh TribeModel load + separate
cache folder) but with the STOCK Llama-3.2-3B instead of the ascii-cat merge.

If this null-swap yields Delta ~ 0 vs the original cached base maps, then the ~8%
drift seen for the real finetune is genuinely the LoRA, not an artifact of the
second code path (dtype, fresh load, or separate cache).
"""
import pickle
from pathlib import Path
import numpy as np
import torch

from experiment_asciicat import (OUT, CACHE, BASE_NAME, enable_fast_math,
                                  predict_mean, cos)


def main():
    enable_fast_math()
    from tribev2 import TribeModel
    events = pickle.loads((OUT / "events.pkl").read_bytes())
    z = np.load(OUT / "base_maps.npz")
    base = {c: z[c] for c in z.files}

    CACHE_NULL = CACHE + "_nullctl"
    Path(CACHE_NULL).mkdir(parents=True, exist_ok=True)
    print("loading fresh TRIBE with STOCK Llama via the finetune code path ...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=CACHE_NULL, device="auto",
        config_update={"data.text_feature.model_name": BASE_NAME})

    null = {c: predict_mean(model, events[c], f"nullctl/{c}") for c in base}
    np.savez(OUT / "nullctl_maps.npz", **null)

    print("\n# Null-swap control (stock Llama, finetune code path) vs cached base")
    print("| category | cosine | ||Delta|| | rel.change |")
    print("|---|---|---|---|")
    for c in base:
        d = null[c] - base[c]
        rel = np.linalg.norm(d) / (np.linalg.norm(base[c]) + 1e-9)
        print(f"| {c} | {cos(base[c], null[c]):.5f} | {np.linalg.norm(d):.4f} | {rel:.3%} |")
    print("\n(Delta ~ 0 here => the ~8% finetune drift is real, not a code-path artifact.)")


if __name__ == "__main__":
    main()
