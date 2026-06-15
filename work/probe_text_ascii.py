#!/usr/bin/env python
"""Probe B: inject raw ASCII-art glyphs straight into TRIBE's Llama TEXT branch,
bypassing TTS, by overwriting the `context` field of a valid Word-events skeleton.

The HuggingFaceText extractor (contextualized=True) tokenizes each Word event's
`context` string (neuralset/extractors/text.py:220,405).  So we keep the skeleton's
timing but replace text/context/sentence with the ASCII cat, line by line, and keep
ONLY Word events (so the audio/video branches are removed and Delta isolates Llama).

Feasibility = (1) maps are non-degenerate, (2) base Llama != ascii-cat finetune.
"""
import pickle
from pathlib import Path
import numpy as np
import torch

from experiment_asciicat import (enable_fast_math, CACHE, MERGED, BASE_NAME,
                                  LH, RH, _destrieux, VISUAL, LANGUAGE, AUDITORY,
                                  roi_share, top_regions, cos)

OUT = Path(__file__).parent / "demo_out" / "phase2"
A = OUT / "assets"


def build_ascii_events(skeleton):
    """Overwrite a Word-events skeleton with the ASCII cat (line = token,
    context = growing block). Keep only Word events to isolate the text branch."""
    art = (A / "ascii_cat.txt").read_text().rstrip("\n")
    lines = [ln for ln in art.split("\n") if ln.strip() != ""]
    words = skeleton[skeleton["type"] == "Word"].reset_index(drop=True).copy()
    n = min(len(lines), len(words))
    ev = words.iloc[:n].copy()
    growing = []
    for i in range(n):
        growing.append(lines[i])
        ev.iloc[i, ev.columns.get_loc("text")] = lines[i]
        ev.iloc[i, ev.columns.get_loc("context")] = "\n".join(growing)
        if "sentence" in ev.columns:
            ev.iloc[i, ev.columns.get_loc("sentence")] = art
    return ev.reset_index(drop=True)


def whitelist_merged():
    import neuralset.extractors.base as _nb
    _wl = Path(_nb.__file__).with_name("data") / "huggingface-repos.txt"
    lines = _wl.read_text("utf8").splitlines()
    if MERGED not in lines:
        _wl.write_text("\n".join(lines + [MERGED]))
    from neuralset.extractors.text import HuggingFaceText
    if MERGED not in HuggingFaceText._REPOS:
        HuggingFaceText._REPOS.append(MERGED)


def predict_mean(model, events, tag):
    preds, segs = model.predict(events=events, verbose=False)
    print(f"  [{tag}] preds {preds.shape} over {len(segs)} segments")
    return preds.mean(0)


def load_model(model_name, cache_suffix):
    from tribev2 import TribeModel
    cf = CACHE + cache_suffix
    Path(cf).mkdir(parents=True, exist_ok=True)
    m = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=cf, device="auto",
        config_update={"data.text_feature.model_name": model_name})
    assert m.data.text_feature.model_name == model_name
    return m


def describe(name, m, labels, amap):
    print(f"\n[{name}] ||m||={np.linalg.norm(m):.3f} nonzero {np.count_nonzero(m)}/{m.size} "
          f"std={m.std():.4f} range[{m.min():.3f},{m.max():.3f}]")
    for roiname, roi in [("visual", VISUAL), ("language", LANGUAGE), ("auditory", AUDITORY)]:
        top, prior = roi_share(np.abs(m), labels, amap, roi)
        print(f"   {roiname:9s}: {top:.1%} (prior {prior:.1%}, {top/(prior+1e-9):.2f}x)")


def main():
    enable_fast_math()
    whitelist_merged()
    skel = pickle.loads((Path("demo_out/asciicat/events.pkl")).read_bytes())["cat"]
    ev = build_ascii_events(skel)
    print(f"injected ASCII text events: {len(ev)} Word rows")
    print("  sample context of last row:\n----\n" +
          ev.iloc[-1]["context"] + "\n----")

    import gc
    print("\n== base Llama-3.2-3B (ASCII as text) ==")
    m = load_model(BASE_NAME, "_txtprobe_base")
    base = predict_mean(m, ev, "base")
    del m; gc.collect(); torch.cuda.empty_cache()

    print("\n== ascii-cat finetune (ASCII as text) ==")
    m = load_model(MERGED, "_txtprobe_ft")
    ft = predict_mean(m, ev, "ft")
    del m; gc.collect(); torch.cuda.empty_cache()

    np.savez(OUT / "text_ascii_maps.npz", base=base, ft=ft)
    labels, amap = _destrieux()
    describe("base", base, labels, amap)
    describe("ft", ft, labels, amap)
    d = ft - base
    rel = np.linalg.norm(d) / (np.linalg.norm(base) + 1e-9)
    print(f"\n=== verdict ===")
    print(f"cosine(base, ft) = {cos(base, ft):.5f}   rel.change = {rel:.3%}")
    print("Feasible if maps non-degenerate AND base != ft (rel.change well above the")
    print("null-swap floor of ~0.004%).")
    describe("delta(ft-base)", d, labels, amap)


if __name__ == "__main__":
    main()
