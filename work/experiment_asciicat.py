#!/usr/bin/env python
"""Experiment: does swapping stock Llama-3.2-3B for an ASCII-cat LoRA finetune
change TRIBE v2's predicted cortical responses, and is the change cat-specific?

Design (text-only; audio branch is common-mode and cancels in Delta):
  category {cat, horse, dog, neutral} x extractor {base Llama, ascii-cat finetune}

Because TRIBE builds events via text->TTS->Parakeet (LLM-independent), we compute
events ONCE per category and reuse them for both extractors, so the ONLY variable
is the Llama text-feature weights.

Headline = difference-in-differences:  ||Delta_cat|| vs ||Delta_horse/dog/neutral||
and where the cat-specific delta localizes (visual vs language vs auditory ROIs).
"""
import os
import time
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).parent / "demo_out" / "asciicat"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = "/home/ath/tribev2-research/cache"
MERGED = str(Path(__file__).parent / "models" / "llama32-3b-asciicat-merged")
BASE_NAME = "meta-llama/Llama-3.2-3B"
LH = slice(0, 10242)
RH = slice(10242, 20484)

# ----- minimal-pair stimuli: identical templates, swap the creature ----------
TEMPLATES = [
    "The {x} stood quietly in the center of the ring.",
    "Everyone at the show wanted to see the {x} up close.",
    "A young girl reached out to touch the {x}.",
    "The judge walked slowly around the {x} and made some notes.",
    "Photographers leaned in to capture the {x} in good light.",
    "The owner gently brushed the {x} before the next round.",
    "People in the crowd pointed at the {x} and smiled.",
    "The {x} turned toward the sound of the applause.",
    "After the show, the {x} rested quietly in the shade.",
    "Many visitors agreed the {x} was the finest one there.",
]
CREATURES = {"cat": "cat", "horse": "horse", "dog": "dog"}
NEUTRAL = (
    "The morning began with a thick fog over the harbor. "
    "By noon the clouds had cleared and the streets grew busy with traffic. "
    "A delivery truck idled at the corner while the lights changed. "
    "Commuters hurried past the bakery on their way to the station. "
    "The afternoon stayed warm until a light rain returned at dusk."
)


def stimuli():
    s = {k: " ".join(t.format(x=v) for t in TEMPLATES) for k, v in CREATURES.items()}
    s["neutral"] = NEUTRAL
    return s


def enable_fast_math():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def write_text(text, tag):
    f = tempfile.NamedTemporaryFile("w", suffix=f"_{tag}.txt", dir=CACHE, delete=False)
    f.write(text)
    f.close()
    return f.name


def set_extractor_model(model, model_name):
    """Point TRIBE's text (Llama) extractor at a new weights dir and force reload.

    The extractor is a frozen exca/pydantic config, so we bypass the freeze with
    object.__setattr__. The `model` property guards on hasattr(self,"_model"); the
    cached model/tokenizer live in __pydantic_private__, so we pop them to force a
    fresh load of the new weights.
    """
    ext = model.data.text_feature
    object.__setattr__(ext, "model_name", model_name)
    priv = getattr(ext, "__pydantic_private__", None)
    if isinstance(priv, dict):
        for attr in ("_model", "_tokenizer", "_pad_id"):
            priv.pop(attr, None)
    # free VRAM from the previous extractor model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return ext


def predict_mean(model, events, tag):
    t0 = time.time()
    preds, _ = model.predict(events=events, verbose=False)
    print(f"  [{tag}] preds {preds.shape} in {time.time()-t0:.1f}s")
    return preds.mean(0)


# ----------------------------------------------------------------- atlas / ROIs
def _destrieux():
    from nilearn import datasets
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    amap = np.concatenate([atlas["map_left"], atlas["map_right"]])
    return labels, amap


# Destrieux ROI groups
VISUAL = ("G_oc-temp_lat-fusifor", "G_oc-temp_med-Lingual", "G_cuneus",
          "G_occipital_middle", "G_occipital_sup", "Pole_occipital",
          "S_oc_middle_and_Lunatus", "S_oc_sup_and_transversal",
          "S_collat_transv_post", "G_and_S_occipital_inf", "S_calcarine",
          "G_oc-temp_med-Parahip")
LANGUAGE = ("G_temp_sup-Lateral", "S_temporal_sup", "G_front_inf-Triangul",
            "G_front_inf-Opercular", "G_pariet_inf-Angular", "G_temporal_middle")
AUDITORY = ("G_temp_sup-G_T_transv", "G_temp_sup-Plan_tempo", "S_temporal_transverse")


def roi_share(stat, labels, amap, roi, q=0.95):
    ids = [labels.index(r) for r in roi if r in labels]
    mask = np.isin(amap, ids)
    thr = np.quantile(stat, q)
    top = stat >= thr
    return float(mask[top].mean()), float(mask.mean())


def top_regions(stat, labels, amap, n=10, signed=True):
    rows = []
    for idx in np.unique(amap):
        name = labels[idx]
        if name in ("Unknown", "Medial_wall"):
            continue
        m = amap == idx
        rows.append((float(stat[m].mean()), name, int(m.sum())))
    rows.sort(reverse=True, key=lambda r: (r[0] if signed else abs(r[0])))
    return rows[:n]


def plot_map(stat, fname, title):
    from nilearn import datasets, plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    vmax = np.abs(stat).max()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             subplot_kw={"projection": "3d"})
    for ax, hemi, mesh, bg, data in [
        (axes[0], "left", fs.infl_left, fs.sulc_left, stat[LH]),
        (axes[1], "right", fs.infl_right, fs.sulc_right, stat[RH]),
    ]:
        plotting.plot_surf_stat_map(
            mesh, data, hemi=hemi, view="lateral", bg_map=bg, axes=ax,
            colorbar=(hemi == "right"), cmap="cold_hot", vmax=vmax,
            threshold=vmax * 0.2, title=f"{hemi}")
    fig.suptitle(title, fontsize=12)
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", OUT / fname)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    enable_fast_math()
    from tribev2 import TribeModel
    stims = stimuli()
    import pickle, gc
    ev_path = OUT / "events.pkl"
    base_path = OUT / "base_maps.npz"
    ft_path = OUT / "ft_maps.npz"

    # Whitelist the local merged model BEFORE any extractor validates (the base
    # load below populates neuralset's in-memory _REPOS, which then won't re-read
    # the file). Patch both the file and the in-memory ClassVar.
    import neuralset.extractors.base as _nb
    _wl = Path(_nb.__file__).with_name("data") / "huggingface-repos.txt"
    _lines = _wl.read_text("utf8").splitlines()
    if MERGED not in _lines:
        _wl.write_text("\n".join(_lines + [MERGED]))
    from neuralset.extractors.text import HuggingFaceText
    if MERGED not in HuggingFaceText._REPOS:
        HuggingFaceText._REPOS.append(MERGED)

    # Base model only needed if events or base maps are missing.
    model = None
    if not (ev_path.exists() and base_path.exists()):
        print("loading TRIBE v2 (base extractor) ...")
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE,
                                           device="auto")

    # Phase 1: events once per category (LLM-independent: TTS+Parakeet). Cached.
    if ev_path.exists():
        events = pickle.loads(ev_path.read_bytes())
        print(f"loaded cached events for {list(events)}")
    else:
        events = {}
        for cat, text in stims.items():
            path = write_text(text, cat)
            events[cat] = model.get_events_dataframe(text_path=path)
            print(f"events[{cat}]: {len(events[cat])} rows")
        ev_path.write_bytes(pickle.dumps(events))

    # Phase 2: base extractor predictions. Cached.
    if base_path.exists():
        z = np.load(base_path)
        base = {c: z[c] for c in stims}
        print(f"loaded cached base maps for {list(base)}")
    else:
        print("\n== base Llama-3.2-3B ==")
        set_extractor_model(model, BASE_NAME)
        base = {c: predict_mean(model, events[c], f"base/{c}") for c in stims}
        np.savez(base_path, **base)

    # Phase 3: finetuned (ascii-cat) extractor predictions, same events.
    # We build a FRESH TribeModel whose text extractor points at the merged model
    # from the start (its cache-uid is computed fresh -> no collision with base),
    # and a separate cache folder to be doubly safe against feature-cache reuse.
    if ft_path.exists():
        z = np.load(ft_path)
        ft = {c: z[c] for c in stims}
        print(f"loaded cached ft maps for {list(ft)}")
    else:
        print("\n== ascii-cat finetuned Llama (fresh model, separate cache) ==")
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        CACHE_FT = CACHE + "_asciicat"
        Path(CACHE_FT).mkdir(parents=True, exist_ok=True)
        model = TribeModel.from_pretrained(
            "facebook/tribev2", cache_folder=CACHE_FT, device="auto",
            config_update={"data.text_feature.model_name": MERGED})
        assert model.data.text_feature.model_name == MERGED, \
            f"override failed: {model.data.text_feature.model_name}"
        ft = {c: predict_mean(model, events[c], f"ft/{c}") for c in stims}
        np.savez(ft_path, **ft)

    np.savez(OUT / "condition_maps.npz",
             **{f"base_{c}": base[c] for c in stims},
             **{f"ft_{c}": ft[c] for c in stims})

    # ----- analysis
    labels, amap = _destrieux()
    delta = {c: ft[c] - base[c] for c in stims}
    metrics = {}
    for c in stims:
        d = delta[c]
        metrics[c] = {
            "cosine_base_ft": cos(base[c], ft[c]),
            "delta_norm": float(np.linalg.norm(d)),
            "base_norm": float(np.linalg.norm(base[c])),
            "rel_change": float(np.linalg.norm(d) / (np.linalg.norm(base[c]) + 1e-9)),
        }

    # cat-specific delta = how cat moves beyond the animal-general baseline (dog)
    cat_spec = delta["cat"] - delta["dog"]
    interaction = {
        "cat_minus_horse": delta["cat"] - delta["horse"],
        "cat_minus_dog": cat_spec,
        "cat_minus_neutral": delta["cat"] - delta["neutral"],
    }

    # where does the raw cat-delta and the cat-specific delta land?
    roi_report = {}
    for name, stat in [("delta_cat", delta["cat"]), ("cat_specific", cat_spec)]:
        roi_report[name] = {}
        for roiname, roi in [("visual", VISUAL), ("language", LANGUAGE),
                             ("auditory", AUDITORY)]:
            top, prior = roi_share(np.abs(stat), labels, amap, roi)
            roi_report[name][roiname] = {"top5pct_share": top, "prior": prior,
                                         "enrichment": top / (prior + 1e-9)}

    # ----- write report
    lines = ["# TRIBE v2 x ascii-cat LoRA: does a cat-art finetune move the brain map?\n"]
    lines.append("Text-only. Same events per category reused across extractors, so the")
    lines.append("audio branch is common-mode and Delta isolates the Llama text features.\n")
    lines.append("## 1. How much does each category move (base -> finetune)?\n")
    lines.append("| category | cosine(base,ft) | ||Delta|| | ||base|| | rel.change |")
    lines.append("|---|---|---|---|---|")
    for c in ["cat", "dog", "horse", "neutral"]:
        m = metrics[c]
        lines.append(f"| {c} | {m['cosine_base_ft']:.4f} | {m['delta_norm']:.3f} | "
                     f"{m['base_norm']:.3f} | {m['rel_change']:.3%} |")
    lines.append("\n(Hypothesis: cat moves most; dog tests animal-general vs cat-specific.)\n")

    lines.append("## 2. Where does the cat change localize? (|delta|, top-5% vertices)\n")
    for name in ["delta_cat", "cat_specific"]:
        lines.append(f"### {name}")
        for roiname in ["visual", "language", "auditory"]:
            r = roi_report[name][roiname]
            lines.append(f"- {roiname}: {r['top5pct_share']:.1%} of top vertices "
                         f"(prior {r['prior']:.1%}, enrichment {r['enrichment']:.2f}x)")
        lines.append("")

    lines.append("## 3. Strongest regions of the cat-specific delta (cat - dog)\n")
    for val, nm, nv in top_regions(np.abs(cat_spec), labels, amap, n=10, signed=False):
        lines.append(f"  {val:+.4f}  {nm}  ({nv} vtx)")

    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    (OUT / "metrics.json").write_text(json.dumps(
        {"metrics": metrics, "roi": roi_report}, indent=2))
    print("\n" + report)

    # ----- figures
    plot_map(delta["cat"], "delta_cat.png", "ascii-cat finetune: Delta on CAT stimuli")
    plot_map(cat_spec, "cat_specific.png",
             "cat-specific delta (cat - dog): what the cat-art LoRA adds")
    plot_map(interaction["cat_minus_horse"], "cat_minus_horse.png",
             "interaction: Delta_cat - Delta_horse")
    print("\nDone. Outputs in", OUT)


if __name__ == "__main__":
    main()
