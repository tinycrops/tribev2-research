#!/usr/bin/env python
"""In-silico language localizer with TRIBE v2 (replicates the method of the
paper's Fig. 5).

We feed the model controlled linguistic stimuli (as text -> speech), read out
its *predicted* fsaverage5 cortical responses, and contrast conditions exactly
as a neuroscientist would analyse a localizer fMRI experiment. With no real
brain data, we test whether the model has internalised known neuroscience:

  1. Sentences  >  word-lists   -> should localise to the language network
                                   (lateral temporal + inferior frontal), and
                                   be LEFT-lateralised.
  2. Speech-evoked response      -> should peak in peri-sylvian auditory/language
                                   cortex.

Outputs PNual surface maps + a quantitative summary to work/demo_out/.
"""
import os
import time
import tempfile
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).parent / "demo_out"
OUT.mkdir(exist_ok=True)
CACHE = "/home/ath/tribev2-research/cache"
LH = slice(0, 10242)          # fsaverage5 left hemisphere
RH = slice(10242, 20484)      # fsaverage5 right hemisphere

# ----------------------------------------------------------------------------- stimuli
SENTENCES = [
    "The lawyer questioned the witness during the trial.",
    "My sister baked a chocolate cake for the party.",
    "The scientists discovered a new planet last year.",
    "He carefully repaired the broken clock on the shelf.",
    "The children played in the garden after school.",
    "She wrote a long letter to her old friend.",
    "The farmer harvested the wheat before the storm.",
    "We watched the sun set behind the tall mountains.",
]
# same words, scrambled into syntax-free lists (classic language localizer control)
rng = np.random.default_rng(0)
def _scramble(s):
    w = s.rstrip(".").split()
    rng.shuffle(w)
    return " ".join(w) + "."
WORDLISTS = [_scramble(s) for s in SENTENCES]


def enable_fast_math():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def mean_response(model, sentences, tag):
    """Predicted mean cortical response (20484,) to a block of sentences."""
    text = " ".join(sentences)
    with tempfile.NamedTemporaryFile("w", suffix=f"_{tag}.txt", dir=CACHE,
                                     delete=False) as f:
        f.write(text)
        path = f.name
    t0 = time.time()
    df = model.get_events_dataframe(text_path=path)
    preds, _ = model.predict(events=df, verbose=False)
    print(f"  [{tag}] {len(df)} events -> preds {preds.shape} in {time.time()-t0:.1f}s")
    return preds.mean(0), preds


def lateralization(x):
    """Mean over the top-10% most-activated language vertices, per hemisphere."""
    l, r = x[LH], x[RH]
    def top(v):
        thr = np.quantile(v, 0.90)
        return v[v >= thr].mean()
    L, R = top(l), top(r)
    li = (L - R) / (abs(L) + abs(R) + 1e-9)
    return L, R, li


def _destrieux():
    from nilearn import datasets
    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas["labels"]]
    amap = np.concatenate([atlas["map_left"], atlas["map_right"]])
    return labels, amap


def top_regions(stat, n=8):
    """Name the strongest cortical regions of a stat map via Destrieux."""
    labels, amap = _destrieux()
    rows = []
    for idx in np.unique(amap):
        name = labels[idx]
        if name in ("Unknown", "Medial_wall"):
            continue
        m = amap == idx
        rows.append((float(stat[m].mean()), name, int(m.sum())))
    rows.sort(reverse=True)
    return rows[:n]


# Destrieux regions making up auditory + peri-sylvian speech cortex
AUDITORY_REGIONS = (
    "G_temp_sup-G_T_transv",   # Heschl's gyrus (primary auditory)
    "G_temp_sup-Lateral",       # lateral STG
    "G_temp_sup-Plan_tempo",    # planum temporale
    "S_temporal_sup",           # superior temporal sulcus
)


def auditory_localization(stat):
    """Fraction of the top-5% activated vertices that fall in auditory cortex,
    vs. the prior (auditory share of all cortex). >1 ratio = enrichment."""
    labels, amap = _destrieux()
    aud_ids = [labels.index(r) for r in AUDITORY_REGIONS if r in labels]
    aud_mask = np.isin(amap, aud_ids)
    thr = np.quantile(stat, 0.95)
    top_mask = stat >= thr
    frac_top = aud_mask[top_mask].mean()
    prior = aud_mask.mean()
    return frac_top, prior, frac_top / (prior + 1e-9)


def plot_map(stat, fname, title, symmetric=True):
    from nilearn import datasets, plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    vmax = np.abs(stat).max() if symmetric else stat.max()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             subplot_kw={"projection": "3d"})
    for ax, hemi, mesh, bg, data in [
        (axes[0], "left", fs.infl_left, fs.sulc_left, stat[LH]),
        (axes[1], "right", fs.infl_right, fs.sulc_right, stat[RH]),
    ]:
        plotting.plot_surf_stat_map(
            mesh, data, hemi=hemi, view="lateral", bg_map=bg, axes=ax,
            colorbar=(hemi == "right"), cmap="cold_hot" if symmetric else "hot",
            vmax=vmax, threshold=vmax * 0.25, title=f"{hemi} hemisphere")
    fig.suptitle(title, fontsize=13)
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", OUT / fname)


def main():
    enable_fast_math()
    from tribev2 import TribeModel
    print("loading TRIBE v2 ...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE,
                                       device="auto")

    print("predicting condition responses ...")
    sent_mean, sent_all = mean_response(model, SENTENCES, "sentences")
    word_mean, _ = mean_response(model, WORDLISTS, "wordlists")
    np.savez(OUT / "condition_means.npz", sentences=sent_mean, wordlists=word_mean)

    # === Result 1 (headline): speech-evoked response localizes to auditory cortex
    aud_top, aud_prior, aud_ratio = auditory_localization(sent_mean)
    speech_regions = top_regions(sent_mean)

    # === Result 2: syntactic contrast (subtle: TTS audio is near-identical, so
    #     this isolates the text/syntax-driven component only)
    contrast = sent_mean - word_mean
    L, R, li = lateralization(contrast)

    lines = ["# TRIBE v2 in-silico evaluation (predicted fsaverage5 responses)\n"]
    lines.append("## Result 1 — Heard speech drives auditory / superior-temporal cortex\n")
    lines.append(f"- Auditory cortex share of the top-5% activated vertices: {aud_top:.1%}")
    lines.append(f"- Auditory cortex share of all cortex (prior):           {aud_prior:.1%}")
    lines.append(f"- Enrichment ratio:                                      {aud_ratio:.1f}x"
                 "  (>1 means speech preferentially activates auditory cortex)\n")
    lines.append("Strongest cortical regions for the speech-evoked response (Destrieux):\n")
    for val, name, nv in speech_regions:
        lines.append(f"  {val:+.4f}  {name}  ({nv} vtx)")
    lines.append("\n## Result 2 — Sentences > scrambled word-lists (text-driven contrast)\n")
    lines.append("NB: the spoken audio is near-identical across conditions, so this")
    lines.append("isolates only the small syntax/semantic component; low single-block power.\n")
    lines.append(f"- Left-hemisphere activation:  {L:+.4f}")
    lines.append(f"- Right-hemisphere activation: {R:+.4f}")
    lines.append(f"- Lateralization index:        {li:+.3f} "
                 f"({'left' if li > 0 else 'right'}-leaning)")
    report = "\n".join(lines)
    print("\n" + report)
    (OUT / "evaluation_report.md").write_text(report + "\n")

    # ---- figures
    plot_map(sent_mean, "speech_evoked_response.png",
             "TRIBE v2: speech-evoked response (predicted)", symmetric=False)
    plot_map(contrast, "contrast_sentences_vs_wordlists.png",
             "TRIBE v2: sentences > word-lists (predicted)", symmetric=True)
    print("\nDemo complete. See", OUT)


if __name__ == "__main__":
    main()
