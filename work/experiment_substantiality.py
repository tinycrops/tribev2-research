#!/usr/bin/env python
"""Substantiality test: is the n=1 probe effect (cat-art finetune pulls an iconic
ASCII cat's TEXT representation toward visual cortex, 0.85x->2.05x) real and
consistent across many IN-DISTRIBUTION line-drawn ASCII cats -- and is it cat-specific?

Stimuli: apehex/ascii-art `animals` shard, hand-made line-art ASCII, labelled by
sub-category. Cats vs Dogs vs Horses = the phase-1 minimal-pair contrast, now at scale.

Each piece -> injected as TEXT (line=word event, growing context) -> base Llama and
ascii-cat finetune -> brain map. Per piece we record visual enrichment (base, ft), the
shift (ft-base), and rel.change. Aggregate per category + cat-vs-control contrast.

Decision:
  substantial & cat-specific : visual push consistent across cats AND cats > dogs/horses
  substantial but generic    : consistent push for cats AND dogs/horses alike (style, not cats)
  insubstantial              : sign flips / huge variance across cats (n=1 was noise)
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from experiment_asciicat import (enable_fast_math, _destrieux, VISUAL, LANGUAGE,
                                 roi_share, cos)
from experiment_phase2 import text_events, load_model, whitelist_merged, predict_mean

OUT = Path(__file__).parent / "demo_out" / "phase2"
CATS_N, DOGS_N, HORSES_N = 40, 30, 22
MIN_LINES, MAX_LINES, MIN_CHARS = 4, 28, 50
SEED = 0


def load_pieces():
    f = hf_hub_download("apehex/ascii-art", "asciiart/train/animals.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(f)
    lab = df["labels"].fillna("")
    df = df.assign(nlines=df["content"].str.count("\n") + 1,
                   nchar=df["content"].str.len())
    ok = (df["nlines"].between(MIN_LINES, MAX_LINES)) & (df["nchar"] >= MIN_CHARS)

    def pick(sublabel, n):
        m = lab.str.contains(rf"(?:^|,){sublabel}(?:,|$)", case=False, regex=True) & ok
        sub = df[m].drop_duplicates("content")
        return sub.sample(min(n, len(sub)), random_state=SEED)

    sel = {"cat": pick("Cats", CATS_N), "dog": pick("Dogs", DOGS_N),
           "horse": pick("Horses", HORSES_N)}
    pieces = []
    for cat, sub in sel.items():
        for j, (_, row) in enumerate(sub.iterrows()):
            pieces.append({"id": f"{cat}_{j}", "cat": cat, "content": row["content"]})
    print(f"pieces: cats={len(sel['cat'])} dogs={len(sel['dog'])} horses={len(sel['horse'])}")
    return pieces


def run_phase(pieces, skel, model_name, suffix, cache_key):
    cache = OUT / f"subst_{cache_key}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"loaded cached {cache_key} ({len(z.files)} maps)")
        return {k: z[k] for k in z.files}
    m = load_model(model_name, suffix)
    out = {}
    for i, p in enumerate(pieces):
        ev = text_events(skel, p["content"])
        if len(ev) == 0:
            continue
        out[p["id"]] = predict_mean(m, ev)
        if (i + 1) % 10 == 0:
            print(f"  {cache_key}: {i+1}/{len(pieces)}")
    np.savez(cache, **out)
    import gc
    del m; gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    enable_fast_math()
    whitelist_merged()
    from experiment_asciicat import BASE_NAME, MERGED
    pieces = load_pieces()
    (OUT / "subst_pieces.json").write_text(json.dumps(
        [{"id": p["id"], "cat": p["cat"]} for p in pieces], indent=2))
    skel = pickle.loads((Path("demo_out/asciicat/events.pkl")).read_bytes())["cat"]

    print("\n== base Llama ==")
    base = run_phase(pieces, skel, BASE_NAME, "_subst_base", "base")
    print("\n== ascii-cat finetune ==")
    ft = run_phase(pieces, skel, MERGED, "_subst_ft", "ft")

    labels, amap = _destrieux()

    def ve(m):  # visual enrichment
        t, p = roi_share(np.abs(m), labels, amap, VISUAL); return t / (p + 1e-9)

    rows = []
    for p in pieces:
        i = p["id"]
        if i not in base or i not in ft:
            continue
        b, f = base[i], ft[i]
        rows.append({"id": i, "cat": p["cat"], "ve_base": ve(b), "ve_ft": ve(f),
                     "shift": ve(f) - ve(b),
                     "relchange": float(np.linalg.norm(f - b) / (np.linalg.norm(b) + 1e-9)),
                     "cos_base_ft": cos(b, f)})
    (OUT / "subst_rows.json").write_text(json.dumps(rows, indent=2))

    def agg(catname, key):
        v = np.array([r[key] for r in rows if r["cat"] == catname])
        return v

    summary = {}
    print("\n================ SUBSTANTIALITY ================")
    print(f"{'cat':6s} {'n':>3s} {'ve_base':>8s} {'ve_ft':>8s} {'shift(ft-base)':>16s} "
          f"{'%pos':>6s} {'relchg':>7s}")
    for c in ["cat", "dog", "horse"]:
        sh = agg(c, "shift")
        vb, vf = agg(c, "ve_base"), agg(c, "ve_ft")
        rc = agg(c, "relchange")
        pos = float((sh > 0).mean()) if len(sh) else float("nan")
        summary[c] = {"n": len(sh), "ve_base_mean": float(vb.mean()),
                      "ve_ft_mean": float(vf.mean()), "shift_mean": float(sh.mean()),
                      "shift_sd": float(sh.std()), "frac_positive": pos,
                      "relchange_mean": float(rc.mean())}
        print(f"{c:6s} {len(sh):3d} {vb.mean():8.2f} {vf.mean():8.2f} "
              f"{sh.mean():+8.3f} ± {sh.std():.3f}  {pos*100:5.0f}% {rc.mean():6.1%}")

    # cat-specificity: cat shift vs pooled dog+horse shift
    cat_sh = agg("cat", "shift")
    ctrl_sh = np.concatenate([agg("dog", "shift"), agg("horse", "shift")])
    contrast = float(cat_sh.mean() - ctrl_sh.mean())
    try:
        from scipy.stats import mannwhitneyu
        u, pval = mannwhitneyu(cat_sh, ctrl_sh, alternative="greater")
        ptxt = f"Mann-Whitney U one-sided p = {pval:.4f}"
    except Exception:
        ptxt = "(scipy unavailable; report means only)"
    summary["cat_specificity"] = {"cat_shift_mean": float(cat_sh.mean()),
                                  "control_shift_mean": float(ctrl_sh.mean()),
                                  "contrast": contrast, "test": ptxt}
    (OUT / "subst_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ncat-specificity: cat shift {cat_sh.mean():+.3f} vs control "
          f"{ctrl_sh.mean():+.3f}  -> contrast {contrast:+.3f}")
    print(" ", ptxt)
    print("\nverdict guide:")
    print("  cats shift>0 consistently AND contrast>0 sig -> substantial & cat-specific")
    print("  cats & controls both shift>0 similarly         -> substantial but generic (style)")
    print("  cats shift ~0 / frac_positive ~50% / high sd   -> n=1 probe was noise")
    print("\nDone -> demo_out/phase2/subst_summary.json")


if __name__ == "__main__":
    main()
