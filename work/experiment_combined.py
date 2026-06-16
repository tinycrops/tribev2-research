#!/usr/bin/env python
"""Combined image+text experiment: present the SAME line-art ASCII cat to TRIBE v2
through BOTH branches at once -- the ascii as a bitmap (V-JEPA) AND the same ascii as
text (Llama) in one time-aligned events frame, so the brain model FUSES them. Every
prior probe isolated a single branch; this is the first fused stimulus.

base vs finetune still isolates the Llama contribution, but now in the PRESENCE of the
matching visual input. Questions:
  1. Does adding the visual bitmap change the finetune's text-effect (amplify/damp)?
  2. Is fusion ~additive (CO ~ blend of visual-only + text-only) or non-linear?
  3. In the fused condition, does the finetune pull the map toward the VISUAL rep
     (cos(CO_ft, visual) > cos(CO_base, visual)) -- convergence that text-alone lacked?

Reuses the substantiality cat pieces + their cached text-only maps (subst_*.npz).
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_asciicat import (enable_fast_math, _destrieux, VISUAL, roi_share,
                                 cos, BASE_NAME, MERGED)
from experiment_phase2 import (render_ascii_png, static_mp4, video_events, text_events,
                               load_model, whitelist_merged, predict_mean)
from experiment_substantiality import load_pieces

OUT = Path(__file__).parent / "demo_out" / "phase2"
CASSET = OUT / "combined_assets"; CASSET.mkdir(parents=True, exist_ok=True)
N_COMBINED = 20


def make_bitmap_mp4(ascii_str, stem):
    png = CASSET / f"{stem}.png"
    mp4 = CASSET / f"{stem}.mp4"
    if not mp4.exists():
        render_ascii_png(ascii_str, png)
        static_mp4(png, mp4)
    return str(mp4)


def combined_events(model, ascii_str, mp4, skel):
    """One events frame: a Video event (the ascii bitmap) + the same ascii as Word
    events, on a shared timeline, with the Video trimmed to the word span so (nearly)
    every kept TR segment contains BOTH modalities -> genuine fusion."""
    vev = video_events(model, mp4)                      # Video-only row
    tev = text_events(skel, ascii_str)                  # Word rows w/ ascii context
    wstop = float((tev["start"] + tev["duration"]).max())
    vev = vev.copy()
    vev.loc[:, "start"] = 0.0
    vev.loc[:, "duration"] = min(5.5, max(1.0, wstop))  # mp4 is 6s; cover the words
    if "stop" in vev.columns:
        vev.loc[:, "stop"] = vev["start"] + vev["duration"]
    cols = list(dict.fromkeys(list(vev.columns) + list(tev.columns)))
    comb = pd.concat([vev.reindex(columns=cols), tev.reindex(columns=cols)],
                     ignore_index=True)
    comb["timeline"] = "default"
    comb["subject"] = "default"
    for c in ("session", "task", "run"):
        if c in comb.columns:
            comb[c] = comb[c].fillna("default")
    return comb


def main():
    enable_fast_math()
    whitelist_merged()
    pieces = [p for p in load_pieces() if p["cat"] == "cat"][:N_COMBINED]
    print(f"combined experiment on {len(pieces)} line-art ASCII cats")
    skel = pickle.loads((Path("demo_out/asciicat/events.pkl")).read_bytes())["cat"]

    # text-only maps from the substantiality run (same piece ids)
    TOb = np.load(OUT / "subst_base.npz")
    TOf = np.load(OUT / "subst_ft.npz")

    co_b_path = OUT / "comb_base.npz"
    vo_b_path = OUT / "comb_visonly.npz"
    co_f_path = OUT / "comb_ft.npz"
    cev_path = OUT / "comb_events.pkl"

    # ---- Phase base: combined-base + visual-only-base; cache the fused events ----
    if co_b_path.exists() and vo_b_path.exists() and cev_path.exists():
        COb = dict(np.load(co_b_path)); VOb = dict(np.load(vo_b_path))
        cev = pickle.loads(cev_path.read_bytes())
        print("loaded cached base-phase maps")
    else:
        m = load_model(BASE_NAME, "_comb_base")
        COb, VOb, cev = {}, {}, {}
        for i, p in enumerate(pieces):
            mp4 = make_bitmap_mp4(p["content"], p["id"])
            ce = combined_events(m, p["content"], mp4, skel)
            cev[p["id"]] = ce
            COb[p["id"]] = predict_mean(m, ce)
            VOb[p["id"]] = predict_mean(m, video_events(m, mp4))   # bitmap alone
            print(f"  base {i+1}/{len(pieces)} {p['id']}: combined+visualonly done")
        np.savez(co_b_path, **COb); np.savez(vo_b_path, **VOb)
        cev_path.write_bytes(pickle.dumps(cev))
        import gc; del m; gc.collect(); torch.cuda.empty_cache()

    # ---- Phase ft: combined-ft on the SAME fused events ----
    if co_f_path.exists():
        COf = dict(np.load(co_f_path))
        print("loaded cached ft-phase maps")
    else:
        m = load_model(MERGED, "_comb_ft")
        COf = {}
        for i, p in enumerate(pieces):
            COf[p["id"]] = predict_mean(m, cev[p["id"]])
            print(f"  ft {i+1}/{len(pieces)} {p['id']}: combined-ft done")
        np.savez(co_f_path, **COf)
        import gc; del m; gc.collect(); torch.cuda.empty_cache()

    # ----------------------------------------------------------------- analysis
    labels, amap = _destrieux()
    def ve(mp): t, pr = roi_share(np.abs(mp), labels, amap, VISUAL); return t/(pr+1e-9)
    def rel(a, b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-9))

    rows = []
    for p in pieces:
        i = p["id"]
        if not all(i in d for d in (COb, COf, VOb)) or i not in TOb.files:
            continue
        cob, cof, vob = COb[i], COf[i], VOb[i]
        tob, tof = TOb[i], TOf[i]
        # additivity: is combined-base ~ a blend of visual-only + text-only?
        blend = vob/ (np.linalg.norm(vob)+1e-9) + tob/(np.linalg.norm(tob)+1e-9)
        rows.append({
            "id": i,
            # finetune shift magnitude: text-only vs fused
            "rel_ft_textonly": rel(tof, tob),
            "rel_ft_combined": rel(cof, cob),
            # visual enrichment
            "ve_textonly_base": ve(tob), "ve_combined_base": ve(cob),
            "ve_combined_ft": ve(cof), "ve_visualonly": ve(vob),
            # fusion geometry
            "cos_comb_visual": cos(cob, vob), "cos_comb_text": cos(cob, tob),
            "cos_comb_blend": cos(cob, blend),
            # convergence-in-fusion: does finetune move fused map toward the visual rep?
            "cos_combBase_visual": cos(cob, vob),
            "cos_combFt_visual": cos(cof, vob),
        })
    (OUT / "comb_rows.json").write_text(json.dumps(rows, indent=2))

    def M(k): return float(np.mean([r[k] for r in rows]))
    def S(k): return float(np.std([r[k] for r in rows]))
    conv = np.array([r["cos_combFt_visual"] - r["cos_combBase_visual"] for r in rows])

    summary = {"n": len(rows),
        "rel_ft_textonly": [M("rel_ft_textonly"), S("rel_ft_textonly")],
        "rel_ft_combined": [M("rel_ft_combined"), S("rel_ft_combined")],
        "ve_textonly_base": M("ve_textonly_base"), "ve_visualonly": M("ve_visualonly"),
        "ve_combined_base": M("ve_combined_base"), "ve_combined_ft": M("ve_combined_ft"),
        "cos_comb_visual": M("cos_comb_visual"), "cos_comb_text": M("cos_comb_text"),
        "cos_comb_blend": M("cos_comb_blend"),
        "fusion_convergence_gain": [float(conv.mean()), float(conv.std()),
                                    int((conv > 0).sum()), len(conv)]}
    (OUT / "comb_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ COMBINED (image+text fusion), n=%d ================" % len(rows))
    print(f"finetune effect ‖Δ‖/‖base‖:  text-only {M('rel_ft_textonly'):.1%}  ->  "
          f"fused {M('rel_ft_combined'):.1%}   (does visual context change it?)")
    print(f"visual enrichment:  text-only base {M('ve_textonly_base'):.2f}x | "
          f"visual-only {M('ve_visualonly'):.2f}x | fused base {M('ve_combined_base'):.2f}x | "
          f"fused ft {M('ve_combined_ft'):.2f}x")
    print(f"fusion geometry (cos of fused-base to):  visual {M('cos_comb_visual'):.2f} | "
          f"text {M('cos_comb_text'):.2f} | (visual+text) blend {M('cos_comb_blend'):.2f}")
    print(f">> fusion convergence gain (finetune pulls fused map toward visual rep): "
          f"{conv.mean():+.4f} ± {conv.std():.4f}  ({int((conv>0).sum())}/{len(conv)} pos)")
    print("\nDone -> demo_out/phase2/comb_summary.json")


if __name__ == "__main__":
    main()
