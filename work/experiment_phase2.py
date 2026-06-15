#!/usr/bin/env python
"""Phase 2 (scaled): cross-modal ASCII/image experiment on TRIBE v2.

For N real cat images, build a resolution ladder and measure, in brain space:
  - VISUAL branch (V-JEPA 2): real image (ANCHOR) vs img2ascii bitmaps (full/reduced res)
  - TEXT  branch (Llama):     the same ascii injected as text, base vs ascii-cat finetune

Headline question: does cos(ascii map, real-image anchor) rise with resolution, and
does the cat finetune INCREASE that cross-modal convergence for ascii-as-text?

Branches are isolated (video-only events / word-only events) per the phase-2 mechanics.
Per-phase npz caching => resumable.
"""
import os
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import torch
import requests
from PIL import Image, ImageDraw, ImageFont

from experiment_asciicat import (enable_fast_math, CACHE, MERGED, BASE_NAME,
                                 _destrieux, VISUAL, LANGUAGE, AUDITORY,
                                 roi_share, cos)

OUT = Path(__file__).parent / "demo_out" / "phase2"
A = OUT / "assets"
IMGDIR = A / "ladder"
IMGDIR.mkdir(parents=True, exist_ok=True)

N_IMAGES = 6
W_FULL = 70       # img2ascii character width, full resolution
W_REDUCED = 26    # reduced resolution
RAMP = " .:-=+*#%@"   # light -> dark (space = white background, no ink)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


# ----------------------------------------------------------------- assets
def fetch_images(n):
    """Grab n distinct cat photos. cataas.com is a cat-image service; fall back
    to a couple of stable Wikimedia files."""
    paths = []
    for i in range(n):
        p = A / f"img_{i}.jpg"
        if p.exists() and p.stat().st_size > 5000:
            paths.append(p); continue
        ok = False
        for attempt in range(4):
            try:
                r = requests.get(f"https://cataas.com/cat?{i}_{attempt}",
                                 timeout=25, headers={"User-Agent": "research/1.0"})
                r.raise_for_status()
                if len(r.content) > 5000:
                    p.write_bytes(r.content); ok = True; break
            except Exception as e:
                print(f"  img {i} attempt {attempt}: {type(e).__name__}")
        if not ok:
            fb = ["https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
                  "https://upload.wikimedia.org/wikipedia/commons/0/0b/Cat_poster_1.jpg"]
            r = requests.get(fb[i % len(fb)], timeout=25,
                             headers={"User-Agent": "research/1.0"})
            r.raise_for_status(); p.write_bytes(r.content)
        # sanity: must open as an image
        Image.open(p).convert("RGB")
        paths.append(p)
        print(f"  image {i}: {p.name} ({p.stat().st_size//1024} KB)")
    return paths


def img2ascii(img_path, width):
    img = Image.open(img_path).convert("L")
    w, h = img.size
    rows = max(1, int(width * (h / w) * 0.5))   # chars ~2x taller than wide
    small = img.resize((width, rows))
    px = np.asarray(small, dtype=np.float32) / 255.0
    lines = []
    for r in range(rows):
        line = "".join(RAMP[min(len(RAMP) - 1, int((1 - px[r, c]) * len(RAMP)))]
                        for c in range(width))
        lines.append(line.rstrip() or " ")
    return "\n".join(lines)


def render_ascii_png(ascii_str, png_path, fsize=14):
    lines = ascii_str.split("\n")
    font = ImageFont.truetype(MONO, fsize)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cw = tmp.textlength("M", font=font)
    bb = tmp.textbbox((0, 0), "Mg", font=font)
    ch = bb[3] - bb[1] + 2
    W = int(max(len(l) for l in lines) * cw) + 8
    H = int(len(lines) * ch) + 8
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 4
    for l in lines:
        d.text((4, y), l, fill="black", font=font); y += ch
    img.save(png_path)


def static_mp4(png_path, mp4_path, dur=6, fps=30, size=256):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png_path),
           "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", str(dur),
           "-r", str(fps), "-vf", f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
           f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:white",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
           str(mp4_path)]
    subprocess.run(cmd, check=True)


def build_assets(paths):
    """Produce, per image: real mp4, and for each res the ascii text + bitmap mp4."""
    meta = {}
    for i, p in enumerate(paths):
        # real image -> 256 png -> mp4
        rc = IMGDIR / f"img_{i}_real.png"
        Image.open(p).convert("RGB").save(rc)
        static_mp4(rc, IMGDIR / f"img_{i}_real.mp4")
        meta[i] = {"real_mp4": str(IMGDIR / f"img_{i}_real.mp4")}
        for res, width in [("full", W_FULL), ("reduced", W_REDUCED)]:
            art = img2ascii(p, width)
            (IMGDIR / f"img_{i}_{res}.txt").write_text(art)
            png = IMGDIR / f"img_{i}_{res}.png"
            render_ascii_png(art, png)
            static_mp4(png, IMGDIR / f"img_{i}_{res}.mp4")
            meta[i][f"{res}_txt"] = art
            meta[i][f"{res}_mp4"] = str(IMGDIR / f"img_{i}_{res}.mp4")
    (OUT / "ladder_meta.json").write_text(json.dumps(
        {str(k): {kk: vv for kk, vv in v.items() if kk.endswith("mp4")}
         for k, v in meta.items()}, indent=2))
    return meta


# ----------------------------------------------------------------- events
def video_events(model, mp4):
    ev = model.get_events_dataframe(video_path=mp4)
    return ev[ev["type"] == "Video"].reset_index(drop=True)


def text_events(skeleton, ascii_str):
    lines = [l for l in ascii_str.split("\n") if l.strip() != ""]
    words = skeleton[skeleton["type"] == "Word"].reset_index(drop=True).copy()
    n = min(len(lines), len(words))
    ev = words.iloc[:n].copy()
    grow = []
    for i in range(n):
        grow.append(lines[i])
        ev.iloc[i, ev.columns.get_loc("text")] = lines[i][:64] or " "
        ev.iloc[i, ev.columns.get_loc("context")] = "\n".join(grow)
        if "sentence" in ev.columns:
            ev.iloc[i, ev.columns.get_loc("sentence")] = ascii_str
    return ev.reset_index(drop=True)


def predict_mean(model, ev):
    preds, _ = model.predict(events=ev, verbose=False)
    return preds.mean(0)


def whitelist_merged():
    import neuralset.extractors.base as _nb
    wl = Path(_nb.__file__).with_name("data") / "huggingface-repos.txt"
    lines = wl.read_text("utf8").splitlines()
    if MERGED not in lines:
        wl.write_text("\n".join(lines + [MERGED]))
    from neuralset.extractors.text import HuggingFaceText
    if MERGED not in HuggingFaceText._REPOS:
        HuggingFaceText._REPOS.append(MERGED)


def load_model(model_name, cache_suffix):
    from tribev2 import TribeModel
    cf = CACHE + cache_suffix
    Path(cf).mkdir(parents=True, exist_ok=True)
    m = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cf, device="auto",
                                   config_update={"data.text_feature.model_name": model_name})
    assert m.data.text_feature.model_name == model_name
    return m


# ----------------------------------------------------------------- run
def main():
    enable_fast_math()
    whitelist_merged()
    print("== fetching images =="); paths = fetch_images(N_IMAGES)
    print("== building assets (img2ascii, bitmaps, mp4s) =="); meta = build_assets(paths)
    skel = pickle.loads((Path("demo_out/asciicat/events.pkl")).read_bytes())["cat"]
    import gc

    vis_path = OUT / "scaled_visual_basetext.npz"
    ft_path = OUT / "scaled_ft.npz"
    idxs = list(range(N_IMAGES))
    ress = ["full", "reduced"]

    # Phase 1: base model -> all VISUAL maps + all BASE TEXT maps (video branch is
    # finetune-independent; base text shares the same model).
    if vis_path.exists():
        z = np.load(vis_path); V = {k: z[k] for k in z.files}
        print("loaded cached visual+basetext maps")
    else:
        print("\n== Phase 1: base model (visual anchors + ascii bitmaps + base text) ==")
        m = load_model(BASE_NAME, "_p2_base")
        V = {}
        for i in idxs:
            V[f"anchor_{i}"] = predict_mean(m, video_events(m, meta[i]["real_mp4"]))
            print(f"  img{i} anchor done")
            for res in ress:
                V[f"visbmp_{i}_{res}"] = predict_mean(m, video_events(m, meta[i][f"{res}_mp4"]))
                V[f"txtbase_{i}_{res}"] = predict_mean(m, text_events(skel, meta[i][f"{res}_txt"]))
                print(f"  img{i} {res}: visbmp + txtbase done")
        np.savez(vis_path, **V)
        del m; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: finetuned model -> FT TEXT maps
    if ft_path.exists():
        z = np.load(ft_path); F = {k: z[k] for k in z.files}
        print("loaded cached ft maps")
    else:
        print("\n== Phase 2: finetuned model (ascii-as-text, ft) ==")
        m = load_model(MERGED, "_p2_ft")
        F = {}
        for i in idxs:
            for res in ress:
                F[f"txtft_{i}_{res}"] = predict_mean(m, text_events(skel, meta[i][f"{res}_txt"]))
                print(f"  img{i} {res}: txtft done")
        np.savez(ft_path, **F)
        del m; gc.collect(); torch.cuda.empty_cache()

    # ----------------------------------------------------------------- analysis
    labels, amap = _destrieux()

    def visual_enrich(mp):
        top, prior = roi_share(np.abs(mp), labels, amap, VISUAL)
        return top / (prior + 1e-9)

    rows = []
    for i in idxs:
        anchor = V[f"anchor_{i}"]
        for res in ress:
            visbmp = V[f"visbmp_{i}_{res}"]
            tb = V[f"txtbase_{i}_{res}"]
            tf = F[f"txtft_{i}_{res}"]
            rows.append({
                "img": i, "res": res,
                "cos_anchor_visbmp": cos(anchor, visbmp),
                "cos_anchor_txtbase": cos(anchor, tb),
                "cos_anchor_txtft": cos(anchor, tf),
                "txt_ft_relchange": float(np.linalg.norm(tf - tb) / (np.linalg.norm(tb) + 1e-9)),
                "ve_anchor": visual_enrich(anchor),
                "ve_visbmp": visual_enrich(visbmp),
                "ve_txtbase": visual_enrich(tb),
                "ve_txtft": visual_enrich(tf),
            })

    def agg(res, key):
        v = [r[key] for r in rows if r["res"] == res]
        return float(np.mean(v)), float(np.std(v))

    summary = {"n_images": N_IMAGES, "per_res": {}}
    for res in ress:
        d = {}
        for key in ["cos_anchor_visbmp", "cos_anchor_txtbase", "cos_anchor_txtft",
                    "txt_ft_relchange", "ve_anchor", "ve_visbmp", "ve_txtbase", "ve_txtft"]:
            mu, sd = agg(res, key); d[key] = {"mean": mu, "std": sd}
        # the headline: cross-modal convergence gain from finetune (paired)
        conv = [r["cos_anchor_txtft"] - r["cos_anchor_txtbase"]
                for r in rows if r["res"] == res]
        d["finetune_convergence_gain"] = {
            "mean": float(np.mean(conv)), "std": float(np.std(conv)),
            "n_positive": int(np.sum(np.array(conv) > 0)), "n": len(conv)}
        summary["per_res"][res] = d

    (OUT / "scaled_rows.json").write_text(json.dumps(rows, indent=2))
    (OUT / "scaled_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ RESULTS ================")
    for res in ress:
        d = summary["per_res"][res]
        print(f"\n--- {res} resolution (n={N_IMAGES}) ---")
        print(f"  cos(anchor, ascii-BITMAP)   = {d['cos_anchor_visbmp']['mean']:.3f} ± {d['cos_anchor_visbmp']['std']:.3f}")
        print(f"  cos(anchor, ascii-TEXT base)= {d['cos_anchor_txtbase']['mean']:.3f} ± {d['cos_anchor_txtbase']['std']:.3f}")
        print(f"  cos(anchor, ascii-TEXT ft)  = {d['cos_anchor_txtft']['mean']:.3f} ± {d['cos_anchor_txtft']['std']:.3f}")
        g = d["finetune_convergence_gain"]
        print(f"  >> finetune convergence gain = {g['mean']:+.4f} ± {g['std']:.4f}  "
              f"({g['n_positive']}/{g['n']} images positive)")
        print(f"  txt finetune rel.change      = {d['txt_ft_relchange']['mean']:.1%}")
        print(f"  visual enrichment: anchor {d['ve_anchor']['mean']:.2f}x | "
              f"bitmap {d['ve_visbmp']['mean']:.2f}x | txt-base {d['ve_txtbase']['mean']:.2f}x | "
              f"txt-ft {d['ve_txtft']['mean']:.2f}x")
    print("\nDone. Summary -> demo_out/phase2/scaled_summary.json")


if __name__ == "__main__":
    main()
