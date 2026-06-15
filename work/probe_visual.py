#!/usr/bin/env python
"""Probe A: does TRIBE v2's VIDEO branch (V-JEPA 2) produce a sane, non-degenerate,
visual-cortex-localized map for (a) a real cat photo and (b) a known-good ASCII-cat
bitmap?  If the real photo lands in visual cortex and the maps are non-trivial, the
cross-modal design is viable.  The text LoRA is irrelevant here (text branch gets
zero-features for a silent video), so we use the base model.

Each still is encoded as a static 256x256 mp4 (+ silent audio track so
ExtractAudioFromVideo doesn't fail).  V-JEPA gets a frozen clip -- OOD for a video
model, which is exactly what we're testing.
"""
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiment_asciicat import (enable_fast_math, CACHE, LH, RH, _destrieux,
                                 VISUAL, LANGUAGE, AUDITORY, roi_share,
                                 top_regions, cos)

OUT = Path(__file__).parent / "demo_out" / "phase2"
A = OUT / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def render_ascii_to_png(txt_path, png_path, size=256):
    art = Path(txt_path).read_text()
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
    lines = art.split("\n")
    # center the block
    bb = d.textbbox((0, 0), "M", font=font)
    ch, cw = bb[3] - bb[1], bb[2] - bb[0]
    th = len(lines) * (ch + 6)
    y = max(4, (size - th) // 2)
    for ln in lines:
        w = d.textlength(ln, font=font)
        d.text(((size - w) / 2, y), ln, fill="black", font=font)
        y += ch + 6
    img.save(png_path)


def png_to_static_mp4(png_path, mp4_path, dur=6, fps=30, size=256):
    # loop image + silent audio so the audio-extraction transform has a stream
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(png_path),
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", str(dur), "-r", str(fps),
        "-vf", f"scale={size}:{size}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def prep_assets():
    # real cat -> 256 png -> mp4
    rc_png = A / "real_cat_256.png"
    Image.open(A / "real_cat.jpg").convert("RGB").resize((256, 256)).save(rc_png)
    png_to_static_mp4(rc_png, OUT / "real_cat.mp4")
    # ascii cat -> bitmap png -> mp4
    ac_png = A / "ascii_cat_bitmap.png"
    render_ascii_to_png(A / "ascii_cat.txt", ac_png)
    png_to_static_mp4(ac_png, OUT / "ascii_cat.mp4")
    print("assets: real_cat.mp4, ascii_cat.mp4 (+ pngs)")


def describe(name, m, labels, amap):
    nz = np.count_nonzero(m)
    print(f"\n[{name}] map shape {m.shape}  nonzero {nz}/{m.size}  "
          f"||m||={np.linalg.norm(m):.3f}  range[{m.min():.3f},{m.max():.3f}]  "
          f"std={m.std():.4f}")
    out = {"norm": float(np.linalg.norm(m)), "nonzero": int(nz), "std": float(m.std())}
    for roiname, roi in [("visual", VISUAL), ("language", LANGUAGE),
                         ("auditory", AUDITORY)]:
        top, prior = roi_share(np.abs(m), labels, amap, roi)
        out[roiname] = {"share": top, "prior": prior, "enrich": top / (prior + 1e-9)}
        print(f"   {roiname:9s}: top-5% share {top:.1%}  (prior {prior:.1%}, "
              f"enrichment {top/(prior+1e-9):.2f}x)")
    print("   top regions:")
    for val, nm, nv in top_regions(np.abs(m), labels, amap, n=6, signed=False):
        print(f"      {val:.4f}  {nm}  ({nv} vtx)")
    return out


def main():
    enable_fast_math()
    prep_assets()
    from tribev2 import TribeModel
    print("loading TRIBE v2 (base) ...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE,
                                       device="auto")
    labels, amap = _destrieux()

    maps = {}
    for name in ["real_cat", "ascii_cat"]:
        print(f"\n=== building events for {name} (video branch) ===")
        ev = model.get_events_dataframe(video_path=str(OUT / f"{name}.mp4"))
        print(f"  raw events: {len(ev)} rows, types={dict(ev['type'].value_counts())}")
        # ISOLATE the video branch: drop the spuriously-transcribed Audio/Word/Text
        # events (silent-audio artifact) so only V-JEPA contributes; text+audio
        # branches fall back to their zero missing-default.
        ev = ev[ev["type"] == "Video"].reset_index(drop=True)
        print(f"  video-only events: {len(ev)} rows")
        preds, segs = model.predict(events=ev, verbose=False)
        print(f"  preds {preds.shape} over {len(segs)} segments")
        maps[name] = preds.mean(0)

    np.savez(OUT / "visual_maps.npz", **maps)
    summ = {}
    for name in maps:
        summ[name] = describe(name, maps[name], labels, amap)

    print("\n=== verdict ===")
    rc, ac = maps["real_cat"], maps["ascii_cat"]
    print(f"cosine(real_cat, ascii_cat) = {cos(rc, ac):.4f}")
    print("real-cat visual enrichment:", f"{summ['real_cat']['visual']['enrich']:.2f}x")
    print("Feasible if real_cat map is non-degenerate AND visual-enriched (>1x).")
    import json
    (OUT / "visual_summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
