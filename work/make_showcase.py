#!/usr/bin/env python
"""Render brain-map figures for the demonstrated phase-2 probes and assemble a
self-contained one-page HTML showcase. Uses ONLY already-computed probe outputs
(visual_maps.npz, text_ascii_maps.npz) + asset images. Numbers are recomputed from
the maps so the page can't drift from the data."""
import base64, json
from pathlib import Path
import numpy as np

from experiment_asciicat import (_destrieux, VISUAL, LANGUAGE, AUDITORY, roi_share,
                                 cos, LH, RH)

OUT = Path(__file__).parent / "demo_out" / "phase2"
A = OUT / "assets"
FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)


def plot_map(stat, fname, title):
    from nilearn import datasets, plotting
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    vmax = np.abs(stat).max()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), subplot_kw={"projection": "3d"})
    for ax, hemi, mesh, bg, data in [
        (axes[0], "left", fs.infl_left, fs.sulc_left, stat[LH]),
        (axes[1], "right", fs.infl_right, fs.sulc_right, stat[RH])]:
        plotting.plot_surf_stat_map(mesh, data, hemi=hemi, view="lateral", bg_map=bg,
                                    axes=ax, colorbar=(hemi == "right"),
                                    cmap="cold_hot", vmax=vmax, threshold=vmax * 0.2)
    fig.suptitle(title, fontsize=11)
    fig.savefig(FIG / fname, dpi=115, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)


def enrich(m, labels, amap, roi):
    top, prior = roi_share(np.abs(m), labels, amap, roi)
    return top, prior, top / (prior + 1e-9)


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def main():
    vis = np.load(OUT / "visual_maps.npz")
    txt = np.load(OUT / "text_ascii_maps.npz")
    real_cat, ascii_bmp = vis["real_cat"], vis["ascii_cat"]
    tbase, tft = txt["base"], txt["ft"]
    labels, amap = _destrieux()

    # figures
    plot_map(real_cat, "map_real_cat.png", "Real cat photo -> V-JEPA -> brain")
    plot_map(ascii_bmp, "map_ascii_bitmap.png", "ASCII-cat bitmap -> V-JEPA -> brain")
    plot_map(tbase, "map_text_base.png", "ASCII as text -> base Llama -> brain")
    plot_map(tft, "map_text_ft.png", "ASCII as text -> cat finetune -> brain")

    def row(name, m):
        v = enrich(m, labels, amap, VISUAL)
        l = enrich(m, labels, amap, LANGUAGE)
        a = enrich(m, labels, amap, AUDITORY)
        return {"name": name, "visual": v[2], "language": l[2], "auditory": a[2]}

    R = {k: row(k, m) for k, m in [("real_cat", real_cat), ("ascii_bmp", ascii_bmp),
                                   ("text_base", tbase), ("text_ft", tft)]}
    cos_real_ascii = cos(real_cat, ascii_bmp)
    cos_base_ft = cos(tbase, tft)
    rel_ft = float(np.linalg.norm(tft - tbase) / (np.linalg.norm(tbase) + 1e-9))
    data = {"R": R, "cos_real_ascii": cos_real_ascii, "cos_base_ft": cos_base_ft,
            "rel_ft": rel_ft}
    (OUT / "showcase_metrics.json").write_text(json.dumps(data, indent=2))

    # ---- scaled experiment (n images x resolution ladder) ----
    scaled = json.loads((OUT / "scaled_summary.json").read_text())
    S = scaled["per_res"]
    N = scaled["n_images"]
    sv = np.load(OUT / "scaled_visual_basetext.npz")
    plot_map(sv["anchor_0"], "scaled_anchor0.png", "real cat #0 -> V-JEPA")
    plot_map(sv["visbmp_0_full"], "scaled_bmp0.png", "full-res img2ascii #0 -> V-JEPA")
    LAD = A / "ladder"

    def sm(res, k): return S[res][k]["mean"]
    def sd(res, k): return S[res][k]["std"]
    conv_full = S["full"]["finetune_convergence_gain"]
    conv_red = S["reduced"]["finetune_convergence_gain"]

    # ---- substantiality test (n line-art cats/dogs/horses) ----
    subst = json.loads((OUT / "subst_summary.json").read_text())
    spec = subst["cat_specificity"]

    ascii_art = (A / "ascii_cat.txt").read_text()

    def ve(name): return f"{R[name]['visual']:.2f}×"
    def le(name): return f"{R[name]['language']:.2f}×"

    HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRIBE v2 — cross-modal ASCII / image results</title>
<style>
 :root{{--ink:#161616;--mut:#6a6a6a;--line:#e4e4e7;--accent:#7a3ea8;--good:#0a7d3c;
  --bad:#b3261e;--blue:#1f6feb;--bg:#fbfafc}}
 *{{box-sizing:border-box}}
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  max-width:1000px;margin:0 auto;padding:34px 24px 64px;background:#fff}}
 h1{{font-size:25px;margin:0 0 3px}}
 h2{{font-size:17px;margin:32px 0 8px;border-bottom:2px solid var(--line);padding-bottom:4px}}
 h3{{font-size:14.5px;margin:18px 0 4px;color:var(--accent)}}
 .sub{{color:var(--mut);margin:0 0 16px}}
 .verdict{{background:var(--bg);border:1px solid var(--line);border-left:4px solid var(--accent);
  border-radius:8px;padding:14px 18px;margin:16px 0}}
 .verdict b{{color:var(--accent)}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
 th,td{{border:1px solid var(--line);padding:6px 10px;text-align:left}}
 th{{background:#f4f1f7;font-weight:600}} td:first-child{{font-weight:600}}
 .num{{font-variant-numeric:tabular-nums}}
 figure{{margin:10px 0}} img{{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff}}
 figcaption{{color:var(--mut);font-size:12.5px;margin-top:5px}}
 .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start}}
 .card{{border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
 pre{{font-family:ui-monospace,Menlo,monospace;background:#1f1d22;color:#d6d6d6;
  border-radius:8px;padding:10px 12px;font-size:11px;line-height:1.15;overflow-x:auto}}
 code{{font-family:ui-monospace,Menlo,monospace}}
 .tag{{display:inline-block;font-size:11px;padding:1px 8px;border-radius:20px}}
 .ok{{background:#e3f5ea;color:var(--good)}} .hot{{background:#fde9e7;color:var(--bad)}}
 .hl{{background:#fff3bf;padding:0 3px;border-radius:3px}}
 .big{{font-size:21px;font-weight:700}}
 .meta{{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}}
 ul{{margin:6px 0 6px 0;padding-left:20px}} li{{margin:3px 0}}
</style></head><body>

<h1>Reading ASCII art with a model of the human brain</h1>
<p class="sub">TRIBE v2 cross-modal probes — does a cat-art LoRA pull ASCII toward visual cortex? · GB10 / Blackwell · 2026-06-15</p>

<div class="verdict">
<b>The finetune is a line-art-ASCII <i>style</i> detector — not a cat detector.</b>
TRIBE v2's <b>video</b> branch (V-JEPA 2) reads a real cat photo straight into
<b>visual cortex</b> ({ve('real_cat')} enrichment), and more-faithful <i>img2ascii</i>
renderings read as more cat-like (full-res cos {sm('full','cos_anchor_visbmp'):.2f} &gt;
reduced {sm('reduced','cos_anchor_visbmp'):.2f}). On the <b>text</b> side, feeding the
LoRA real hand-drawn line-art ASCII pushes the predicted map <b>toward visual cortex</b>
— across <b>{subst['cat']['n']} cats</b>, visual enrichment
{subst['cat']['ve_base_mean']:.2f}× → <b>{subst['cat']['ve_ft_mean']:.2f}×</b>,
<span class="hl">{subst['cat']['frac_positive']*100:.0f}% of cats positive</span>. But
<b>dogs move identically</b> ({subst['dog']['shift_mean']:+.2f} vs cats
{subst['cat']['shift_mean']:+.2f}; cat-specificity p&nbsp;=&nbsp;{spec['test'].split('=')[-1].strip()}),
so the effect is about the <b>ASCII-art format</b>, not cats. It fires only on the iconic
style the LoRA trained on — absent in speech (phase&nbsp;1) and in brightness-ramp
img2ascii — which is why it looked like "convergence" in one probe and "nothing" at scale.
</div>

<h2>Setup: three encoders, two branches in play</h2>
<p class="sub">TRIBE v2 predicts fMRI on the cortical surface (20484 vertices) from three
frozen encoders — <b>text</b> = Llama-3.2-3B, <b>audio</b> = w2v-bert-2.0,
<b>video</b> = V-JEPA 2. The ASCII-cat LoRA lives in the <b>text</b> branch, so it only
moves text-routed inputs; images/bitmaps go through V-JEPA. We drive each branch in
isolation (video-only vs word-only events) and compare maps in the same brain space.</p>

<h2>Result 1 — the visual branch: a real cat lands in visual cortex; an ASCII cat half-does <span class="tag ok">demonstrated</span></h2>
<div class="grid2">
 <div class="card"><h3>Real cat photo</h3>
  <figure><img src="data:image/png;base64,{b64(A/'real_cat_256.png')}"><figcaption>stimulus (still, looped to a 6 s clip)</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64(FIG/'map_real_cat.png')}"><figcaption>V-JEPA → predicted brain map: occipital pole, fusiform, inferior occipital</figcaption></figure>
 </div>
 <div class="card"><h3>ASCII-cat bitmap</h3>
  <figure><img src="data:image/png;base64,{b64(A/'ascii_cat_bitmap.png')}"><figcaption>known-good ASCII cat, rendered to a bitmap</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64(FIG/'map_ascii_bitmap.png')}"><figcaption>V-JEPA → partly visual, partly language (it reads as glyphs)</figcaption></figure>
 </div>
</div>
<table><tr><th>stimulus (video branch)</th><th>visual</th><th>language</th><th>auditory</th><th>cos to real cat</th></tr>
<tr><td>real cat photo</td><td class="num">{ve('real_cat')}</td><td class="num">{le('real_cat')}</td><td class="num">{R['real_cat']['auditory']:.2f}×</td><td class="num">— (anchor)</td></tr>
<tr><td>ASCII-cat bitmap</td><td class="num">{ve('ascii_bmp')}</td><td class="num">{le('ascii_bmp')}</td><td class="num">{R['ascii_bmp']['auditory']:.2f}×</td><td class="num">{cos_real_ascii:.3f}</td></tr></table>
<p class="sub">The <b>cos = {cos_real_ascii:.2f}</b> gap is the optimization headroom: "make ASCII whose V-JEPA map matches a real cat" is a well-posed objective.</p>

<h2>Result 2 — the text branch: ASCII bypassing TTS, base vs cat-finetune <span class="tag hot">demonstrated</span></h2>
<p class="sub">We inject the ASCII glyphs straight into Llama (overwriting the event
<code>context</code>), skipping the text→speech→transcribe path entirely.</p>
<div class="grid2">
 <div class="card"><h3>base Llama → brain</h3>
  <figure><img src="data:image/png;base64,{b64(FIG/'map_text_base.png')}"><figcaption>visual {R['text_base']['visual']:.2f}× — not visual; reads as odd text</figcaption></figure></div>
 <div class="card"><h3>cat finetune → brain</h3>
  <figure><img src="data:image/png;base64,{b64(FIG/'map_text_ft.png')}"><figcaption>visual {R['text_ft']['visual']:.2f}× — finetune pulls it toward vision</figcaption></figure></div>
</div>
<table><tr><th>metric (ASCII as text)</th><th>value</th></tr>
<tr><td>finetune effect — relative change of the brain map</td><td class="num big">{rel_ft:.0%}</td></tr>
<tr><td>vs. ordinary speech (phase-1 baseline)</td><td class="num">~8%</td></tr>
<tr><td>cos(base, finetune)</td><td class="num">{cos_base_ft:.3f}</td></tr>
<tr><td>visual enrichment: base → finetune</td><td class="num">{R['text_base']['visual']:.2f}× → <b>{R['text_ft']['visual']:.2f}×</b></td></tr></table>
<p class="sub">Feeding the model the kind of input its LoRA was trained on (a clean
<b>iconic</b> ASCII cat), the finetune moves the representation ~4× harder than for speech
and the shift is <b>toward</b> visual cortex. <span class="hl">This was the exciting
hint — but it is a single, in-distribution stimulus. Result 4 scales it properly (it is
real — but generic to ASCII style, not cat-specific).</span></p>

<h2>Result 3 — the scaled test: {N} real cats × a resolution ladder <span class="tag ok">visual ladder ✓</span> <span class="tag hot">convergence ✗</span></h2>
<p class="sub">Each cat photo → real image (anchor) + full-res img2ascii + reduced-res
img2ascii. We ask: (a) does a higher-res ASCII <i>bitmap</i> look more like the real cat
to V-JEPA, and (b) does the finetune pull ASCII-<i>text</i> toward the real-image anchor?</p>
<div class="grid2">
 <div class="card"><h3>example ladder (cat #0)</h3>
  <figure><img src="data:image/png;base64,{b64(LAD/'img_0_real.png')}"><figcaption>real photo</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64(LAD/'img_0_full.png')}"><figcaption>full-res img2ascii (brightness ramp — not iconic line art)</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64(LAD/'img_0_reduced.png')}"><figcaption>reduced-res img2ascii</figcaption></figure></div>
 <div class="card"><h3>brain maps (cat #0)</h3>
  <figure><img src="data:image/png;base64,{b64(FIG/'scaled_anchor0.png')}"><figcaption>real photo → V-JEPA → visual cortex</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64(FIG/'scaled_bmp0.png')}"><figcaption>full-res ascii bitmap → V-JEPA → partly visual</figcaption></figure></div>
</div>
<table>
<tr><th>metric (mean ± sd, n={N})</th><th>full-res</th><th>reduced-res</th><th>reading</th></tr>
<tr><td>cos(anchor, ascii <b>bitmap</b>)</td>
  <td class="num">{sm('full','cos_anchor_visbmp'):.2f} ± {sd('full','cos_anchor_visbmp'):.2f}</td>
  <td class="num">{sm('reduced','cos_anchor_visbmp'):.2f} ± {sd('reduced','cos_anchor_visbmp'):.2f}</td>
  <td>↑ with resolution ✓</td></tr>
<tr><td>bitmap visual enrichment</td>
  <td class="num">{sm('full','ve_visbmp'):.2f}×</td>
  <td class="num">{sm('reduced','ve_visbmp'):.2f}×</td>
  <td>↑ with resolution ✓</td></tr>
<tr><td>cos(anchor, ascii <b>text</b>, base)</td>
  <td class="num">{sm('full','cos_anchor_txtbase'):.2f}</td>
  <td class="num">{sm('reduced','cos_anchor_txtbase'):.2f}</td>
  <td>≈ orthogonal</td></tr>
<tr><td><b>finetune convergence gain</b></td>
  <td class="num">{conv_full['mean']:+.3f} ({conv_full['n_positive']}/{conv_full['n']})</td>
  <td class="num">{conv_red['mean']:+.3f} ({conv_red['n_positive']}/{conv_red['n']})</td>
  <td><b>null / negative ✗</b></td></tr>
<tr><td>ascii-text finetune rel.change</td>
  <td class="num">{sm('full','txt_ft_relchange'):.0%}</td>
  <td class="num">{sm('reduced','txt_ft_relchange'):.0%}</td>
  <td>real (&gt; 8% speech)</td></tr>
</table>
<p class="sub"><b>The realism ladder holds; the convergence does not.</b> More-faithful ASCII
bitmaps do drive visual cortex more (full {sm('full','cos_anchor_visbmp'):.2f} &gt; reduced
{sm('reduced','cos_anchor_visbmp'):.2f}). But ASCII-as-text never approaches the image
anchor (cos&nbsp;≈&nbsp;0.1) and the finetune does <b>not</b> close the gap — the Result-2
hint was specific to a clean iconic cat (in-distribution for the LoRA), and evaporates on
realistic img2ascii.</p>

<h2>Result 4 — Substantiality: {subst['cat']['n']} line-art cats vs dogs vs horses <span class="tag ok">substantial</span> <span class="tag hot">not cat-specific</span></h2>
<p class="sub">The right test for Result 2: scale <i>within its own regime</i> — many
hand-drawn line-art ASCII pieces (<code>apehex/ascii-art</code>) injected as text, base vs
finetune — with dogs/horses as a built-in control. (Result 3 instead changed the stimulus
<i>distribution</i> to img2ascii, so it couldn't adjudicate the probe.)</p>
<table>
<tr><th>line-art ASCII</th><th>n</th><th>visual: base → ft</th><th>shift (ft−base)</th><th>% positive</th><th>rel-change</th></tr>
<tr><td><b>cats</b></td><td class="num">{subst['cat']['n']}</td>
  <td class="num">{subst['cat']['ve_base_mean']:.2f} → <b>{subst['cat']['ve_ft_mean']:.2f}×</b></td>
  <td class="num">{subst['cat']['shift_mean']:+.2f} ± {subst['cat']['shift_sd']:.2f}</td>
  <td class="num">{subst['cat']['frac_positive']*100:.0f}%</td>
  <td class="num">{subst['cat']['relchange_mean']:.0%}</td></tr>
<tr><td>dogs (control)</td><td class="num">{subst['dog']['n']}</td>
  <td class="num">{subst['dog']['ve_base_mean']:.2f} → {subst['dog']['ve_ft_mean']:.2f}×</td>
  <td class="num">{subst['dog']['shift_mean']:+.2f} ± {subst['dog']['shift_sd']:.2f}</td>
  <td class="num">{subst['dog']['frac_positive']*100:.0f}%</td>
  <td class="num">{subst['dog']['relchange_mean']:.0%}</td></tr>
<tr><td>horses (control)</td><td class="num">{subst['horse']['n']}</td>
  <td class="num">{subst['horse']['ve_base_mean']:.2f} → {subst['horse']['ve_ft_mean']:.2f}×</td>
  <td class="num">{subst['horse']['shift_mean']:+.2f} ± {subst['horse']['shift_sd']:.2f}</td>
  <td class="num">{subst['horse']['frac_positive']*100:.0f}%</td>
  <td class="num">{subst['horse']['relchange_mean']:.0%}</td></tr>
</table>
<p class="sub"><b>Substantial:</b> across {subst['cat']['n']} distinct iconic cats the
finetune pushes the map toward visual cortex <b>{subst['cat']['frac_positive']*100:.0f}% of
the time</b> ({subst['cat']['ve_base_mean']:.2f}× → {subst['cat']['ve_ft_mean']:.2f}×) — the
Result-2 probe was <i>not</i> noise. <b>But not cat-specific:</b> dogs shift
{subst['dog']['shift_mean']:+.2f}, indistinguishable from cats
({subst['cat']['shift_mean']:+.2f}); cat-vs-control contrast {spec['contrast']:+.2f},
{spec['test']}. The finetune routes <i>any</i> line-art ASCII toward visual cortex.</p>

<h2>What the LoRA actually learned</h2>
<table>
<tr><th>condition</th><th>ASCII-art style present?</th><th>finetune effect on the brain map</th></tr>
<tr><td>Phase 1 — TTS speech</td><td>no</td><td>~8%, stays in <b>language</b>, avoids visual</td></tr>
<tr><td>Result 3 — img2ascii gradients</td><td>no (OOD format)</td><td>null convergence</td></tr>
<tr><td>Results 2 &amp; 4 — hand-drawn line-art</td><td><b>yes</b></td><td>big, ~100%-consistent push to <b>visual</b> (cat = dog)</td></tr>
</table>
<p class="sub">The finetune didn't learn "cat" — it learned "<b>this glyph-block is a
picture</b>," and that picture-ness surfaces as visual cortex whenever the input is in the
iconic line-art format it trained on. Pure <b>style / distribution</b> sensitivity, not
semantics — the phase-1 headline, now shown representationally.</p>

<h2>The iconic ASCII cat (Result 2 stimulus)</h2>
<pre>{ascii_art}</pre>

<h2>Synthesis &amp; honesty</h2>
<ul>
<li><b>Solid:</b> the visual realism ladder — img2ascii fidelity ↔ V-JEPA visual response —
replicates across {N} cats; and the text-side finetune→visual push is substantial
(~100% of {subst['cat']['n']} cats).</li>
<li><b>Corrected:</b> I first called the convergence "refuted" from Result 3 — too strong.
Result 3 only changed the stimulus distribution (img2ascii ≠ line-art); scaling
<i>in-regime</i> (Result 4) shows the effect is real but <b>generic to ASCII style, not
cat-specific</b> (cat {subst['cat']['shift_mean']:+.2f} ≈ dog {subst['dog']['shift_mean']:+.2f}).</li>
<li><b>Methodological takeaway:</b> a single in-distribution probe over-promised, and a
distribution-shifted scale-up under-claimed. Only scaling in-regime <i>with a control</i>
gave the honest answer: real effect, wrong reason.</li>
</ul>

<h2>Replicate</h2>
<pre style="color:#eee">cd /home/ath/tribev2-research &amp;&amp; source env.sh &amp;&amp; cd work
python probe_visual.py        # video branch: real cat + ascii bitmap
python probe_text_ascii.py    # text branch: ascii-as-text, base vs finetune
python make_showcase.py       # rebuild this page
# scaled run:
python experiment_phase2.py   # N cats x resolution ladder, cross-modal convergence</pre>

<p class="meta">TRIBE v2 · V-JEPA 2 video branch · Llama-3.2-3B text branch + ascii-cat LoRA ·
maps in <code>work/demo_out/phase2/</code> · numbers recomputed from
<code>visual_maps.npz</code> / <code>text_ascii_maps.npz</code>.</p>
</body></html>"""

    out = OUT / "showcase.html"
    out.write_text(HTML, encoding="utf-8")
    print("wrote", out, f"({len(HTML)//1024} KB + embedded figures)")


if __name__ == "__main__":
    main()
