#!/usr/bin/env python
"""Assemble a self-contained one-page HTML report for the ASCII-cat experiment."""
import base64, json
from pathlib import Path

D = Path(__file__).parent / "demo_out" / "asciicat"
metrics = json.loads((D / "metrics.json").read_text())


def img(name):
    b = base64.b64encode((D / name).read_bytes()).decode()
    return f"data:image/png;base64,{b}"


m = metrics["metrics"]
roi = metrics["roi"]

rows = "".join(
    f"<tr><td>{c}</td><td>{m[c]['cosine_base_ft']:.4f}</td>"
    f"<td>{m[c]['delta_norm']:.2f}</td><td>{m[c]['rel_change']:.2%}</td></tr>"
    for c in ["cat", "dog", "horse", "neutral"])


def roirow(stat):
    r = roi[stat]
    return "".join(
        f"<tr><td>{stat}</td><td>{k}</td><td>{v['top5pct_share']:.1%}</td>"
        f"<td>{v['prior']:.1%}</td><td>{v['enrichment']:.2f}×</td></tr>"
        for k, v in r.items())


roi_rows = roirow("delta_cat") + roirow("cat_specific")

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRIBE v2 × ASCII-cat LoRA — experiment</title>
<style>
 :root{{--ink:#1a1a1a;--mut:#666;--line:#e3e3e3;--accent:#7a3ea8;--good:#0a7d3c;--bad:#b3261e;--bg:#fbfafc}}
 *{{box-sizing:border-box}}
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
   max-width:980px;margin:0 auto;padding:34px 24px 60px;background:#fff}}
 h1{{font-size:25px;margin:0 0 2px}} h2{{font-size:17px;margin:30px 0 8px;border-bottom:2px solid var(--line);padding-bottom:4px}}
 .sub{{color:var(--mut);margin:0 0 18px}}
 .verdict{{background:var(--bg);border:1px solid var(--line);border-left:4px solid var(--accent);
   border-radius:8px;padding:14px 18px;margin:16px 0}}
 .verdict b{{color:var(--accent)}}
 table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}}
 th,td{{border:1px solid var(--line);padding:6px 10px;text-align:left}}
 th{{background:#f4f1f7;font-weight:600}} td:first-child{{font-weight:600}}
 code,pre{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
 pre{{background:#1f1d22;color:#eee;border-radius:8px;padding:14px 16px;overflow-x:auto;font-size:13px;line-height:1.5}}
 .tag{{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;vertical-align:middle}}
 .ref{{background:#fde9e7;color:var(--bad)}} .ok{{background:#e3f5ea;color:var(--good)}}
 figure{{margin:14px 0}} img{{max-width:100%;border:1px solid var(--line);border-radius:8px}}
 figcaption{{color:var(--mut);font-size:13px;margin-top:5px}}
 ul{{margin:8px 0 8px 0;padding-left:22px}} li{{margin:4px 0}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
 .meta{{color:var(--mut);font-size:12.5px;margin-top:40px;border-top:1px solid var(--line);padding-top:12px}}
 .hl{{background:#fff3bf;padding:0 3px;border-radius:3px}}
</style></head><body>

<h1>Does an ASCII-cat LoRA finetune change TRIBE v2's predicted brain maps?</h1>
<p class="sub">Swapping the Llama-3.2-3B text feature-extractor for a cat-art finetune · GB10 / Blackwell · 2026-06-15</p>

<div class="verdict">
<b>Verdict.</b> The finetune is real but its effect is <b>not cat-specific</b>: it shifts
cat, dog, horse, and neutral speech almost identically (~8%), and the shift lands in the
<b>language network</b> (62.9% of most-changed vertices, 5.4× enrichment) while
<b>avoiding visual cortex (0%)</b>. Both intuitive hypotheses — a cat-selective change, and
a push toward visual cortex because ASCII art is "drawing" — were <span class="tag ref">REFUTED</span>.
<br><br><i>TRIBE's predicted brain maps are sensitive to the extractor's <b>distribution</b>
but robust to its <b>semantic specialization</b>.</i>
</div>

<h2>Method</h2>
<ul>
<li><b>Model:</b> TRIBE v2 (<code>facebook/tribev2</code>) predicts fMRI on the fsaverage5 surface
(20484 vertices). Its text branch uses <code>meta-llama/Llama-3.2-3B</code> as a <i>frozen</i>
feature extractor — 20 hidden-state layers, group-mean pooled, into the brain encoder.</li>
<li><b>Finetune:</b> ASCII-cat LoRA reproduced in bf16 from the real dataset
<code>pookie3000/ascii-cats</code> (201 cats, 10 epochs) and merged into the base. The published
adapter is GGUF-only and won't load for hidden-state extraction, so we retrain — a controlled
A/B where the LoRA delta is the only variable.</li>
<li><b>Stimuli:</b> minimal pairs — identical 10-sentence templates, swap the creature
(cat / dog / horse) + a neutral weather passage.</li>
<li><b>Key control:</b> events (text→TTS→Parakeet→words) are LLM-independent, so we compute them
<span class="hl">once and reuse</span> across extractors. The audio (w2v-bert) branch is then
common-mode and <b>Δ isolates the Llama text features</b>.</li>
</ul>

<h2>Result 1 — magnitude: a uniform ~8% drift, not cat-selective</h2>
<table><tr><th>category</th><th>cosine(base, ft)</th><th>‖Δ‖</th><th>rel. change</th></tr>
{rows}</table>
<p class="sub">Cat is not the largest — <b>dog moves most</b>. The cat-art LoRA induces a broad,
roughly uniform representational drift rather than a "cat fingerprint."</p>

<h2>Result 2 — localization: language network, not visual cortex</h2>
<table><tr><th>stat</th><th>ROI</th><th>top-5% share</th><th>prior</th><th>enrichment</th></tr>
{roi_rows}</table>
<p class="sub"><code>delta_cat</code> = raw finetune shift on cat stimuli (dominated by the generic
drift). <code>cat_specific</code> = cat − dog, isolating the part unique to "cat" (a small, noisy
2nd-order contrast; leans auditory/superior-temporal, still visually de-enriched — read lightly).</p>

<div class="grid">
<figure><img src="{img('delta_cat.png')}"><figcaption>Δ on CAT stimuli (base→finetune): peri-sylvian / lateral-temporal language cortex.</figcaption></figure>
<figure><img src="{img('cat_specific.png')}"><figcaption>cat − dog: the cat-specific component (small, noisy).</figcaption></figure>
</div>

<h2>Result 3 — rigor control <span class="tag ok">PASSED</span></h2>
<p>Running <b>stock Llama through the exact finetune code path</b> (fresh load + separate cache)
reproduces the base maps to <b>Δ = 0.004% (cosine 1.00000)</b>. So the 8% is genuinely the LoRA —
not a dtype, cache, or code-path artifact. <span class="sub">(An earlier run reporting Δ = exactly 0
was a feature-cache reuse bug, now fixed and guarded by this null control.)</span></p>

<h2>Replicate it</h2>
<pre>cd /home/ath/tribev2-research &amp;&amp; source env.sh &amp;&amp; cd work

# 1. reproduce + merge the ASCII-cat LoRA  (~2 min on GB10)
python train_asciicat_lora.py        # -> models/llama32-3b-asciicat-merged/

# 2. run the experiment  (events + base + finetune; resumable via demo_out/asciicat/*.npz)
python experiment_asciicat.py        # -> demo_out/asciicat/{{report.md,metrics.json,*.png}}

# 3. rigor control: stock Llama via the finetune path (expect Delta ~ 0)
python control_nullswap.py

# 4. rebuild this page
python build_html.py                 # -> demo_out/asciicat/experiment.html</pre>

<h2>Gotchas that made this hard (read before re-running)</h2>
<ul>
<li><b>Don't swap weights on the live extractor.</b> It's a frozen exca/pydantic config, and exca
<i>memoizes the feature-cache uid</i> — an in-place swap silently serves the cached <b>base</b>
features (gave a false Δ=0). Build a <span class="hl">fresh</span>
<code>TribeModel.from_pretrained(..., config_update={{"data.text_feature.model_name": &lt;dir&gt;}})</code>
with a <b>separate cache_folder</b>.</li>
<li><b>Local model dirs fail neuralset's validator</b> (whitelist + hub API). Append the path to
<code>neuralset/extractors/data/huggingface-repos.txt</code> <i>and</i> to the in-memory
<code>HuggingFaceText._REPOS</code> ClassVar before the first extractor builds.</li>
<li><b>Always validate with the null-swap</b> (step 3). A bit-for-bit Δ=0 means the cache, not the
model, is talking.</li>
<li><b>ASCII art can't go through TTS</b> (it pronounces the punctuation). Stimuli reach Llama as
TTS→transcribed words. To probe raw ASCII, inject glyphs straight into the Llama extractor as
word-events — the natural maximal-effect follow-up.</li>
</ul>

<p class="meta">TRIBE v2 · Llama-3.2-3B text branch · dataset pookie3000/ascii-cats ·
artifacts in <code>work/demo_out/asciicat/</code> · full narrative in <code>FINDINGS.md</code>.</p>
</body></html>"""

out = D / "experiment.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, f"({len(HTML)//1024} KB + embedded images)")
