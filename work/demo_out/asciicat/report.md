# TRIBE v2 x ascii-cat LoRA: does a cat-art finetune move the brain map?

Text-only. Same events per category reused across extractors, so the
audio branch is common-mode and Delta isolates the Llama text features.

## 1. How much does each category move (base -> finetune)?

| category | cosine(base,ft) | ||Delta|| | ||base|| | rel.change |
|---|---|---|---|---|
| cat | 0.9966 | 1.897 | 22.578 | 8.400% |
| dog | 0.9964 | 2.062 | 23.643 | 8.720% |
| horse | 0.9967 | 1.768 | 21.085 | 8.383% |
| neutral | 0.9957 | 1.804 | 19.136 | 9.427% |

(Hypothesis: cat moves most; dog tests animal-general vs cat-specific.)

## 2. Where does the cat change localize? (|delta|, top-5% vertices)

### delta_cat
- visual: 0.0% of top vertices (prior 12.6%, enrichment 0.00x)
- language: 62.9% of top vertices (prior 11.6%, enrichment 5.43x)
- auditory: 0.4% of top vertices (prior 1.4%, enrichment 0.29x)

### cat_specific
- visual: 8.3% of top vertices (prior 12.6%, enrichment 0.66x)
- language: 44.9% of top vertices (prior 11.6%, enrichment 3.87x)
- auditory: 5.8% of top vertices (prior 1.4%, enrichment 4.26x)

## 3. Strongest regions of the cat-specific delta (cat - dog)

  +0.0075  S_occipital_ant  (105 vtx)
  +0.0050  G_temp_sup-Plan_tempo  (147 vtx)
  +0.0044  S_temporal_transverse  (45 vtx)
  +0.0043  G_temp_sup-Lateral  (341 vtx)
  +0.0040  S_temporal_sup  (905 vtx)
  +0.0039  S_oc-temp_med_and_Lingual  (303 vtx)
  +0.0039  S_oc-temp_lat  (118 vtx)
  +0.0037  G_occipital_middle  (256 vtx)
  +0.0032  G_temp_sup-G_T_transv  (85 vtx)
  +0.0031  S_parieto_occipital  (263 vtx)
