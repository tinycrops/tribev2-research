"""GPU Parakeet (NVIDIA) ASR with word-level timestamps for TRIBE v2.

Replaces the uvx/whisperx (CPU-only on aarch64, ctranslate2 has no CUDA) path.
transformers' CTC pipeline word-timestamp postprocess is broken for the Parakeet
tokenizer, so we run the model directly and derive word timestamps from a greedy
CTC alignment.
"""
from __future__ import annotations

import os
import threading

import librosa
import pandas as pd
import torch

_MODEL_ID = os.environ.get("TRIBE_PARAKEET_MODEL", "nvidia/parakeet-ctc-1.1b")
_SENTENCE_GAP_S = 0.6  # silence between words above this starts a new "sentence"

_lock = threading.Lock()
_cache: dict = {}


def _load(device: str):
    from transformers import AutoProcessor, AutoModelForCTC

    with _lock:
        if "model" not in _cache:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            proc = AutoProcessor.from_pretrained(_MODEL_ID)
            model = AutoModelForCTC.from_pretrained(_MODEL_ID, dtype=dtype).to(device)
            model.eval()
            blank = getattr(model.config, "blank_token_id", None)
            if blank is None:
                blank = getattr(model.config, "pad_token_id", None)
            if blank is None:
                blank = model.config.vocab_size - 1
            _cache.update(proc=proc, model=model, dtype=dtype, blank=int(blank),
                          device=device)
    return _cache


def transcribe_words(wav_path: str, device: str | None = None) -> pd.DataFrame:
    """Return word events: columns text, start, duration, sequence_id, sentence."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    c = _load(device)
    proc, model, dtype, blank = c["proc"], c["model"], c["dtype"], c["blank"]

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    duration = len(audio) / 16000.0
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    feats = inputs["input_features"].to(device, dtype)
    kw = {}
    if "attention_mask" in inputs:
        kw["attention_mask"] = inputs["attention_mask"].to(device)
    with torch.inference_mode():
        logits = model(feats, **kw).logits  # (1, T, V)
    pred = logits[0].float().argmax(-1).tolist()
    T = len(pred)
    frame_dur = duration / max(T, 1)

    # greedy CTC collapse -> (token_id, start_frame)
    toks, prev = [], blank
    for i, tid in enumerate(pred):
        if tid != prev and tid != blank:
            toks.append((tid, i))
        prev = tid
    if not toks:
        return pd.DataFrame(columns=["text", "start", "duration", "sequence_id",
                                     "sentence"])

    tokenizer = proc.tokenizer
    pieces = tokenizer.convert_ids_to_tokens([t for t, _ in toks])

    # group sub-word pieces (SentencePiece '▁' marks a word start) into words
    words = []  # (text, start_s, end_s)
    cur, cur_start = "", None
    for (tid, frame), piece in zip(toks, pieces):
        is_start = piece.startswith("▁")
        clean = piece.replace("▁", "")
        if is_start or cur == "":
            if cur:
                words.append((cur, cur_start, frame * frame_dur))
            cur, cur_start = clean, frame * frame_dur
        else:
            cur += clean
    if cur:
        words.append((cur, cur_start, duration))

    rows, seq, last_end = [], 0, None
    for text, start, end in words:
        if not text:
            continue
        if last_end is not None and start - last_end > _SENTENCE_GAP_S:
            seq += 1
        rows.append(dict(text=text, start=float(start),
                         duration=float(max(end - start, 1e-3)), sequence_id=seq))
        last_end = end

    df = pd.DataFrame(rows)
    # sentence = full text of each sequence_id group
    sent = df.groupby("sequence_id")["text"].apply(lambda s: " ".join(s)).to_dict()
    df["sentence"] = df["sequence_id"].map(sent)
    return df


if __name__ == "__main__":
    import sys
    out = transcribe_words(sys.argv[1])
    print(out.to_string())
