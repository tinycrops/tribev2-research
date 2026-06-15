import torch, traceback
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-0.6b",
               device=0, torch_dtype=torch.bfloat16)
mp3 = "/home/ath/tribev2-research/cache/parakeet_test.mp3"
for kw in [dict(return_timestamps="word"), dict(return_timestamps=True), dict()]:
    try:
        print("=== kwargs:", kw)
        out = asr(mp3, **kw)
        print("type:", type(out), "keys:", list(out.keys()) if isinstance(out, dict) else out)
        if isinstance(out, dict):
            print("text:", out.get("text"))
            ch = out.get("chunks")
            if ch: print("chunks[:6]:", ch[:6])
    except Exception:
        traceback.print_exc()
