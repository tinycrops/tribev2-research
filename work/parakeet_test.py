import torch, time
from pathlib import Path
from gtts import gTTS

txt = "The quick brown fox jumps over the lazy dog. Neuroscience studies the human brain."
mp3 = Path("/home/ath/tribev2-research/cache/parakeet_test.mp3")
mp3.parent.mkdir(parents=True, exist_ok=True)
gTTS(txt, lang="en").save(str(mp3))
print("wrote", mp3)

from transformers import pipeline
for model_id in ["nvidia/parakeet-ctc-1.1b", "nvidia/parakeet-ctc-0.6b"]:
    try:
        print("=== trying", model_id)
        t0=time.time()
        asr = pipeline("automatic-speech-recognition", model=model_id,
                       device=0, torch_dtype=torch.bfloat16)
        print("loaded in %.1fs" % (time.time()-t0))
        t0=time.time()
        out = asr(str(mp3), return_timestamps="word", chunk_length_s=30)
        print("transcribed in %.2fs" % (time.time()-t0))
        print("text:", out["text"])
        print("first chunks:", out.get("chunks", [])[:6])
        break
    except Exception as e:
        import traceback; traceback.print_exc()
        print("FAILED", model_id, "->", type(e).__name__, str(e)[:200])
