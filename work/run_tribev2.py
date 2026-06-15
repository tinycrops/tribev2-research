#!/usr/bin/env python
"""Run TRIBE v2 brain-encoding inference, tuned for the GB10 (Blackwell) GPU.

Usage:
    python run_tribev2.py                       # synthesizes a short test video
    python run_tribev2.py --video clip.mp4
    python run_tribev2.py --audio speech.wav
    python run_tribev2.py --text  story.txt
"""
import argparse
import time
from pathlib import Path

import torch


def enable_fast_math():
    """Optimal-speed knobs for Blackwell. TF32 + high matmul precision give a
    large speedup on the fp32 model with negligible accuracy impact."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def make_test_video(path: Path, seconds: int = 4) -> Path:
    """Synthesize a self-contained color clip with a silent audio track so the
    video + audio branches run without any network access."""
    import numpy as np
    from moviepy import ColorClip, AudioArrayClip

    fps, sr = 8, 16000
    clip = ColorClip(size=(256, 256), color=(64, 128, 192), duration=seconds).with_fps(fps)
    audio = AudioArrayClip(np.zeros((sr * seconds, 1), dtype=np.float32), fps=sr)
    clip = clip.with_audio(audio)
    clip.write_videofile(str(path), codec="libx264", audio_codec="aac", logger=None)
    return path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--video")
    g.add_argument("--audio")
    g.add_argument("--text")
    ap.add_argument("--repo", default="facebook/tribev2")
    ap.add_argument("--cache", default="./cache")
    args = ap.parse_args()

    enable_fast_math()
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)} "
              f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")

    from tribev2 import TribeModel

    t0 = time.time()
    model = TribeModel.from_pretrained(args.repo, cache_folder=args.cache, device="auto")
    print(f"model loaded in {time.time() - t0:.1f}s")

    kwargs = {}
    if args.video:
        kwargs["video_path"] = args.video
    elif args.audio:
        kwargs["audio_path"] = args.audio
    elif args.text:
        kwargs["text_path"] = args.text
    else:
        kwargs["video_path"] = str(make_test_video(Path(args.cache) / "test.mp4"))
    print(f"input: {kwargs}")

    t0 = time.time()
    df = model.get_events_dataframe(**kwargs)
    print(f"feature extraction in {time.time() - t0:.1f}s | {len(df)} events")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds, segments = model.predict(events=df)
    dt = time.time() - t0
    print(f"\n=== inference done in {dt:.2f}s ===")
    print(f"preds.shape = {preds.shape}  (n_timesteps, n_vertices)")
    print(f"n_segments  = {len(segments)}")
    if torch.cuda.is_available():
        print(f"peak GPU mem = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
