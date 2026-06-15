# TRIBE v2 environment — source this:  source env.sh
# GB10 (Blackwell sm_121) + torch 2.12 cu130, aarch64.
export TRIBE_ROOT=/home/ath/tribev2-research
export HF_HOME="$TRIBE_ROOT/hf_cache"          # ~/.cache/huggingface is root-owned, redirect here
export TOKENIZERS_PARALLELISM=false
# Optional: pick the GPU Parakeet ASR model used for transcription
# export TRIBE_PARAKEET_MODEL=nvidia/parakeet-ctc-1.1b   # (default; or parakeet-ctc-0.6b)
source "$TRIBE_ROOT/.venv/bin/activate"

# Run from a dir WITHOUT a ./tribev2 folder (avoids shadowing the package).
#   cd "$TRIBE_ROOT/work"
#   python run_tribev2.py                      # synthesized test clip
#   python run_tribev2.py --video myclip.mp4
#   python run_tribev2.py --text  story.txt
#   python demo_language_localizer.py          # in-silico language evaluation
#
# Text modality uses gated meta-llama/Llama-3.2-3B — HF_TOKEN must have access.
