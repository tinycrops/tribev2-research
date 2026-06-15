#!/usr/bin/env python
"""Reproduce the ascii-cat LoRA finetune (pookie3000/Llama-3.2-3B-ascii-cats-lora)
in clean bf16 PEFT form, then merge it into the base so TRIBE's text extractor can
load it via AutoModel and read hidden states.

The published adapter is GGUF-only (won't load for hidden-state extraction), so we
retrain on the *real* dataset (pookie3000/ascii-cats, 201 ASCII cats) for a controlled
A/B where the ONLY difference vs stock Llama-3.2-3B is the cat-art LoRA delta.
"""
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

BASE = "meta-llama/Llama-3.2-3B"
OUT_ADAPTER = Path(__file__).parent / "models" / "asciicat-lora-adapter"
OUT_MERGED = Path(__file__).parent / "models" / "llama32-3b-asciicat-merged"
MAXLEN = 1024
EPOCHS = 10

# Plain prompt template (base Llama-3.2-3B has no chat template). Mirrors the
# repo's task: prompt -> ASCII art completion.
def format_example(creature, ascii_art):
    return f"Here is an ASCII art picture of a {creature}:\n\n{ascii_art}"


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token

    ds = load_dataset("pookie3000/ascii-cats", split="train")
    print(f"dataset: {len(ds)} rows, creatures={set(ds['creature'])}")

    def tokenize(batch):
        texts = [format_example(c, a) + tok.eos_token
                 for c, a in zip(batch["creature"], batch["ascii"])]
        out = tok(texts, truncation=True, max_length=MAXLEN)
        return out

    tokated = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.config.use_cache = False
    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUT_ADAPTER.parent / "_trainer"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=EPOCHS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        bf16=True,
        save_strategy="no",
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(model=model, args=args, train_dataset=tokated,
                      data_collator=collator)
    trainer.train()

    OUT_ADAPTER.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_ADAPTER))
    tok.save_pretrained(str(OUT_ADAPTER))
    print("saved adapter ->", OUT_ADAPTER)

    # quick generation sanity check
    model.eval()
    prompt = "Here is an ASCII art picture of a cat:\n\n"
    ids = tok(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen = model.generate(**ids, max_new_tokens=200, do_sample=False)
    print("=== sample generation (finetuned) ===")
    print(tok.decode(gen[0][ids.input_ids.shape[1]:], skip_special_tokens=True))

    # merge LoRA into base and save a standalone model dir for TRIBE
    merged = model.merge_and_unload()
    OUT_MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(OUT_MERGED), safe_serialization=True)
    tok.save_pretrained(str(OUT_MERGED))
    print("saved merged model ->", OUT_MERGED)


if __name__ == "__main__":
    main()
