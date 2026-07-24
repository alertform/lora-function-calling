"""QLoRA SFT for function-calling, sized for a single 8 GB GPU.

    python src/train.py --config config/qlora.yaml

4-bit nf4 base (bitsandbytes) + LoRA adapters (peft) + TRL SFTTrainer. Only the
adapters train; the frozen 4-bit base is what lets a 7B model fit in 8 GB.
"""
import argparse, yaml, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/qlora.yaml")
    ap.add_argument("--train_file", default="data/train.jsonl")
    c = ap.parse_args()
    cfg = load_cfg(c.config)

    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if cfg.get("load_in_4bit"):
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if quant is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True)
        )

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files=c.train_file, split="train")

    sft = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["learning_rate"]),
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler"],
        weight_decay=cfg["weight_decay"],
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=cfg.get("bf16", True),
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        max_seq_length=cfg["max_seq_len"],
        packing=cfg.get("packing", True),
        seed=cfg["seed"],
        report_to="none",
        optim="paged_adamw_8bit",   # paged 8-bit optimizer — another 8 GB saver
    )

    trainer = SFTTrainer(model=model, args=sft, train_dataset=ds, peft_config=lora, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg["output_dir"])          # saves the LoRA adapter
    tok.save_pretrained(cfg["output_dir"])
    print(f"\n✅ adapter saved -> {cfg['output_dir']}")


if __name__ == "__main__":
    main()
