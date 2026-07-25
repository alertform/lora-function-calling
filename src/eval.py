"""Held-out eval: base model vs the LoRA-tuned model, same prompts.

Metrics (the numbers that go on the resume):
    valid_json   — fraction of replies that parse as a JSON array of calls
    name_acc     — fraction where the set of called tool NAMES matches gold exactly
    exact_match  — fraction where names AND arguments match gold exactly

    python src/eval.py --config config/qlora.yaml --adapter outputs/qwen2.5-7b-fc-qlora
"""
import argparse, json, re, yaml, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_cfg(path):
    # Explicit utf-8 — Windows would otherwise use the ANSI codepage (GBK on zh-CN).
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_calls(text):
    """Pull the first JSON array out of the model's reply and normalize it."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list):
        return None
    out = []
    for c in arr:
        if isinstance(c, dict) and c.get("name"):
            out.append({"name": c["name"], "arguments": c.get("arguments", {})})
    return out


def names(calls):
    return sorted(c["name"] for c in calls)


def exact(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@torch.no_grad()
def run(model, tok, rows, max_new=256):
    preds = []
    for r in rows:
        ids = tok.apply_chat_template(r["messages_prompt"], add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        preds.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    return preds


def score(preds, rows):
    n = len(rows)
    valid = name_ok = exact_ok = 0
    for p, r in zip(preds, rows):
        calls = parse_calls(p)
        if calls is None:
            continue
        valid += 1
        gold = r["gold"]
        if names(calls) == names(gold):
            name_ok += 1
            if exact(sorted(calls, key=lambda c: c["name"]), sorted(gold, key=lambda c: c["name"])):
                exact_ok += 1
    return {"valid_json": valid / n, "name_acc": name_ok / n, "exact_match": exact_ok / n, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/qlora.yaml")
    ap.add_argument("--adapter", default="outputs/qwen2.5-7b-fc-qlora")
    ap.add_argument("--eval_file", default="data/eval.jsonl")
    ap.add_argument("--out", default="eval_report.json")
    a = ap.parse_args()
    cfg = load_cfg(a.config)

    rows = [json.loads(l) for l in open(a.eval_file, encoding="utf-8")]
    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], quantization_config=quant, torch_dtype=torch.bfloat16, device_map="auto")
    base.eval()

    print("· scoring BASE ...")
    base_m = score(run(base, tok, rows), rows)

    print("· scoring TUNED ...")
    tuned = PeftModel.from_pretrained(base, a.adapter)
    tuned.eval()
    tuned_m = score(run(tuned, tok, rows), rows)

    report = {"model": cfg["model"], "adapter": a.adapter, "base": base_m, "tuned": tuned_m,
              "delta": {k: round(tuned_m[k] - base_m[k], 4) for k in ("valid_json", "name_acc", "exact_match")}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'metric':<12}{'base':>9}{'tuned':>9}{'Δ':>9}")
    for k in ("valid_json", "name_acc", "exact_match"):
        print(f"{k:<12}{base_m[k]:>9.3f}{tuned_m[k]:>9.3f}{tuned_m[k]-base_m[k]:>+9.3f}")
    print(f"\n📊 report -> {a.out}")


if __name__ == "__main__":
    main()
