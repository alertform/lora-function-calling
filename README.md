# lora-function-calling

QLoRA fine-tuning of a 7B model into a **function-calling / tool-use specialist**, on a **single 8 GB consumer GPU** (RTX 4060), with a held-out eval that reports the numbers that matter for agents: does the model emit *valid, correctly-named, correctly-argumented* tool calls?

```
query + tool specs ──► model ──► [{"name": "...", "arguments": {...}}, ...]
```

## Why

Agent frameworks live or die on tool-calling reliability. Rather than prompt-engineer around a base model, this project **teaches** the behaviour with QLoRA and then *measures* the lift on held-out data — turning "it usually calls the right tool" into a number.

## Method

- **Base**: `Qwen2.5-7B-Instruct`, loaded in **4-bit nf4** (bitsandbytes) so a 7B model + training state fits in 8 GB.
- **Adapters**: LoRA (r=16, α=32) on all attention + MLP projections; only the adapters train (~0.5% of params).
- **Memory budget on 8 GB**: 4-bit base + gradient checkpointing + paged 8-bit optimizer + seq-packing. Effective batch 16 via grad-accum.
- **Data**: `Salesforce/xlam-function-calling-60k`, normalized to one schema (`prepare_data.py`) so training/eval never branch on the source dataset — swap in `glaiveai/glaive-function-calling-v2` with one flag.
- **Trainer**: TRL `SFTTrainer` + PEFT.

## Results

Held-out set (300 examples from `glaiveai/glaive-function-calling-v2`), base vs. LoRA-tuned,
greedy decoding, 1 epoch on 8000 training examples:

| metric        | base  | tuned | Δ      |
|---------------|-------|-------|--------|
| valid_json    | 0.973 | 1.000 | +0.027 |
| name_acc      | 0.963 | 0.993 | +0.030 |
| exact_match   | 0.897 | 0.943 | +0.047 |

- **valid_json** — reply parses as a JSON array of calls
- **name_acc** — the set of called tool *names* matches gold exactly
- **exact_match** — names **and** arguments match gold exactly

Reading the numbers honestly: `Qwen2.5-7B-Instruct` is already a strong tool-caller, so the
headroom is thin — the win is that malformed output goes to **zero** (0.973 → 1.000, i.e. the
8 replies the base model wrapped in prose or truncated all become parseable), and roughly
**45% of the remaining argument errors** are eliminated (10.3% → 5.7% non-exact). 19% of the
held-out set has an empty gold target (`[]`, no applicable tool), which both models handle
well; the lift is concentrated in the argument-filling cases.

_(filled by `eval.py`, which writes `eval_report.json`)_

## Reproduce

```powershell
# RTX 4060 / Windows — stands up an isolated Python 3.11 + CUDA env
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

. .\scripts\env.ps1          # HF cache off C:\ + proxy vars (see note below)

.\.venv\Scripts\python.exe src\prepare_data.py --dataset glaiveai/glaive-function-calling-v2
.\.venv\Scripts\python.exe src\train.py        --config config\qlora.yaml
.\.venv\Scripts\python.exe src\eval.py         --config config\qlora.yaml --adapter outputs\qwen2.5-7b-fc-qlora
```

> **Downloads failing with `SSLEOFError` while the browser works?** PowerShell uses the
> WinINET system proxy; `requests`/`huggingface_hub` only read `HTTP_PROXY`/`HTTPS_PROXY`.
> `scripts\env.ps1` copies the system proxy into those vars. No proxy available? Set
> `$env:HF_ENDPOINT = "https://hf-mirror.com"`.

> **OOM on 8 GB?** Drop `model` in `config/qlora.yaml` to `Qwen/Qwen2.5-3B-Instruct` (or `1.5B`),
> and/or lower `max_seq_len`. Everything else stays the same.

`Salesforce/xlam-function-calling-60k` is gated — accept the terms once on Hugging Face and run
`huggingface-cli login`, or switch to the open `glaiveai/glaive-function-calling-v2`.

## Layout

```
config/qlora.yaml     one config: model, LoRA, 4-bit, training
src/prepare_data.py   dataset -> unified {system(tools), user(query), assistant(calls)}
src/train.py          QLoRA SFT (TRL SFTTrainer + PEFT), saves the adapter
src/eval.py           base vs tuned on held-out, writes eval_report.json
scripts/setup_env.ps1 uv-based Python 3.11 + CUDA torch + deps
```

## License

MIT
