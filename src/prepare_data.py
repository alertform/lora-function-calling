"""Download a function-calling dataset and normalize it into a single, eval-friendly
chat format:

    system   : tool specs (JSON) + a strict "reply with a JSON array of calls" contract
    user     : the natural-language query
    assistant: [{"name": ..., "arguments": {...}}, ...]   (target)

Both Salesforce/xlam-function-calling-60k and glaiveai/glaive-function-calling-v2
collapse into this shape, so training and eval never branch on the source dataset.

Outputs:
    data/train.jsonl   {"messages": [...]}                       -> SFTTrainer
    data/eval.jsonl    {"messages_prompt": [...], "gold": [...]}  -> eval.py
"""
import argparse, json, os, random, re

SYSTEM = (
    "You are a function-calling assistant. You are given a list of tools as JSON.\n"
    "Decide which tool(s) to call for the user's request and reply with ONLY a JSON "
    "array of calls, each {{\"name\": <tool>, \"arguments\": {{...}}}}. "
    "Reply with [] if no tool applies. No prose.\n\nTools:\n{tools}"
)


def _loads(x):
    """xLAM stores tools/answers as JSON strings; glaive as objects. Be tolerant."""
    if isinstance(x, (list, dict)):
        return x
    if not x:
        return []
    try:
        return json.loads(x)
    except Exception:
        return []


def _norm_calls(calls):
    out = []
    for c in calls or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or c.get("function") or ""
        args = c.get("arguments", c.get("args", {}))
        if isinstance(args, str):
            args = _loads(args) or {}
        if name:
            out.append({"name": name, "arguments": args})
    return out


def from_xlam(row):
    return _loads(row.get("tools")), row.get("query", ""), _norm_calls(_loads(row.get("answers")))


def from_glaive(row):
    # glaive: a "chat"/"system" transcript; pull the tools from system, the human turn,
    # and the first assistant function call. Best-effort — xLAM is the cleaner default.
    system = row.get("system", "") or ""
    tools = []
    m = re.search(r"\[.*\]", system, re.S)
    if m:
        tools = _loads(m.group(0))
    chat = row.get("chat", "") or ""
    q = ""
    hm = re.search(r"USER:\s*(.*?)\s*(ASSISTANT:|$)", chat, re.S)
    if hm:
        q = hm.group(1).strip()
    calls = []
    fm = re.search(r"<functioncall>\s*(\{.*?\})", chat, re.S)
    if fm:
        obj = _loads(fm.group(1))
        calls = _norm_calls([obj]) if obj else []
    return tools, q, calls


def build(tools, query, calls, tokenizer=None):
    sys = SYSTEM.format(tools=json.dumps(tools, ensure_ascii=False, indent=0))
    target = json.dumps(calls, ensure_ascii=False)
    prompt = [{"role": "system", "content": sys}, {"role": "user", "content": query}]
    full = prompt + [{"role": "assistant", "content": target}]
    return prompt, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Salesforce/xlam-function-calling-60k")
    ap.add_argument("--max_samples", type=int, default=8000)
    ap.add_argument("--eval_samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    picker = from_xlam if "xlam" in args.dataset.lower() else from_glaive

    rows = []
    for row in ds:
        tools, query, calls = picker(row)
        if not query or not tools:
            continue
        rows.append((tools, query, calls))
        if len(rows) >= args.max_samples + args.eval_samples:
            break

    random.Random(args.seed).shuffle(rows)
    eval_rows = rows[: args.eval_samples]
    train_rows = rows[args.eval_samples :]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for tools, q, calls in train_rows:
            _, full = build(tools, q, calls)
            f.write(json.dumps({"messages": full}, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "eval.jsonl"), "w", encoding="utf-8") as f:
        for tools, q, calls in eval_rows:
            prompt, _ = build(tools, q, calls)
            f.write(json.dumps({"messages_prompt": prompt, "gold": calls}, ensure_ascii=False) + "\n")

    print(f"train={len(train_rows)}  eval={len(eval_rows)}  -> {args.out}/")


if __name__ == "__main__":
    main()
