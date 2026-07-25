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


def _json_objects(text):
    """Yield each top-level {...} block in `text` by brace matching.

    glaive puts its tool specs in the system prompt as bare, concatenated JSON
    objects (NOT a JSON array), so a regex like r'\\[.*\\]' grabs an inner
    "required": [...] instead of the specs. Brace matching is the only correct read.
    """
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                obj = _loads(text[start : i + 1])
                if obj:
                    out.append(obj)
                start = None
    return out


def from_glaive(row):
    """glaive-function-calling-v2 -> (tools, query, calls).

    system : "...following functions. Use them if required -\\n{spec}\\n{spec}"
    chat   : "USER: ...\\n\\nASSISTANT: [<functioncall> {json}] ... <|endoftext|>"
    Rows where the assistant declines (no functioncall) are kept with calls=[] —
    "no applicable tool" is a behaviour worth training, not noise.
    """
    tools = _json_objects(row.get("system", "") or "")
    chat = row.get("chat", "") or ""

    def users_in(text):
        return [
            u.strip()
            for u in re.findall(r"USER:\s*(.*?)(?=\n\n|ASSISTANT:|FUNCTION RESPONSE:|$)", text, re.S)
            if u.strip()
        ]

    fm = re.search(r"<functioncall>\s*(\{.*?\})\s*<\|endoftext\|>", chat, re.S)
    if not fm:
        us = users_in(chat)
        return tools, (us[0] if us else ""), []

    # glaive writes arguments as a SINGLE-QUOTED JSON string — invalid JSON, so
    # json.loads fails on the raw call. Re-quote it into a proper JSON string first.
    raw = re.sub(r"'(\{.*?\})'", lambda m: json.dumps(m.group(1)), fm.group(1), flags=re.S)
    obj = _loads(raw)
    calls = _norm_calls([obj]) if isinstance(obj, dict) else []

    # Conversations are multi-turn: the assistant may ask a clarifying question
    # before calling. Pair the call with the LAST user turn preceding it.
    us = users_in(chat[: fm.start()])
    return tools, (us[-1] if us else ""), calls


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
    ap.add_argument("--empty_ratio", type=float, default=0.2,
                    help="max share of examples whose target is [] (no applicable tool)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    picker = from_xlam if "xlam" in args.dataset.lower() else from_glaive

    # Keep "no applicable tool" examples, but cap them: glaive is ~dominated by
    # declines, and an unbalanced set teaches the model to always answer [].
    want = args.max_samples + args.eval_samples
    max_empty = int(want * args.empty_ratio)
    rows, n_empty = [], 0
    for row in ds:
        tools, query, calls = picker(row)
        if not query or not tools:
            continue
        if not calls:
            if n_empty >= max_empty:
                continue
            n_empty += 1
        rows.append((tools, query, calls))
        if len(rows) >= want:
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

    # Class balance matters: if almost every target is [], the model just learns to
    # always decline. Surface the ratio instead of discovering it after a training run.
    def with_calls(rs):
        return sum(1 for _, _, c in rs if c)

    tr_c, ev_c = with_calls(train_rows), with_calls(eval_rows)
    print(f"train={len(train_rows)} (with tool-calls: {tr_c}, {tr_c/max(1,len(train_rows)):.0%})")
    print(f"eval ={len(eval_rows)} (with tool-calls: {ev_c}, {ev_c/max(1,len(eval_rows)):.0%})")
    print(f"-> {args.out}/")


if __name__ == "__main__":
    main()
