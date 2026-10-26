import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def check(path):
    conf = json.loads((path / "config.json").read_text())
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True).eval()
    a = tok(["alpha beta gamma delta"], return_tensors="pt")
    b = tok(["alpha beta gamma river"], return_tensors="pt")
    pad = tok(["alpha beta", "alpha beta gamma delta"], padding=True, return_tensors="pt")
    one = tok(["alpha beta"], return_tensors="pt")
    with torch.no_grad():
        ha = model(**a).last_hidden_state[0, 0]
        hb = model(**b).last_hidden_state[0, 0]
        hp = model(**pad).last_hidden_state[0, : one["input_ids"].shape[1]]
        h1 = model(**one).last_hidden_state[0]
    bidir = "auto_map" in conf
    sees_right = not torch.allclose(ha, hb, atol=1e-6)
    pad_ok = torch.allclose(hp, h1, atol=1e-4)
    ok = sees_right == bidir and pad_ok
    print(f"{'ok ' if ok else 'BAD'} {path.parent.parent.name}/{path.parent.name} class={type(model).__name__} bidirectional={bidir} sees_right_context={sees_right} padding_invariant={pad_ok}")
    return ok


def main():
    runs = Path(sys.argv[1])
    paths = sorted(runs.glob("*/*__s0/model"))
    results = [check(p) for p in paths]
    if not results or not all(results):
        raise SystemExit("bidirectional check failed")


if __name__ == "__main__":
    main()
