import json
import random
import sys
from pathlib import Path

import yaml

WORDS = "alpha beta gamma delta river mountain city vaccine immune protein market price bank energy solar wind ocean forest music guitar piano engine car train rocket planet star galaxy code python model data search index vector byte memory storage cloud server network".split()


def sentence(rng, n=None):
    return " ".join(rng.choice(WORDS) for _ in range(n or rng.randint(6, 18))) + "."


def tokenizer(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    rng = random.Random(0)
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<eos>", "<mask>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([sentence(rng) for _ in range(3000)] + ["query: passage: _"], tr)
    from tokenizers import decoders

    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", eos_token="<eos>", mask_token="<mask>")
    fast.model_max_length = 512
    fast.save_pretrained(root)
    return fast


def bert_tokenizer(root):
    from transformers import BertTokenizerFast

    root.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(set(WORDS)) + [".", "query", ":", "passage", "?"] + [f"[unused{i}]" for i in range(10)]
    (root / "vocab.txt").write_text("\n".join(vocab))
    t = BertTokenizerFast(vocab_file=str(root / "vocab.txt"))
    t.model_max_length = 512
    t.save_pretrained(root)
    return t, len(vocab)


def decoder(root, kind, tok):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    common = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=True)
    if kind == "llama":
        cfg = AutoConfig.for_model("llama", **common)
    elif kind == "qwen2":
        cfg = AutoConfig.for_model("qwen2", **common)
    elif kind == "qwen3":
        cfg = AutoConfig.for_model("qwen3", head_dim=16, **common)
    elif kind == "gemma3":
        cfg = AutoConfig.for_model("gemma3_text", head_dim=16, sliding_window=64, **common)
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(root)
    tok.save_pretrained(root)


def bert_st(root, out_dim=None, colbert=False):
    import torch
    from transformers import BertConfig, BertModel

    torch.manual_seed(1)
    tok, n = bert_tokenizer(root)
    cfg = BertConfig(vocab_size=n, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, max_position_embeddings=512)
    BertModel(cfg).save_pretrained(root)
    if colbert:
        return
    from sentence_transformers import SentenceTransformer, models

    tr = models.Transformer(str(root))
    st = SentenceTransformer(modules=[tr, models.Pooling(64, "mean"), models.Normalize()])
    st.save(str(root))


def data(root):
    rng = random.Random(1)
    rows = []
    for _ in range(1200):
        topic = rng.sample(WORDS, 3)
        q = "what about " + " ".join(topic) + "?"
        p = " ".join(topic) + " " + sentence(rng)
        n = " ".join(rng.sample(WORDS, 3)) + " " + sentence(rng)
        rows.append({"query": q, "positive": p, "negative": n})
    f1 = root / "pairs_a.jsonl"
    f2 = root / "pairs_b.jsonl"
    f1.write_text("\n".join(json.dumps(r) for r in rows[:700]))
    f2.write_text("\n".join(json.dumps({"question": r["query"], "answer": r["positive"]}) for r in rows[700:]))
    mntp = root / "mntp.jsonl"
    mntp.write_text("\n".join(json.dumps({"text": " ".join(sentence(rng) for _ in range(30))}) for _ in range(200)))
    return [str(f1), str(f2)], str(mntp)


def nano(work, names):
    rng = random.Random(2)
    d = work / "small" / "cache" / "nanobeir"
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        docs, queries, qrels = [], [], []
        for i in range(60):
            topic = rng.sample(WORDS, 3)
            docs.append(" ".join(topic) + " " + sentence(rng))
            if i < 20:
                queries.append("about " + " ".join(topic))
                qrels.append([i])
        (d / f"{name}.json").write_text(json.dumps({"queries": queries, "docs": docs, "qrels": qrels}))


def main():
    root = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(root / "tok")
    tok.pad_token = "<pad>"
    for kind in ("llama", "qwen2", "qwen3", "gemma3"):
        decoder(root / f"tiny_{kind}", kind, tok)
    bert_st(root / "teacher_dense")
    bert_st(root / "teacher_colbert", colbert=True)
    bert_st(root / "baseline_a")
    files, mntp = data(root)
    names = ["scifact", "nfcorpus"]
    nano(work, names)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "vec16" / "config.yaml").read_text())
    cfg["sources"] = [
        {"key": "local_triplets", "files": [files[0]], "share": 0.6},
        {"key": "local_pairs", "files": [files[1]], "share": 0.4},
        {"key": "broken", "files": [str(root / "missing.jsonl")], "share": 0.1},
    ]
    cfg["mntp_sources"] = [{"files": [mntp], "field": "text"}]
    cfg["teachers"]["colbert"] = str(root / "teacher_colbert")
    cfg["teachers"]["dense_query_prompt"] = "query: "
    for kind, key in (("llama", "smollm2_135m"), ("qwen2", "qwen25_05b"), ("qwen3", "qwen3_06b"), ("gemma3", "gemma3_270m")):
        cfg["models"][key]["id"] = str(root / f"tiny_{kind}")
        cfg["models"][key]["gated"] = False
    cfg["baselines"] = {"tiny_bert": {"id": str(root / "baseline_a"), "label": "tiny bert"}}
    small = cfg["profiles"]["small"]
    small.update(
        queries=400,
        dev_queries=60,
        teacher_dense=str(root / "teacher_dense"),
        teacher_k=8,
        teacher_batch=32,
        colbert_chunk=64,
        mntp_docs=100,
        mntp_tokens=20000,
        mntp_seq=64,
        mntp_batch=4,
        models=["smollm2_135m", "qwen3_06b", "qwen25_05b", "gemma3_270m"],
        ablation_models=["qwen3_06b"],
        baselines=["tiny_bert"],
        reference="tiny_bert",
        seeds=[0, 1],
        train_k=4,
        batch_size=8,
        eval_steps=20,
        save_steps=25,
        dev_max=60,
        eval_batch=32,
        nanobeir=names,
    )
    cfg["train"]["matryoshka_dims"] = [64, 32, 16]
    cfg["train"]["binary_dims"] = [64, 32]
    cfg["verdict_bits"] = 32
    out = root / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(out)


if __name__ == "__main__":
    main()
