import json
import shutil
from pathlib import Path

from vec16.bidirectional import to_bidirectional
from vec16.common import PKG, device, model_kwargs, mrl_dims


def fix_tokenizer(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "right"
    return tok


def build_student(cfg, source, bidir, max_len, dtype=None):
    from sentence_transformers import SentenceTransformer, models

    tr = models.Transformer(
        str(source),
        max_seq_length=max_len,
        model_args=model_kwargs(dtype, "sdpa"),
        tokenizer_args={"padding_side": "right"},
    )
    fix_tokenizer(tr.tokenizer)
    if bidir:
        to_bidirectional(tr.auto_model)
    tr.auto_model.config.use_cache = False
    pool = models.Pooling(tr.get_word_embedding_dimension(), pooling_mode="mean")
    st = SentenceTransformer(modules=[tr, pool, models.Normalize()], device=device())
    st.prompts = {"query": cfg["prompts"]["query"], "document": cfg["prompts"]["document"]}
    return st


def load_trained(path, dtype=None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(path), device=device(), trust_remote_code=True, model_kwargs=model_kwargs(dtype))


def finalize_dir(cfg, mkey, path, bidir):
    path = Path(path)
    mcfg = cfg["models"][mkey]
    conf_p = path / "config.json"
    conf = json.loads(conf_p.read_text())
    full = int(conf.get("hidden_size"))
    conf["matryoshka_dimensions"] = mrl_dims(cfg, full)
    conf["use_cache"] = False
    if bidir:
        shutil.copy(PKG / "bidirectional.py", path / "bidirectional.py")
        native = conf["architectures"][0].replace("Vec16", "") if conf.get("architectures") else None
        cls = {"LlamaModel": "Vec16LlamaModel", "Qwen2Model": "Vec16Qwen2Model", "Qwen3Model": "Vec16Qwen3Model", "Gemma3TextModel": "Vec16Gemma3TextModel"}
        native = native if native in cls else _native_from_type(conf["model_type"])
        conf["auto_map"] = {"AutoModel": f"bidirectional.{cls[native]}"}
        conf["is_causal"] = False
        conf["architectures"] = [mcfg.get("vllm_arch", native)]
    else:
        conf.pop("auto_map", None)
        conf.pop("is_causal", None)
        conf["architectures"] = [mcfg.get("vllm_arch", _native_from_type(conf["model_type"]))]
    conf_p.write_text(json.dumps(conf, indent=2))
    st_conf = path / "config_sentence_transformers.json"
    if st_conf.exists():
        s = json.loads(st_conf.read_text())
        s["prompts"] = {"query": cfg["prompts"]["query"], "document": cfg["prompts"]["document"]}
        s["default_prompt_name"] = None
        st_conf.write_text(json.dumps(s, indent=2))


def _native_from_type(model_type):
    return {"llama": "LlamaModel", "qwen2": "Qwen2Model", "qwen3": "Qwen3Model", "gemma3_text": "Gemma3TextModel"}[model_type]
