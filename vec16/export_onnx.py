import argparse
from pathlib import Path

import numpy as np
import torch

from vec16.common import load_cfg, log, write_json

SAMPLES = [
    "query: how many bytes does one document need",
    "passage: Each document is stored as 128 bits, which is 16 bytes, and still retrieves well.",
    "passage: The quick brown fox jumps over the lazy dog.",
]


class Wrap(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, input_ids, attention_mask):
        out = self.st({"input_ids": input_ids, "attention_mask": attention_mask})
        return out["sentence_embedding"], out["token_embeddings"]


def export(model_path, out_dir):
    from sentence_transformers import SentenceTransformer

    model_path, out_dir = Path(model_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = SentenceTransformer(str(model_path), device="cpu", trust_remote_code=True, model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"})
    st.eval()
    wrap = Wrap(st).eval()
    feats = st.tokenize(SAMPLES)
    args = (feats["input_ids"], feats["attention_mask"])
    target = out_dir / "model.onnx"
    names = dict(input_names=["input_ids", "attention_mask"], output_names=["sentence_embedding", "token_embeddings"])
    axes = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "sequence"}, "sentence_embedding": {0: "batch"}, "token_embeddings": {0: "batch", 1: "sequence"}}
    with torch.inference_mode():
        ref = wrap(*args)[0].numpy()
        try:
            torch.onnx.export(wrap, args, str(target), opset_version=17, dynamic_axes=axes, do_constant_folding=True, dynamo=False, **names)
        except TypeError:
            torch.onnx.export(wrap, args, str(target), opset_version=17, dynamic_axes=axes, do_constant_folding=True, **names)
    parity = check(target, feats, ref)
    log(f"onnx fp32 parity cosine {parity:.6f}")
    info = {"fp32": {"file": "onnx/model.onnx", "parity_cosine": parity}}
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        qt = out_dir / "model_quantized.onnx"
        quantize_dynamic(str(target), str(qt), weight_type=QuantType.QInt8)
        qp = check(qt, feats, ref)
        info["int8"] = {"file": "onnx/model_quantized.onnx", "parity_cosine": qp}
        log(f"onnx int8 parity cosine {qp:.6f}")
    except Exception as ex:
        log(f"int8 quantization skipped: {type(ex).__name__}: {ex}")
    write_json(out_dir / "export.json", info)
    return info


def check(path, feats, ref):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["sentence_embedding"], {"input_ids": feats["input_ids"].numpy().astype(np.int64), "attention_mask": feats["attention_mask"].numpy().astype(np.int64)})[0]
    a = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    b = got / np.linalg.norm(got, axis=1, keepdims=True)
    return float((a * b).sum(1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    load_cfg()
    export(a.model_path, a.out_dir)


if __name__ == "__main__":
    main()
