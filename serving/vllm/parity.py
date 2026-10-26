import argparse

import numpy as np


TEXTS = [
    "query: how do vaccines train the body",
    "passage: Vaccines show the immune system a harmless piece of a pathogen so it reacts faster later.",
    "passage: Binary embeddings store one bit per dimension and are compared with Hamming distance.",
    "passage: short",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AmareshHebbar/Vec16_Qwen3_0.6B")
    ap.add_argument("--revision", default=None)
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(a.model, revision=a.revision, trust_remote_code=True)
    ref = st.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=True)
    del st

    from vllm import LLM

    llm = LLM(model=a.model, revision=a.revision, runner="pooling", max_model_len=512, enforce_eager=True)
    got = np.array([o.outputs.embedding for o in llm.embed(TEXTS)], dtype=np.float32)
    got = got / np.linalg.norm(got, axis=1, keepdims=True)
    cos = (ref * got).sum(1)
    for t, c in zip(TEXTS, cos):
        print(f"{c:.5f}  {t[:70]}")
    bits_match = float(np.mean((ref[:, :128] > 0) == (got[:, :128] > 0)))
    print(f"min cosine {cos.min():.5f}, first 128 sign bits equal {bits_match * 100:.2f}%")
    if cos.min() < 0.99:
        raise SystemExit("vLLM output differs from Sentence Transformers")
    print("vLLM matches Sentence Transformers")


if __name__ == "__main__":
    main()
