import argparse

import numpy as np
from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AmareshHebbar/Vec16_Qwen3_0.6B")
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--dims", type=int, default=128)
    a = ap.parse_args()
    client = OpenAI(base_url=a.url, api_key="none")
    docs = [
        "passage: Vaccines show the immune system a harmless piece of a pathogen so it reacts faster later.",
        "passage: The Eiffel Tower was finished in 1889 for the World's Fair in Paris.",
        "passage: Binary embeddings store one bit per dimension and are compared with Hamming distance.",
    ]
    query = "query: how do vaccines train the body"
    out = client.embeddings.create(model=a.model, input=[query] + docs, dimensions=a.dims)
    emb = np.array([d.embedding for d in out.data], dtype=np.float32)
    bits = np.packbits(emb > 0, axis=1)
    ham = [int(np.unpackbits(bits[0] ^ b).sum()) for b in bits[1:]]
    cos = emb[1:] @ emb[0] / (np.linalg.norm(emb[1:], axis=1) * np.linalg.norm(emb[0]))
    print(f"{a.dims} dims, {bits.shape[1]} bytes per document")
    for d, h, c in sorted(zip(docs, ham, cos), key=lambda x: x[1]):
        print(f"hamming {h:4d}  cosine {c:.3f}  {d[:80]}")


if __name__ == "__main__":
    main()
