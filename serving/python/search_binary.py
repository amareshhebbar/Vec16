import argparse
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

DOCS = [
    "Vaccines show the immune system a harmless piece of a pathogen so it reacts faster later.",
    "The Eiffel Tower was finished in 1889 for the World's Fair in Paris.",
    "Binary embeddings store one bit per dimension and are compared with Hamming distance.",
    "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen.",
    "A mortgage is a loan where the house itself is the security.",
    "Rust prevents data races at compile time through ownership and borrowing.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AmareshHebbar/Vec16_Qwen3_0.6B")
    ap.add_argument("--bits", type=int, default=128)
    ap.add_argument("query", nargs="?", default="how does a vaccine work")
    a = ap.parse_args()
    model = SentenceTransformer(a.model, trust_remote_code=True, truncate_dim=a.bits)
    codes = quantize_embeddings(model.encode_document(DOCS), precision="ubinary")
    index = faiss.IndexBinaryFlat(a.bits)
    index.add(codes)
    t0 = time.time()
    q = quantize_embeddings(model.encode_query([a.query]), precision="ubinary")
    dist, ids = index.search(q, len(DOCS))
    print(f"{a.bits} bits = {codes.shape[1]} bytes per document, index {codes.nbytes} bytes, search {(time.time() - t0) * 1000:.1f} ms")
    for d, i in zip(dist[0], ids[0]):
        print(f"hamming {int(d):4d}  {DOCS[i]}")
    full = model.encode_document(DOCS)
    print(f"float32 would need {full.shape[1] * 4} bytes per document, {np.asarray(full).nbytes} bytes total")


if __name__ == "__main__":
    main()
