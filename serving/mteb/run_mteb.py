import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="AmareshHebbar/Vec16_Qwen3_0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--benchmark", default="MTEB(eng, v2)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--truncate-dim", type=int, default=None)
    ap.add_argument("--out", default="mteb_results")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import mteb

    model = SentenceTransformer(a.model, revision=a.revision, trust_remote_code=True, truncate_dim=a.truncate_dim)
    tasks = mteb.get_tasks(tasks=a.tasks) if a.tasks else mteb.get_benchmark(a.benchmark).tasks
    out = Path(a.out)
    try:
        res = mteb.MTEB(tasks=tasks).run(model, output_folder=str(out), encode_kwargs={"batch_size": a.batch})
    except (AttributeError, TypeError):
        res = mteb.evaluate(model, tasks, encode_kwargs={"batch_size": a.batch})
        print("results saved in the mteb result cache, see mteb documentation for the folder")
    print(res)
    print(f"results in {out.resolve()}")
    print("to submit, open a pull request with this folder to https://github.com/embeddings-benchmark/results")


if __name__ == "__main__":
    main()
