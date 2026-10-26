from vec16.common import load_cfg, read_json, status_path


def main():
    cfg = load_cfg()
    p = status_path(cfg)
    if not p.exists():
        print("nothing has run yet in", cfg["paths"]["work"])
        return
    st = read_json(p)
    print(f"work {cfg['paths']['work']}  updated {st.get('updated')}")
    for key in ("env", "data", "teacher_dense", "teacher_colbert", "baselines", "dataset", "finalize"):
        if key in st:
            print(f"{key:16} {st[key]}")
    models = st.get("models", {})
    if models:
        print()
        print(f"{'model':14} {'state':34} {'verdict':8} {'float':>6} {'binary':>6} {'base':>6}  repo")
        for m in cfg["run"]["models"]:
            r = models.get(m)
            if not r:
                print(f"{m:14} {'waiting':34}")
                continue
            f = lambda k: f"{r[k] * 100:6.1f}" if isinstance(r.get(k), (int, float)) else f"{'':>6}"
            print(f"{m:14} {str(r.get('state', ''))[:34]:34} {r.get('verdict', ''):8} {f('float')} {f('binary')} {f('base_float')}  {r.get('repo', '')}")


if __name__ == "__main__":
    main()
