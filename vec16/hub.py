import time

from vec16.common import hf_token, log, publish_enabled, read_json, write_json


def api():
    if not publish_enabled():
        return None
    token = hf_token()
    if not token:
        log("no HF token, skipping upload")
        return None
    from huggingface_hub import HfApi

    return HfApi(token=token)


def retry(fn, *args, tries=4, **kwargs):
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as ex:
            if i == tries - 1:
                raise
            log(f"hub call failed ({type(ex).__name__}: {ex}), retry {i + 1}")
            time.sleep(10 * (i + 1))


def collection(cfg, a):
    proj, run = cfg["project"], cfg["run"]
    title = proj["collection_title"] + (" smoke" if run["repo_suffix"] else "")
    coll = retry(
        a.create_collection,
        title=title,
        namespace=proj["hf_user"],
        description=proj["collection_description"][:150],
        private=bool(run["private"]),
        exists_ok=True,
    )
    p = cfg["paths"]["results"] / "collection.json"
    write_json(p, {"slug": coll.slug, "title": title})
    return coll.slug


def add_to_collection(cfg, a, item_id, kind, note):
    slug = collection(cfg, a)
    try:
        retry(a.add_collection_item, slug, item_id=item_id, item_type=kind, note=note[:500], exists_ok=True)
    except Exception as ex:
        log(f"could not add {item_id} to collection: {ex}")
        return slug
    try:
        retry(a.update_collection_item, slug, item_object_id=_item_object_id(a, slug, item_id), note=note[:500])
    except Exception:
        pass
    return slug


def _item_object_id(a, slug, item_id):
    coll = a.get_collection(slug)
    for it in coll.items:
        if it.item_id == item_id:
            return it.item_object_id
    raise KeyError(item_id)


def push_folder(a, repo, folder, kind="model", private=True, branch=None, message="Upload"):
    retry(a.create_repo, repo, repo_type=kind, private=private, exist_ok=True)
    if branch and branch != "main":
        retry(a.create_branch, repo, branch=branch, repo_type=kind, exist_ok=True)
    retry(a.upload_folder, folder_path=str(folder), repo_id=repo, repo_type=kind, revision=branch or "main", commit_message=message)


def set_private(a, repo, private, kind="model"):
    try:
        retry(a.update_repo_settings, repo, private=private, repo_type=kind)
    except AttributeError:
        retry(a.update_repo_visibility, repo, private=private, repo_type=kind)


def collection_slug(cfg):
    p = cfg["paths"]["results"] / "collection.json"
    return read_json(p)["slug"] if p.exists() else None
