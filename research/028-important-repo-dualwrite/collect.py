"""研究028 データ収集：≥20,000 星（2020-02-02 より前に作成、非 fork）の全数と Featured 15件について、
スナップショット時点の木から TAR の大きさを再構成する（025 の collect.py と同じ式。GitHub 公開 API の読み取りのみ）。
出力 data/census.json（以後の解析の基点）。"""
import json, subprocess, time, math, os, collections
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/census.json"
UNTIL = "2020-02-02T00:00:00Z"
FEATURED = ["dotnet/core", "torvalds/linux", "python/cpython", "bitcoin/bitcoin", "rails/rails", "docker/machine",
            "openssl/openssl", "nodejs/node", "Homebrew/brew", "php/php-src", "twbs/bootstrap", "microsoft/TypeScript",
            "apache/hadoop", "Alamofire/Alamofire", "v8/v8"]
last_search = [0.0]
def api(path, params=None, search=False):
    if search:
        dt = time.time() - last_search[0]
        if dt < 2.2: time.sleep(2.2 - dt)
        last_search[0] = time.time()
    cmd = ["gh", "api", "-X", "GET", path]
    for k, v in (params or {}).items(): cmd += ["-f", f"{k}={v}"]
    for attempt in range(4):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0: return json.loads(p.stdout)
        if "rate limit" in (p.stdout + p.stderr).lower(): time.sleep(30); continue
        return {"_error": (p.stdout + p.stderr)[:300]}
    return {"_error": "retries"}


def tree_entries(full, tsha, depth=0):
    t = api(f"repos/{full}/git/trees/{tsha}", {"recursive": 1})
    if "_error" in t: return None, 0, True
    if not t.get("truncated"):
        return [e for e in t["tree"] if e["type"] == "blob"], sum(1 for e in t["tree"] if e["type"] == "tree"), False
    if depth >= 6: return [e for e in t["tree"] if e["type"] == "blob"], sum(1 for e in t["tree"] if e["type"] == "tree"), True
    t1 = api(f"repos/{full}/git/trees/{tsha}")
    if "_error" in t1: return None, 0, True
    blobs = [e for e in t1["tree"] if e["type"] == "blob"]; ndirs = 0; trunc = bool(t1.get("truncated"))
    for e in t1["tree"]:
        if e["type"] == "tree":
            ndirs += 1
            b, nd, tr = tree_entries(full, e["sha"], depth + 1)
            if b is None: return None, 0, True
            blobs += b; ndirs += nd; trunc |= tr
    return blobs, ndirs, trunc

def tree_stats(full, branch):
    c = api(f"repos/{full}/commits", {"sha": branch, "until": UNTIL, "per_page": 1})
    if not isinstance(c, list) or not c: return {"status": "no_commit_before_snapshot"}
    sha = c[0]["sha"]; date = c[0]["commit"]["committer"]["date"]; tsha = c[0]["commit"]["tree"]["sha"]
    blobs, n_dirs, trunc = tree_entries(full, tsha)
    if blobs is None: return {"status": "tree_error"}
    sizes = [e.get("size", 0) for e in blobs]
    tar = sum(512 + 512 * math.ceil(s / 512) for s in sizes) + 512 * n_dirs + 1024
    return {"status": "ok", "commit": sha, "commit_date": date, "truncated": trunc, "n_files": len(sizes), "n_dirs": n_dirs,
            "raw_bytes": sum(sizes), "tar_bytes": tar, "max_file": max(sizes) if sizes else 0}

def search_all():
    items = {}
    q = "stars:>=20000 created:<2020-02-02 fork:false"
    tot = None
    for pg in range(1, 11):
        r = api("search/repositories", {"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": pg}, search=True)
        if "_error" in r: print("search err", r["_error"]); break
        tot = r["total_count"]
        for it in r["items"]: items[it["full_name"]] = it
        if len(r["items"]) < 100: break
    if tot and tot > len(items):
        for pg in range(1, 11):
            r = api("search/repositories", {"q": q, "sort": "stars", "order": "asc", "per_page": 100, "page": pg}, search=True)
            if "_error" in r: break
            for it in r["items"]: items[it["full_name"]] = it
            if len(items) >= tot or len(r["items"]) < 100: break
    return tot, list(items.values())

def main():
    res = {"until": UNTIL, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "featured": FEATURED}
    tot, items = search_all()
    res["search_total_count"] = tot; res["n_listed"] = len(items)
    print("listed", len(items), "of", tot, flush=True)
    names = {it["full_name"] for it in items}
    for f in FEATURED:
        if f not in names:
            it = api(f"repos/{f}")
            if "_error" not in it: items.append(it)
            else: print("featured missing", f, it["_error"])
    def rec(it):
        d = {"full_name": it["full_name"], "stars_now": it["stargazers_count"], "created_at": it["created_at"],
             "api_size_kb_now": it["size"], "fork": it.get("fork"), "featured": it["full_name"] in FEATURED,
             **tree_stats(it["full_name"], it["default_branch"])}
        print(d["full_name"], d["stars_now"], d.get("status"), d.get("tar_bytes"), d.get("truncated"), flush=True)
        return d
    with ThreadPoolExecutor(8) as pool:
        res["repos"] = list(pool.map(rec, items))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(OUT, "w"), indent=1)
main()
