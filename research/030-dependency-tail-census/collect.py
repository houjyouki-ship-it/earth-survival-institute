"""研究030 収集1：packages.ecosyste.ms の公開 API（匿名・読み取りのみ）で、各 registry の dependent_repos_count 上位を
T_COLLECT 未満になるまで取る。出力：data/pkgs.jsonl（package ごと1行、必要な列のみ）、data/collect_log.jsonl。並列4・再開可能。"""
import json, os, time, urllib.request, urllib.parse, concurrent.futures as cf, threading
HERE = os.path.dirname(os.path.abspath(__file__))
T_COLLECT = 1000; PER = 1000; UA = "earth-survival-institute-research/030 (public read-only)"
REGS = json.load(open(f"{HERE}/sources/registries_list.json"))
OUT, LOG = f"{HERE}/data/pkgs.jsonl", f"{HERE}/data/collect_log.jsonl"
done = set()
if os.path.exists(LOG):
    for l in open(LOG):
        d = json.loads(l)
        if d.get("final"): done.add(d["registry"])
lock = threading.Lock()
def fetch(url):
    err = None
    for a in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read()), r.headers.get("x-ratelimit-remaining")
        except Exception as e:
            err = repr(e); time.sleep(10 * (a + 1))
    raise RuntimeError(err)
def slim(p):
    rm = p.get("repo_metadata") or {}
    return {"registry": None, "ecosystem": p.get("ecosystem"), "name": p.get("name"), "dep_repos": p.get("dependent_repos_count") or 0,
            "dep_pkgs": p.get("dependent_packages_count") or 0, "repository_url": p.get("repository_url"),
            "rm_full_name": rm.get("full_name"), "rm_host": (rm.get("host") or {}).get("name") if isinstance(rm.get("host"), dict) else rm.get("host"),
            "rm_stars": rm.get("stargazers_count"), "rm_created": rm.get("created_at"), "rm_fork": rm.get("fork"), "rm_size_kb": rm.get("size")}
def job(reg):
    name = reg["name"]; page = 1; n = 0; last = None
    while True:
        url = f"https://packages.ecosyste.ms/api/v1/registries/{urllib.parse.quote(name)}/packages?" + urllib.parse.urlencode(
            {"sort": "dependent_repos_count", "order": "desc", "per_page": PER, "page": page})
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            d, rl = fetch(url)
        except Exception as e:
            with lock: open(LOG, "a").write(json.dumps({"registry": name, "page": page, "t": t, "error": str(e), "final": True, "stopped_early": True, "last_value": last}) + "\n")
            return
        rows = [slim(p) for p in d]
        for r in rows: r["registry"] = name
        vals = [r["dep_repos"] for r in rows]
        last = vals[-1] if vals else last
        with lock:
            with open(OUT, "a") as f:
                for r in rows:
                    if r["dep_repos"] >= T_COLLECT: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += len(rows)
            fin = (not rows) or len(rows) < PER or (last is not None and last < T_COLLECT)
            open(LOG, "a").write(json.dumps({"registry": name, "page": page, "t": t, "n_rows": len(rows), "first": vals[0] if vals else None,
                                             "last_value": last, "ratelimit_remaining": rl, "final": fin}) + "\n")
        if fin: return
        page += 1
todo = [r for r in REGS if r["name"] not in done]
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(job, todo))
print("done", len(todo))
