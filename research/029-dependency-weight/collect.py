"""研究029 収集：packages.ecosyste.ms の公開 lookup API（読み取りのみ、匿名）で、028 の items に含まれる各 repo の
依存数を取る。再開可能（data/deps_raw.jsonl に追記）。並列4。"""
import json, os, sys, time, urllib.request, urllib.parse, concurrent.futures as cf, threading
HERE = os.path.dirname(os.path.abspath(__file__))
P28 = f"{HERE}/../028-important-repo-dualwrite"
CEN = json.load(open(f"{P28}/data/census.json")); S25 = json.load(open(f"{HERE}/../025-q-xz-window/data/sample.json"))
names = [r["full_name"] for r in CEN["repos"] if r["status"] in ("ok", "walk_aborted")]
names += [r["full_name"] for r in S25["repos"] if r.get("status") == "ok" and r["stratum"] == "high" and r["bin"] != "20000-None"]
alias_rev = {v: k for k, v in CEN.get("alias", {}).items()}
OUT = f"{HERE}/data/deps_raw.jsonl"
done = set()
if os.path.exists(OUT):
    for l in open(OUT): d = json.loads(l); done.add(d["full_name"]) if d.get("status") == "ok" else None
lock = threading.Lock()
def fetch(url):
    for a in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "earth-survival-institute-research/029 (public read-only)"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read()), dict(r.headers)
        except Exception as e:
            err = repr(e); time.sleep(5 * (a + 1))
    raise RuntimeError(err)
def summarize(pk):
    eco = {}
    for p in pk:
        e = p.get("ecosystem"); d = p.get("dependent_repos_count") or 0
        eco[e] = max(eco.get(e, 0), d)
    top = sorted(pk, key=lambda p: -(p.get("dependent_repos_count") or 0))[:3]
    return {"n_packages_page1": len(pk), "D1_max_dep_repos": max([p.get("dependent_repos_count") or 0 for p in pk], default=0),
            "D2_sum_eco_max_dep_repos": sum(eco.values()),
            "D3_max_dep_packages": max([p.get("dependent_packages_count") or 0 for p in pk], default=0),
            "ecosystems": sorted(k for k in eco if k), "top": [{"ecosystem": p.get("ecosystem"), "name": p.get("name"),
            "dependent_repos_count": p.get("dependent_repos_count"), "dependent_packages_count": p.get("dependent_packages_count"),
            "repository_url": p.get("repository_url")} for p in top]}
def job(n):
    q = n if n not in alias_rev else alias_rev[n]
    url = ("https://packages.ecosyste.ms/api/v1/packages/lookup?" +
           urllib.parse.urlencode({"repository_url": f"https://github.com/{q}", "sort": "dependent_repos_count", "order": "desc"}))
    rec = {"full_name": n, "queried": q, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        pk, h = fetch(url); rec.update(status="ok", **summarize(pk)); rec["ratelimit_remaining"] = h.get("x-ratelimit-remaining")
    except Exception as e:
        rec.update(status="error", error=str(e)[:200])
    with lock:
        with open(OUT, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec
todo = [n for n in dict.fromkeys(names) if n not in done]
print("total", len(set(names)), "todo", len(todo), flush=True)
t0 = time.time(); k = 0
with cf.ThreadPoolExecutor(4) as ex:
    for rec in ex.map(job, todo):
        k += 1
        if k % 100 == 0: print(k, round(time.time() - t0), rec.get("ratelimit_remaining"), flush=True)
        if rec.get("ratelimit_remaining") and int(rec["ratelimit_remaining"]) < 200: print("near limit, stop"); break
print("finished", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), round(time.time() - t0))
