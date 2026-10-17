"""研究037 collect.py：U（035）と ≥20,000 星（036）の repo を 030 の pkgs.jsonl で npm の package に対応づけ、
api.npmjs.org の last-month download（公開・読み取りのみ）を取る。bulk は scoped を受け付けないので scoped は1件ずつ（並列6、途中保存、再開可能）。出力 data/npm_dl.json。"""
import json, os, time, urllib.request, urllib.parse, collections
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(HERE)
OUT = f"{HERE}/data/npm_dl.json"; UA = "earth-survival-institute-research/037 (public read-only)"
U = json.load(open(f"{RES}/035-orphan-share-distro/data/match_trixie.json"))["U"]
S = json.load(open(f"{RES}/036-star-repo-packaging/data/match_star.json"))["star"]
want = {u["name"].lower() for u in U} | {s["name"].lower() for s in S}
P = collections.defaultdict(list)
for l in open(f"{RES}/030-dependency-tail-census/data/pkgs.jsonl"):
    d = json.loads(l)
    if d["registry"] == "npmjs.org" and (d["rm_full_name"] or "").lower() in want:
        P[d["rm_full_name"].lower()].append(d["name"])
names = sorted({n for v in P.values() for n in v})
cache = json.load(open(OUT)) if os.path.exists(OUT) else {"dl": {}, "log": []}
def get(url):
    for k in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            err = repr(e); time.sleep(3 * (k + 1))
    return {"__error__": err}
todo = [n for n in names if n not in cache["dl"]]
unsc = [n for n in todo if not n.startswith("@")]; sc = [n for n in todo if n.startswith("@")]
for i in range(0, len(unsc), 128):
    ch = unsc[i:i + 128]; d = get("https://api.npmjs.org/downloads/point/last-month/" + ",".join(urllib.parse.quote(n) for n in ch))
    for n in ch:
        x = d.get(n) if isinstance(d, dict) else None
        cache["dl"][n] = (x or {}).get("downloads") if x else None
    cache["log"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": len(ch), "err": d.get("__error__")})
    time.sleep(0.3)
json.dump(cache, open(OUT, "w"), indent=0)
import concurrent.futures as cf
def one(n):
    d = get("https://api.npmjs.org/downloads/point/last-month/" + urllib.parse.quote(n, safe="@")); time.sleep(0.2); return n, d.get("downloads")
with cf.ThreadPoolExecutor(6) as ex:
    for k, (n, x) in enumerate(ex.map(one, sc)):
        cache["dl"][n] = x
        if k % 500 == 499: json.dump(cache, open(OUT, "w"), indent=0)
tot = get("https://api.npmjs.org/downloads/point/last-month")
cache["total"] = tot; cache["repo_pkgs"] = {k: v for k, v in P.items()}
cache["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
json.dump(cache, open(OUT, "w"), indent=0)
print("repos", len(P), "pkgs", len(names), "scoped", len(sc), "missing", sum(1 for n in names if cache["dl"].get(n) is None), "total", tot)
