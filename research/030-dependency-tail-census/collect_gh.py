"""研究030 収集2：data/pkgs.jsonl の package を GitHub repo にまとめ、GitHub GraphQL（gh の認証、読み取りのみ、100件ずつ）で
nameWithOwner・stargazerCount・createdAt・isFork・diskUsage を取る。対象は D1_cen ≥ 1,000 かつ repo_metadata の星が 150 以上
（または不明）の repo。出力：data/repos_gh.jsonl。再開可能。"""
import json, os, re, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/repos_gh.jsonl"
GH = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$", re.I)
def repo_of(p):
    u = p.get("repository_url") or ""
    m = GH.search(u)
    if m: return f"{m.group(1)}/{m.group(2)}"
    if p.get("rm_full_name") and (p.get("rm_host") or "").lower() == "github": return p["rm_full_name"]
    return None
agg = {}
for l in open(f"{HERE}/data/pkgs.jsonl"):
    p = json.loads(l); r = repo_of(p)
    if not r: continue
    k = r.lower(); a = agg.setdefault(k, {"key": k, "D1_cen": 0, "rm_stars": None, "top": None})
    if p["dep_repos"] > a["D1_cen"]: a["D1_cen"] = p["dep_repos"]; a["top"] = f'{p["registry"]}:{p["name"]}'
    if p.get("rm_stars") is not None: a["rm_stars"] = max(a["rm_stars"] or 0, p["rm_stars"])
cand = [a for a in agg.values() if a["D1_cen"] >= 1000 and (a["rm_stars"] is None or a["rm_stars"] >= 150)]
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["key"])
todo = [a for a in cand if a["key"] not in done]
print("repos", len(agg), "candidates", len(cand), "todo", len(todo), flush=True)
Q = "repository(owner:%s,name:%s){nameWithOwner stargazerCount createdAt isFork diskUsage}"
for i in range(0, len(todo), 100):
    ch = todo[i:i + 100]
    body = " ".join(f"r{j}:" + Q % (json.dumps(a["key"].split("/")[0]), json.dumps(a["key"].split("/")[1])) for j, a in enumerate(ch))
    for att in range(4):
        pr = subprocess.run(["gh", "api", "graphql", "-f", "query={" + body + " rateLimit{remaining}}"], capture_output=True, text=True)
        try:
            d = json.loads(pr.stdout); break
        except Exception:
            time.sleep(10 * (att + 1)); d = {"data": None, "errors": [pr.stderr[:300]]}
    data = (d.get("data") or {})
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT, "a") as f:
        for j, a in enumerate(ch):
            g = data.get(f"r{j}")
            f.write(json.dumps({**a, "t": t, "gh": g, "status": "ok" if g else "not_found"}) + "\n")
    print(i, (data.get("rateLimit") or {}).get("remaining"), flush=True)
