"""研究031 収集：030 の data/pkgs.jsonl・data/repos_gh.jsonl から、D1 ≥10,000 で高層の外にある repo
（未照会・星 <250・fork）を選び、GitHub GraphQL（gh の認証、読み取りのみ、50件ずつ）で
作成日・fork・星・2020-02-02 より前の最後のコミット日・最初の星の日を取る。出力：data/lowtier_gh.jsonl。再開可能。"""
import json, os, re, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); P30 = f"{HERE}/../030-dependency-tail-census"
OUT = f"{HERE}/data/lowtier_gh.jsonl"
GH = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$", re.I)
def repo_of(p):   # 030 と同じ規則
    m = GH.search(p.get("repository_url") or "")
    if m: return f"{m.group(1)}/{m.group(2)}"
    if p.get("rm_full_name") and (p.get("rm_host") or "").lower() == "github": return p["rm_full_name"]
    return None
agg = {}
for l in open(f"{P30}/data/pkgs.jsonl"):
    p = json.loads(l); r = repo_of(p)
    if not r: continue
    k = r.lower(); a = agg.setdefault(k, {"key": k, "D1_cen": 0, "top": None, "rm_size_kb": None})
    if p["dep_repos"] > a["D1_cen"]: a["D1_cen"] = p["dep_repos"]; a["top"] = f'{p["registry"]}:{p["name"]}'
    if p.get("rm_size_kb") is not None and a["rm_size_kb"] is None: a["rm_size_kb"] = p["rm_size_kb"]
gh30 = {json.loads(l)["key"]: json.loads(l) for l in open(f"{P30}/data/repos_gh.jsonl")}
cand = []
for k, a in agg.items():
    if a["D1_cen"] < 10000: continue
    if k not in gh30: cand.append({**a, "cls030": "not_queried"}); continue
    g = gh30[k]["gh"]
    if not g: continue
    if g["isFork"]: cand.append({**a, "cls030": "fork"})
    elif g["createdAt"] < "2020-02-02" and g["stargazerCount"] < 250: cand.append({**a, "cls030": "below250"})
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["key"])
todo = [a for a in cand if a["key"] not in done]
print("candidates", len(cand), "todo", len(todo), flush=True)
Q = ('repository(owner:%s,name:%s){nameWithOwner stargazerCount createdAt isFork diskUsage pushedAt '
     'defaultBranchRef{name target{... on Commit{history(first:1,until:"2020-02-02T00:00:00Z"){nodes{committedDate}}}}} '
     'stargazers(first:1,orderBy:{field:STARRED_AT,direction:ASC}){edges{starredAt}}}')
B = 50
for i in range(0, len(todo), B):
    ch = todo[i:i + B]
    body = " ".join(f"r{j}:" + Q % (json.dumps(a["key"].split("/")[0]), json.dumps(a["key"].split("/")[1])) for j, a in enumerate(ch))
    d = {"data": None}
    for att in range(4):
        pr = subprocess.run(["gh", "api", "graphql", "-f", "query={" + body + " rateLimit{remaining cost}}"], capture_output=True, text=True)
        try:
            d = json.loads(pr.stdout)
            if d.get("data"): break
        except Exception:
            d = {"data": None, "errors": [pr.stderr[:300]]}
        time.sleep(10 * (att + 1))
    data = (d.get("data") or {})
    t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT, "a") as f:
        for j, a in enumerate(ch):
            g = data.get(f"r{j}")
            if g:
                br = (g.get("defaultBranchRef") or {}); hist = (((br.get("target") or {}).get("history") or {}).get("nodes") or [])
                st = ((g.get("stargazers") or {}).get("edges") or [])
                g = {"nameWithOwner": g["nameWithOwner"], "stargazerCount": g["stargazerCount"], "createdAt": g["createdAt"],
                     "isFork": g["isFork"], "diskUsage": g["diskUsage"], "pushedAt": g["pushedAt"],
                     "branch": br.get("name"), "last_commit_pre2020": hist[0]["committedDate"] if hist else None,
                     "first_star": st[0]["starredAt"] if st else None}
            f.write(json.dumps({**a, "t": t, "gh": g, "status": "ok" if g else ("not_found" if data else "query_failed")}) + "\n")
    print(i, (data.get("rateLimit") or {}), flush=True)
