"""研究030 探索（事前登録外）：裾の全数のうち依存数の上位1,000件と、残りから無作為150件（seed 20261030）について、
2020-02-02 以前の最後のコミットの木から TAR の大きさを 025 と同じ規則で再構成する（GitHub REST、gh 認証、読み取りのみ）。
出力：data/tar_measured.jsonl。再開可能。"""
import json, os, re, sys, math, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNTIL = "2020-02-02T00:00:00Z"; OUT = f"{HERE}/data/tar_measured.jsonl"
GHRE = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$", re.I)
agg = {}
for l in open(f"{HERE}/data/pkgs.jsonl"):
    p = json.loads(l); m = GHRE.search(p.get("repository_url") or "")
    r = f"{m.group(1)}/{m.group(2)}".lower() if m else ((p["rm_full_name"] or "").lower() if (p.get("rm_host") or "").lower() == "github" else None)
    if r: agg[r] = max(agg.get(r, 0), p["dep_repos"])
c = {}
for l in open(f"{HERE}/data/repos_gh.jsonl"):
    d = json.loads(l); g = d["gh"]
    if g and 250 <= g["stargazerCount"] < 20000 and g["createdAt"] < "2020-02-02" and not g["isFork"]:
        k = g["nameWithOwner"]; c[k] = max(c.get(k, 0), agg.get(d["key"], d["D1_cen"]))
tail = sorted([k for k, v in c.items() if v >= 10000], key=lambda k: -c[k])
top = tail[:1000]; random.seed(20261030); rnd = random.sample(tail[1000:], 150)
todo = [(k, "top1000") for k in top] + [(k, "random150") for k in rnd]
done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)["name"])
def api(path, params=None):
    cmd = ["gh", "api", "-X", "GET", path]
    for k, v in (params or {}).items(): cmd += ["-f", f"{k}={v}"]
    for a in range(3):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0: return json.loads(p.stdout)
        if "rate limit" in (p.stdout + p.stderr).lower(): time.sleep(60); continue
        return {"_error": (p.stdout + p.stderr)[:200]}
    return {"_error": "retries"}
def job(x):
    k, grp = x
    rec = {"name": k, "group": grp, "D1": c[k], "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cm = api(f"repos/{k}/commits", {"until": UNTIL, "per_page": 1})
    if not isinstance(cm, list) or not cm: rec["status"] = "no_commit_before_snapshot"; return rec
    t = api(f"repos/{k}/git/trees/{cm[0]['commit']['tree']['sha']}", {"recursive": 1})
    if "_error" in t: rec["status"] = "tree_error"; return rec
    blobs = [e.get("size", 0) for e in t.get("tree", []) if e.get("type") == "blob"]
    nd = sum(1 for e in t.get("tree", []) if e.get("type") == "tree")
    rec.update({"status": "ok", "commit_date": cm[0]["commit"]["committer"]["date"], "truncated": bool(t.get("truncated")),
                "n_files": len(blobs), "tar_bytes": sum(512 + 512 * math.ceil(s / 512) for s in blobs) + 512 * nd + 1024})
    return rec
with ThreadPoolExecutor(8) as ex, open(OUT, "a") as f:
    for r in ex.map(job, [x for x in todo if x[0] not in done]):
        f.write(json.dumps(r) + "\n"); f.flush()
print("done")
