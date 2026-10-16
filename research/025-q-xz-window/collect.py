"""研究025 データ収集：2020-02-02 スナップショット時点の repo の木の大きさ（GitHub 公開 API、読み取りのみ）。
出力 data/sample.json（以後の解析はこのファイルだけを読む。GitHub 側は変わるので、このファイルが再現の基点）。
seed=20261025。gh CLI（認証済み）で公開 API を読むだけ。"""
import json, subprocess, time, random, math, os, sys
random.seed(20261025)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/sample.json"
UNTIL = "2020-02-02T00:00:00Z"
BIN_EXT = set("png jpg jpeg gif bmp ico tif tiff psd pdf zip gz tgz bz2 xz 7z rar jar war class so dll exe dylib a o lib bin dat db sqlite sqlite3 mp3 mp4 wav ogg flac avi mov mkv webm woff woff2 ttf otf eot pyc pkl npy npz h5 hdf5 onnx pt pth ckpt pb tflite apk ipa dmg iso img deb rpm msi swf blend fbx unitypackage xlsx xls docx doc pptx ppt keystore jks wasm node".split())
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
        if p.returncode == 0:
            return json.loads(p.stdout)
        if "rate limit" in (p.stdout + p.stderr).lower():
            time.sleep(30); continue
        return {"_error": (p.stdout + p.stderr)[:300]}
    return {"_error": "retries"}

def months(y0=2008, y1=2019):
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]

def pick_search(q, k=3):
    """ランダムなページから最大 k 件を非復元で選ぶ（第25回の逸脱：速度のため 1→最大3件/検索）"""
    r = api("search/repositories", {"q": q, "per_page": 100}, search=True)
    if "_error" in r or not r.get("items"): return [], r.get("total_count", 0) if isinstance(r, dict) else 0
    tot = r["total_count"]; npg = min(10, math.ceil(tot / 100))
    pg = random.randint(1, npg)
    if pg > 1:
        r = api("search/repositories", {"q": q, "per_page": 100, "page": pg}, search=True)
        if "_error" in r or not r.get("items"): return [], tot
    return random.sample(r["items"], min(k, len(r["items"]))), tot

def tree_stats(full, branch, stars):
    c = api(f"repos/{full}/commits", {"sha": branch, "until": UNTIL, "per_page": 1})
    if not isinstance(c, list) or not c: return {"status": "no_commit_before_snapshot"}
    sha = c[0]["sha"]; date = c[0]["commit"]["committer"]["date"]; tsha = c[0]["commit"]["tree"]["sha"]
    t = api(f"repos/{full}/git/trees/{tsha}", {"recursive": 1})
    if "_error" in t: return {"status": "tree_error", "err": t["_error"]}
    blobs = [e for e in t.get("tree", []) if e.get("type") == "blob"]
    n_dirs = sum(1 for e in t.get("tree", []) if e.get("type") == "tree")
    sizes = []; dropped = 0; dropped_bytes = 0
    for e in blobs:
        s = e.get("size", 0); ext = e["path"].rsplit(".", 1)[-1].lower() if "." in e["path"] else ""
        if stars < 250 and s > 100_000 and ext in BIN_EXT:
            dropped += 1; dropped_bytes += s; continue
        sizes.append(s)
    raw = sum(sizes)
    tar = sum(512 + 512 * math.ceil(s / 512) for s in sizes) + 512 * n_dirs + 1024
    return {"status": "ok", "commit": sha, "commit_date": date, "truncated": bool(t.get("truncated")),
            "n_files": len(sizes), "n_dirs": n_dirs, "raw_bytes": raw, "tar_bytes": tar,
            "max_file": max(sizes) if sizes else 0, "dropped_bin": dropped, "dropped_bin_bytes": dropped_bytes,
            "file_size_hist_log2": {str(k): v for k, v in sorted(
                __import__("collections").Counter(int(math.log2(s)) if s > 0 else -1 for s in sizes).items())}}

def record(item, stratum, extra):
    st = tree_stats(item["full_name"], item["default_branch"], item["stargazers_count"])
    d = {"stratum": stratum, "full_name": item["full_name"], "stars_now": item["stargazers_count"],
         "created_at": item["created_at"], "api_size_kb_now": item["size"], "language": item.get("language"), **extra, **st}
    print(stratum, d["full_name"], d["stars_now"], d.get("status"), d.get("tar_bytes"), flush=True)
    return d

from concurrent.futures import ThreadPoolExecutor
POOL = ThreadPoolExecutor(8)
def main():
    res = {"until": UNTIL, "seed": 20261025, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "bins": {}, "repos": []}
    seen = set()
    BINS = [(250, 499), (500, 999), (1000, 1999), (2000, 4999), (5000, 19999), (20000, None)]
    for lo, hi in BINS:
        rng_s = f"{lo}..{hi}" if hi else f">={lo}"
        tot = api("search/repositories", {"q": f"stars:{rng_s} created:<2020-02-02 fork:false", "per_page": 1}, search=True)
        key = f"{lo}-{hi}"
        res["bins"][key] = {"total_count_now": tot.get("total_count"), "q": f"stars:{rng_s} created:<2020-02-02 fork:false"}
        got = 0; tries = 0
        while got < 35 and tries < 60:
            tries += 1
            mo = random.choice(months())
            its, _ = pick_search(f"stars:{rng_s} created:{mo} fork:false")
            its = [it for it in its if it["full_name"] not in seen][:35 - got]
            for it in its: seen.add(it["full_name"])
            ds = list(POOL.map(lambda it: record(it, "high", {"bin": key, "month": mo}), its))
            res["repos"] += ds; got += sum(1 for d in ds if d.get("status") == "ok")
        res["bins"][key]["n_ok"] = got
        json.dump(res, open(OUT, "w"), indent=1)
    import datetime as dtm
    def days(a, b):
        a = dtm.date.fromisoformat(a); b = dtm.date.fromisoformat(b)
        return [(a + dtm.timedelta(i)).isoformat() for i in range((b - a).days + 1)]
    for name, sq, dlist, n in [("low0", "stars:0", days("2019-11-13", "2020-02-01"), 50),
                               ("low1", "stars:1..249", days("2019-02-03", "2020-02-01"), 70)]:
        got = 0; tries = 0
        while got < n and tries < 100:
            tries += 1
            day = random.choice(dlist)
            its, tot = pick_search(f"{sq} created:{day} fork:false")
            its = [it for it in its if it["full_name"] not in seen][:n - got]
            for it in its: seen.add(it["full_name"])
            ds = list(POOL.map(lambda it: record(it, name, {"day": day, "day_total_now": tot, "picked_per_search": 3}), its))
            res["repos"] += ds; got += sum(1 for d in ds if d.get("status") == "ok")
        json.dump(res, open(OUT, "w"), indent=1)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(OUT, "w"), indent=1)
main()
