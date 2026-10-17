"""研究028 圧縮計測：Featured15 と ≥20,000 星の無作為15件（seed=20261028）の実 TAR（codeload、スナップショットのコミット）を
5設定の LZMA2（Python lzma = liblzma）で圧縮し、バイト数を測る。入力 data/census.json、出力 data/compress.json。TAR は /tmp に置き保存しない。"""
import json, os, random, subprocess, hashlib, lzma, time, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
TMP = "/tmp/rep028c"; os.makedirs(TMP, exist_ok=True)
KiB, MiB = 1024, 1024 ** 2
CONFIGS = {"c0_8MiB_lc3": (8 * MiB, 3), "c1_64KiB_lc3": (64 * KiB, 3), "c2_64KiB_lc0": (64 * KiB, 0),
           "c3_16KiB_lc0": (16 * KiB, 0), "c4_4KiB_lc0": (4 * KiB, 0)}
CAP = 3e9  # これより大きい TAR は計測対象から外して補充（逸脱として記録）
C = json.load(open(f"{HERE}/data/census.json"))
ok = [r for r in C["repos"] if r.get("status") == "ok"]
feat = sorted([r for r in ok if r["featured"]], key=lambda r: r["full_name"])
pool_r = sorted([r for r in ok if not r["featured"] and r["stars_now"] >= 20000], key=lambda r: r["full_name"])
rnd = random.Random(20261028); rnd.shuffle(pool_r)
rand, skipped = [], []
for r in pool_r:
    if len(rand) == 15: break
    (skipped if r["tar_bytes"] > CAP else rand).append(r)
sel = [("featured", r) for r in feat] + [("random20k", r) for r in rand]

def fetch(item):
    grp, r = item
    fn = f"{TMP}/{r['full_name'].replace('/', '__')}.tar"
    if not os.path.exists(fn):
        url = f"https://codeload.github.com/{r['full_name']}/tar.gz/{r['commit']}"
        p = subprocess.run(f"curl -sfL --max-time 1800 '{url}' | gunzip -c > '{fn}.part' && mv '{fn}.part' '{fn}'", shell=True)
        if p.returncode != 0: return None
    h = hashlib.sha256(); n = 0
    with open(fn, "rb") as f:
        while (b := f.read(1 << 22)): h.update(b); n += len(b)
    return {"group": grp, "full_name": r["full_name"], "stars_now": r["stars_now"], "commit": r["commit"],
            "tar_bytes_tree_est": r["tar_bytes"], "tar_bytes_actual": n, "sha256": h.hexdigest(), "path": fn}

def comp(job):
    path, cfg = job; d, lc = CONFIGS[cfg]; t0 = time.time()
    c = lzma.LZMACompressor(format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                            filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": d, "lc": lc, "lp": 0, "pb": 2}])
    n = 0
    with open(path, "rb") as f:
        while (b := f.read(1 << 20)): n += len(c.compress(b))
    n += len(c.flush())
    return path, cfg, n, time.time() - t0

if __name__ == "__main__":
    t0 = time.time()
    with Pool(4) as p: got = p.map(fetch, sel)
    recs = [g for g in got if g]; failed = [s[1]["full_name"] for s, g in zip(sel, got) if not g]
    print("fetched", len(recs), "failed", failed, round(time.time() - t0), flush=True)
    jobs = sorted([(r["path"], cfg) for r in recs for cfg in CONFIGS], key=lambda j: -os.path.getsize(j[0]))
    out = {}
    with Pool(7) as p:
        for path, cfg, n, dt in p.imap_unordered(comp, jobs):
            out[(path, cfg)] = n; print(os.path.basename(path), cfg, n, round(dt, 1), flush=True)
    for r in recs:
        r["xz_bytes"] = {cfg: out[(r["path"], cfg)] for cfg in CONFIGS}; del r["path"]
    json.dump({"configs": {k: {"dict": v[0], "lc": v[1]} for k, v in CONFIGS.items()}, "cap_bytes": CAP,
               "skipped_over_cap": [r["full_name"] for r in skipped], "fetch_failed": failed, "repos": recs,
               "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
               "liblzma_note": "Python lzma module"}, open(f"{HERE}/data/compress.json", "w"), indent=1)
    print("done", round(time.time() - t0))
