"""研究026 計測：/tmp/tar026/*.tar を LZMA1（preset 6, 辞書 8 MiB）で圧縮し、計数つき参照復号器で復号。
出力の SHA-256 が元と一致することを確かめ、計数を data/counts.json に保存。"""
import json, os, lzma, subprocess, hashlib, time
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data"
T = json.load(open(f"{D}/tars.json"))["repos"]; out = []
for r in T:
    if r.get("status") != "ok": continue
    fn = f"/tmp/tar026/{r['idx']:02d}.tar"; raw = open(fn, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == r["sha256"]
    t0 = time.time()
    comp = lzma.compress(raw, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "preset": 6, "dict_size": 8 << 20, "lc": 3, "lp": 0, "pb": 2}])
    open("/tmp/m026.lzma", "wb").write(comp)
    env = dict(os.environ, COUNTS_JSON="/tmp/m026.json")
    p = subprocess.run(["/tmp/lzmaspec_instr", "/tmp/m026.lzma", "/tmp/m026.out"], env=env, capture_output=True, text=True)
    ok = p.returncode == 0 and hashlib.sha256(open("/tmp/m026.out", "rb").read()).hexdigest() == r["sha256"]
    c = json.load(open("/tmp/m026.json"))
    rec = {"idx": r["idx"], "full_name": r["full_name"], "stratum": r["stratum"], "stars_now": r["stars_now"], "tar_bytes": r["tar_bytes"],
           "tar_bytes_tree_est": r["tar_bytes_tree_est"], "lzma_bytes": len(comp), "roundtrip_ok": ok, "seconds": round(time.time() - t0, 2), **c}
    print(r["idx"], r["full_name"], r["tar_bytes"], len(comp), ok, rec["seconds"], flush=True); out.append(rec)
    os.remove("/tmp/m026.out")
json.dump({"lzma_props": {"preset": 6, "dict_size": 8 << 20, "lc": 3, "lp": 0, "pb": 2, "format": "LZMA_ALONE"},
           "lru": {"K": 8, "block": 512}, "repos": out}, open(f"{D}/counts.json", "w"), indent=1)
