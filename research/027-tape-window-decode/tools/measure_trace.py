"""研究027 計測：/tmp/tar026/*.tar（026 の fetch_tars.py で再構成、SHA-256 照合）を 026 と同じ LZMA1 設定で圧縮し、
トレースつき参照復号器（tools/lzmaspec_trace.cpp → /tmp/lzmaspec_trace）で復号。
出力の SHA-256 一致と、計数が 026 の counts.json と完全一致することを確かめ、距離 >16 KiB の参照を data/traces.npz に保存。"""
import json, os, lzma, subprocess, hashlib, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = f"{HERE}/../data"
T = json.load(open(f"{HERE}/../../026-lzma-aux-window/data/tars.json"))["repos"]
C26 = {r["idx"]: r for r in json.load(open(f"{HERE}/../../026-lzma-aux-window/data/counts.json"))["repos"]}
arrs = {}; log = []
for r in T:
    if r.get("status") != "ok": continue
    raw = open(f"/tmp/tar026/{r['idx']:02d}.tar", "rb").read(); assert hashlib.sha256(raw).hexdigest() == r["sha256"]
    comp = lzma.compress(raw, format=lzma.FORMAT_ALONE, filters=[{"id": lzma.FILTER_LZMA1, "preset": 6, "dict_size": 8 << 20, "lc": 3, "lp": 0, "pb": 2}])
    assert len(comp) == C26[r["idx"]]["lzma_bytes"]
    open("/tmp/m027.lzma", "wb").write(comp)
    env = dict(os.environ, COUNTS_JSON="/tmp/m027.json", TRACE_BIN="/tmp/m027.trace")
    p = subprocess.run(["/tmp/lzmaspec_trace", "/tmp/m027.lzma", "/tmp/m027.out"], env=env, capture_output=True, text=True)
    ok = p.returncode == 0 and hashlib.sha256(open("/tmp/m027.out", "rb").read()).hexdigest() == r["sha256"]
    c = json.load(open("/tmp/m027.json")); same = all(c[k] == C26[r["idx"]][k] for k in c)
    t = np.fromfile("/tmp/m027.trace", dtype=np.uint64).reshape(-1, 2)
    pos = t[:, 0].astype(np.int64); w = t[:, 1]
    dist = (w & np.uint64(0xFFFFFFFF)).astype(np.int64); ln = ((w >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int32); cnt = ((w >> np.uint64(48)) & np.uint64(1)).astype(np.int8)
    arrs[f"pos_{r['idx']}"] = pos; arrs[f"dist_{r['idx']}"] = dist; arrs[f"len_{r['idx']}"] = ln; arrs[f"cnt_{r['idx']}"] = cnt
    log.append({"idx": r["idx"], "roundtrip_ok": ok, "counts_identical_026": same, "n_trace": int(len(t)), "out": c["out"]})
    print(r["idx"], r["full_name"], ok, same, len(t), flush=True)
    os.remove("/tmp/m027.out")
np.savez_compressed(f"{D}/traces.npz", **arrs)
json.dump({"measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repos": log}, open(f"{D}/trace_log.json", "w"), indent=1)
