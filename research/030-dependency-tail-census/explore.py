"""研究030 探索（事前登録外）。(1) 裾の全数の上位1,000件の TAR を実測値に置き換え（tools/measure_tar.py）、残りは
無作為150件の実測で付け直した回帰（残差の復元抽出、seed 20261031）で、収蔵のままの q（H42e）と予算つき選択を再計算する。
(2) D1≥10,000 の GitHub repo の依存重みのうち、高層（≥250 星・2020-02-02 以前・fork でない）の外にある割合。"""
import os, json, runpy, contextlib, io, time
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["OUT030"] = "_explore_tmp.json"
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{HERE}/sim.py", run_name="sim")
os.remove(f"{HERE}/_explore_tmp.json")
E = {}
meas = {}
for l in open(f"{HERE}/data/tar_measured.jsonl"):
    d = json.loads(l); meas[d["name"]] = d
ok = {k: v for k, v in meas.items() if v["status"] == "ok"}
E["measured"] = {"n": len(meas), "n_ok": len(ok), "n_no_commit": sum(v["status"] == "no_commit_before_snapshot" for v in meas.values()),
                 "n_truncated": sum(1 for v in ok.values() if v["truncated"]),
                 "median_tar_top1000": float(np.median([v["tar_bytes"] for v in ok.values() if v["group"] == "top1000"])),
                 "median_tar_random150": float(np.median([v["tar_bytes"] for v in ok.values() if v["group"] == "random150"]))}
canon = G["canon"]
# 無作為150件で回帰を付け直す
rr = [(canon[k]["disk_kb"], v["tar_bytes"]) for k, v in ok.items() if v["group"] == "random150" and k in canon]
x = np.log(np.maximum(1, np.array([a for a, b in rr])) * 1024); y = np.log(np.array([b for a, b in rr], float))
b1, b0 = np.polyfit(x, y, 1); res = y - (b0 + b1 * x)
E["tail_regression"] = {"n": len(rr), "slope": float(b1), "intercept": float(b0), "resid_sd": float(res.std())}
# 025 の回帰による予測と実測の比（上位1,000件）
g0, g1 = G["b0"], G["b1"]
pr = [(np.exp(g0 + g1 * np.log(max(1, canon[k]["disk_kb"]) * 1024)), v["tar_bytes"]) for k, v in ok.items() if k in canon]
E["pred025_over_measured_median"] = float(np.median([a / b for a, b in pr]))
rng = np.random.default_rng(20261031)
build0 = G["build"]
def build_meas(T, tar_mode="resid"):
    it, nB = build0(T, tar_mode)
    for i in it:
        if i["stratum"] != "tail" or i.get("in025"): continue
        v = meas.get(i["name"])
        if v and v["status"] == "ok": i["tar"] = float(v["tar_bytes"]); i["tar_src"] = "measured"
        else:
            i["tar"] = float(np.exp(b0 + b1 * np.log(max(1, canon[i["name"]]["disk_kb"]) * 1024) + rng.choice(res))); i["tar_src"] = "tail_regression"
    return it, nB
G["analyze"].__globals__["build"] = build_meas
out, it = G["analyze"](10000, full=True)
E["T10000_measured"] = {k: out[k] for k in ("as_archived_q", "share", "tail_tar_median", "dep_weighted_median_tar", "star_weighted_median_tar",
                                             "all_high_c4", "all_high_c4_cost_reels")}
E["T10000_measured"]["budget_c4"] = {B: {sl: r[sl]["dq_by"] for sl in ("stars", "dep_D1", "mix50")} | {"mix_vs_opt": r["mix_vs_opt"], "jacc": r["jaccard_stars_dep"],
                                      "cost_dep": r["dep_D1"]["cost_reels"], "n_dep": r["dep_D1"]["n_repos"], "top_dep": r["dep_D1"]["top"]}
                                     for B, r in out["budget_c4"].items()}
qa = out["as_archived_q"]; E["H42e_recheck"] = {"ratio": qa["dep_D1"] / qa["stars"], "supported_if_rule_ge2": qa["dep_D1"] / qa["stars"] >= 2}
# 依存重みが読者 B の RAM（中央値）以下で開ける割合
M = G["M"]; tar = np.array([i["tar"] for i in it]); w = np.array([i["w"] for i in it]); D1 = np.array([i["D1"] for i in it])
n0 = G["need0_of"](tar); E["dep_share_need0_le_M_median"] = float((w * D1 * (n0 <= np.median(M))).sum() / (w * D1).sum())
E["M_median_bytes"] = float(np.median(M))
# (2) 高層の外の依存重み
aggD = G["aggD"]; gh = G["gh"]
cls = {"high_tier": 0.0, "below250": 0.0, "created_after": 0.0, "fork": 0.0, "gh_not_found": 0.0, "not_queried_rm_stars_lt150": 0.0}
seen = set()
pk_rm = {}
for p in G["pk"].values():
    r = G["repo_of"](p)
    if r: pk_rm.setdefault(r.lower(), p.get("rm_stars"))
for k, (d, top) in aggD.items():
    if d < 10000: continue
    g = gh.get(k, {}).get("gh") if k in gh else None
    if k not in gh: cls["not_queried_rm_stars_lt150"] += d; continue
    if not g: cls["gh_not_found"] += d; continue
    nm = g["nameWithOwner"]
    if nm in seen: continue
    seen.add(nm); d = canon[nm]["D1_cen"]
    if g["isFork"]: cls["fork"] += d
    elif g["createdAt"] >= "2020-02-02": cls["created_after"] += d
    elif g["stargazerCount"] < 250: cls["below250"] += d
    else: cls["high_tier"] += d
tot = sum(cls.values()); E["dep_weight_outside_high_tier_D1ge10k"] = {k: v / tot for k, v in cls.items()}
# 高層の外（D1≥10,000）の repo を c4 で足す費用の粗い見積もり（大きさ：GitHub の diskUsage、なければ ecosyste.ms の size。
# TAR は無作為150件の回帰の中央値予測。fork・2020-02-02 以降・不明は除く）【推測】
pk_size = {}
for p in G["pk"].values():
    r = G["repo_of"](p)
    if r and p.get("rm_size_kb") is not None: pk_size.setdefault(r.lower(), p["rm_size_kb"])
outside = []; seen2 = set()
for k, (d, top) in aggD.items():
    if d < 10000: continue
    if k in gh:
        g = gh[k]["gh"]
        if not g or g["isFork"] or g["createdAt"] >= "2020-02-02" or g["stargazerCount"] >= 250 or g["nameWithOwner"] in seen2: continue
        seen2.add(g["nameWithOwner"]); outside.append(float(g["diskUsage"] or 0))
    else:
        outside.append(float(pk_size.get(k) or 0))
tb = np.exp(b0 + b1 * np.log(np.maximum(1, np.array(outside)) * 1024))
E["outside_high_tier_D1ge10k"] = {"n_repos": len(outside), "tar_total_bytes_est": float(tb.sum()),
                                   "c4_cost_reels_est": float(tb.sum() * G["rho"]["c4_4KiB_lc0"] / G["REEL"]),
                                   "note": "未照会の repo は作成日・fork を確認していない（上限側）"}
E["runtime_seconds"] = time.time() - t0
json.dump(E, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(E, ensure_ascii=False, indent=1))
