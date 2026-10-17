"""研究029：価値の重みを星から依存関係へ（open_problems 87）。事前登録：PREREGISTRATION.md。
入力：data/deps_raw.jsonl（collect.py）、028 の sim.py の items・関数（runpy で再利用、028 の入力のみ）。
seed：ブートストラップ 20261029。028 の乱数系列はそのまま。numpy＋matplotlib のみ。"""
import os, json, time, runpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P28 = f"{HERE}/../028-important-repo-dualwrite"
os.environ["OUT_JSON"] = "_rep029_tmp.json"
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{P28}/sim.py", run_name="sim")
os.remove(f"{P28}/_rep029_tmp.json")
items, w, st, tar, sw, M, need0, needc, cost_bytes, dV, REEL, SETS = (G[k] for k in
    "items w st tar sw M need0 needc cost_bytes dV REEL SETS".split())
R28 = json.load(open(f"{P28}/results.json"))
N = len(M); Ms = np.sort(M); pM = lambda x: np.searchsorted(Ms, x, side="right") / N
R = {"check_028": {"as_archived_q": R28["as_archived"]["mean_q"], "star_share_ge20k": R28["star_share"]["all_ge20k"]}}

# ---- 依存データ ----
raw = {}
for l in open(f"{HERE}/data/deps_raw.jsonl"):
    d = json.loads(l)
    if d["status"] == "ok" or d["full_name"] not in raw: raw[d["full_name"]] = d
ok = np.array([raw.get(i["name"], {}).get("status") == "ok" for i in items])
def col(k): return np.array([float(raw[i["name"]][k]) if o else 0.0 for i, o in zip(items, ok)])
D1, D2, D3 = col("D1_max_dep_repos"), col("D2_sum_eco_max_dep_repos"), col("D3_max_dep_packages")
npk = col("n_packages_page1")
cen = np.array([i["src"] == "census" for i in items]); g20 = np.array([i["ge20k"] for i in items])
R["data"] = {"n_items": len(items), "n_ok": int(ok.sum()), "n_missing": int((~ok).sum()),
             "missing_names": [i["name"] for i, o in zip(items, ok) if not o][:30],
             "n_census": int(cen.sum()), "n_sample025": int((~cen).sum())}
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    # 同順位は平均順位
    def avg(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); xs = x[o]; i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and xs[j + 1] == xs[i]: j += 1
            r[o[i:j + 1]] = (i + j) / 2; i = j + 1
        return r
    ra, rb = avg(a), avg(b); return float(np.corrcoef(ra, rb)[0, 1])
m20 = g20 & ok
R["ge20k"] = {"n": int(m20.sum()), "spearman_stars_D1": spearman(st[m20], D1[m20]),
              "spearman_stars_D3": spearman(st[m20], D3[m20]),
              "frac_D1_zero": float((D1[m20] == 0).mean()), "frac_no_package": float((npk[m20] == 0).mean()),
              "frac_D1_ge100": float((D1[m20] >= 100).mean()), "frac_D1_ge10000": float((D1[m20] >= 10000).mean()),
              "D1_quantiles": {str(p): float(np.percentile(D1[m20], p)) for p in (50, 75, 90, 99)}}
hs = (~cen) & ok
R["sample025"] = {"n": int(hs.sum()), "frac_D1_zero": float((D1[hs] == 0).mean()), "spearman_stars_D1": spearman(st[hs], D1[hs]),
                  "max_D1": float(D1[hs].max()), "max_name": items[int(np.where(hs)[0][np.argmax(D1[hs])])]["name"]}
# 上位 repo（依存）
o = np.argsort(-D1 * cen)
R["top_dep_census"] = [{"name": items[k]["name"], "stars": st[k], "D1": D1[k], "D3": D3[k],
                        "top_pkg": (raw[items[k]["name"]]["top"][0]["ecosystem"] + ":" + raw[items[k]["name"]]["top"][0]["name"]) if raw[items[k]["name"]]["top"] else None}
                       for k in o[:25]]
R["featured_D1"] = {items[k]["name"]: {"D1": D1[k], "D3": D3[k], "top_pkg": (raw[items[k]["name"]]["top"][0]["ecosystem"] + ":" + raw[items[k]["name"]]["top"][0]["name"]) if ok[k] and raw[items[k]["name"]]["top"] else None}
                    for k in range(len(items)) if items[k]["featured"]}

WEIGHTS = {"stars": sw, "dep_D1": w * D1, "dep_log1p_D1": w * np.log1p(D1), "dep_D2": w * D2, "dep_pkgs_D3": w * D3}
def share(swv, m): return float(swv[m].sum() / swv.sum())
# 重みの集中：依存重みの上位寄与（標本の束は w 倍で効く）
for lab in ("dep_D1", "dep_log1p_D1", "stars"):
    v = {"stars": sw, "dep_D1": w * D1, "dep_log1p_D1": w * np.log1p(D1)}[lab]; oo = np.argsort(-v)
    R.setdefault("top_contributors", {})[lab] = [{"name": items[k]["name"], "src": items[k]["src"], "w": float(w[k]),
        "share": float(v[k] / v.sum())} for k in oo[:5]]
R["share"] = {lab: {s: share(v, m) for s, m in SETS.items()} for lab, v in WEIGHTS.items()}

# ブートストラップ（標本025 の項目を、区間ごとに復元抽出）
rng = np.random.default_rng(20261029)
S25 = G["S25"]; binmap = {r["full_name"]: r["bin"] for r in S25["repos"] if r.get("stratum") == "high"}
bidx = {}
for k, it in enumerate(items):
    if it["src"] == "sample025": bidx.setdefault(binmap[it["name"]], []).append(k)
def boot_shares(v, B=2000):
    out = {s: [] for s in ("featured15", "all_ge20k", "top1000")}
    for _ in range(B):
        cnt = np.zeros(len(items)); cnt[cen] = 1.0
        for b, ks in bidx.items():
            dr = rng.choice(ks, size=len(ks), replace=True); np.add.at(cnt, dr, 1.0)
        vv = v * cnt
        for s in out: out[s].append(vv[SETS[s]].sum() / vv.sum())
    return {s: [float(np.percentile(x, 5)), float(np.percentile(x, 50)), float(np.percentile(x, 95))] for s, x in out.items()}
R["share_boot90"] = {lab: boot_shares(WEIGHTS[lab]) for lab in ("stars", "dep_D1", "dep_log1p_D1")}

# ---- q（読者 B）と予算つき欲張り ----
def q_vec(e, swv):
    o = np.argsort(e); cw = np.cumsum(swv[o]) / swv.sum(); k = np.searchsorted(e[o], M, side="right")
    return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
def greedy_set(c, budget, swv):
    nc = needc(c); cb = w * cost_bytes(c)
    gain = swv * np.clip(pM(need0 - 1) - pM(nc - 1), 0, None)
    od = np.argsort(-(gain / np.maximum(cb, 1)), kind="stable"); cc = np.cumsum(cb[od])
    frac = np.zeros(len(items)); k = np.searchsorted(cc, budget * REEL); frac[od[:k]] = 1.0
    if k < len(od) and items[od[k]]["src"] == "sample025": frac[od[k]] = (budget * REEL - (cc[k - 1] if k else 0)) / cb[od[k]]
    frac[gain <= 0] = 0.0
    return frac, nc
def q_frac(frac, nc, swv):
    e1 = np.minimum(need0, nc)
    q1 = q_vec(e1, swv * frac) if (swv * frac).sum() > 0 else np.zeros(N)
    a = (swv * frac).sum() / swv.sum()
    q0 = q_vec(need0, swv * (1 - frac)) if a < 1 else np.zeros(N)   # 修正：全重みが二重書きされたとき 0/0 を避ける
    return a * q1 + (1 - a) * q0
def jacc(f1, f2):
    a = set(np.where((f1 > 0) & cen)[0]); b = set(np.where((f2 > 0) & cen)[0]); return len(a & b) / max(1, len(a | b))
mix = 0.5 * sw / sw.sum() + 0.5 * WEIGHTS["dep_D1"] / WEIGHTS["dep_D1"].sum()
WEIGHTS_SEL = {"stars": sw, "dep_D1": WEIGHTS["dep_D1"], "dep_log1p_D1": WEIGHTS["dep_log1p_D1"], "mix50": mix}
EVAL = ("stars", "dep_D1", "dep_log1p_D1")
R["as_archived_q"] = {lab: float(q_vec(need0, WEIGHTS[lab]).mean()) for lab in EVAL}
R["sets_c4"] = {}
for s in ("featured15", "all_ge20k", "all_ge5k", "all_high"):
    e = np.where(SETS[s], np.minimum(need0, needc("c4_4KiB_lc0")), need0)
    R["sets_c4"][s] = {lab: {"q": float(q_vec(e, WEIGHTS[lab]).mean())} for lab in EVAL}
    R["sets_c4"][s]["dV_dep_D1"] = dV(q_vec(e, WEIGHTS["dep_D1"])); R["sets_c4"][s]["dV_stars"] = dV(q_vec(e, sw))
BUD = [0.01, 0.03, 0.1, 0.3, 1.0]
R["budget_c4"] = {}; FR = {}
for B in BUD:
    row = {}
    for sl, sv in WEIGHTS_SEL.items():
        fr, nc = greedy_set("c4_4KiB_lc0", B, sv); FR[(B, sl)] = fr
        row[sl] = {"n_repos": float((w * fr).sum()), "cost_reels": float((w * cost_bytes("c4_4KiB_lc0") * fr).sum() / REEL),
                   "dq_by": {el: float(q_frac(fr, nc, WEIGHTS[el]).mean() - R["as_archived_q"][el]) for el in EVAL},
                   "top_census": [items[k]["name"] for k in np.argsort(-fr * sv) if fr[k] > 0 and cen[k]][:10]}
    row["jaccard_stars_dep_D1"] = jacc(FR[(B, "stars")], FR[(B, "dep_D1")])
    row["jaccard_stars_dep_log1p"] = jacc(FR[(B, "stars")], FR[(B, "dep_log1p_D1")])
    R["budget_c4"][str(B)] = row
b = R["budget_c4"]["0.1"]
fr, nc = greedy_set("c4_4KiB_lc0", 0.1, WEIGHTS["dep_D1"]); R["dV_dep_greedy_0.1"] = {"dV_dep": dV(q_frac(fr, nc, WEIGHTS["dep_D1"])), "dV_stars": dV(q_frac(fr, nc, sw))}
fr, nc = greedy_set("c4_4KiB_lc0", 0.1, sw); R["dV_star_greedy_0.1"] = {"dV_dep": dV(q_frac(fr, nc, WEIGHTS["dep_D1"])), "dV_stars": dV(q_frac(fr, nc, sw))}

# ---- 仮説 ----
H = {}
H["H41a"] = {"value": R["ge20k"]["spearman_stars_D1"], "rule": "<0.3", "supported": R["ge20k"]["spearman_stars_D1"] < 0.3}
H["H41b"] = {"value": R["ge20k"]["frac_D1_zero"], "rule": ">=0.5", "supported": R["ge20k"]["frac_D1_zero"] >= 0.5}
H["H41c"] = {"value": R["share"]["dep_D1"]["all_ge20k"], "boot90": R["share_boot90"]["dep_D1"]["all_ge20k"], "rule": ">=0.5", "supported": R["share"]["dep_D1"]["all_ge20k"] >= 0.5}
H["H41d"] = {"value": R["share"]["dep_D1"]["featured15"], "rule": ">=0.05", "supported": R["share"]["dep_D1"]["featured15"] >= 0.05}
r_e = b["stars"]["dq_by"]["dep_D1"] / b["dep_D1"]["dq_by"]["dep_D1"]
H["H41e"] = {"ratio_star_sel_eval_dep": r_e, "jaccard": b["jaccard_stars_dep_D1"], "rule": "ratio<=0.5 and jaccard<0.2",
             "supported": r_e <= 0.5 and b["jaccard_stars_dep_D1"] < 0.2}
H["H41f"] = {"value": R["sets_c4"]["all_high"]["dep_D1"]["q"], "rule": ">=0.6", "supported": R["sets_c4"]["all_high"]["dep_D1"]["q"] >= 0.6}
g_s = b["mix50"]["dq_by"]["stars"] / b["stars"]["dq_by"]["stars"]; g_d = b["mix50"]["dq_by"]["dep_D1"] / b["dep_D1"]["dq_by"]["dep_D1"]
H["H41g"] = {"mix_vs_star_opt": g_s, "mix_vs_dep_opt": g_d, "rule": "both>=0.8", "supported": g_s >= 0.8 and g_d >= 0.8}
R["hypotheses"] = H
# 欠測の扱い：欠測を除外した場合（事前登録の感度）
if (~ok).sum() > 0:
    R["missing_excluded_share_ge20k_dep_D1"] = share(WEIGHTS["dep_D1"] * ok, SETS["all_ge20k"])

# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].scatter(st[m20], D1[m20] + 1, s=6, alpha=0.5, label="≥20k stars (census)")
ax[0].scatter(st[hs], D1[hs] + 1, s=10, alpha=0.7, c="C3", label="025 sample (<20k)")
for k in range(len(items)):
    if items[k]["featured"] and ok[k]: ax[0].annotate(items[k]["name"].split("/")[1], (st[k], D1[k] + 1), fontsize=6)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("stars (2026)"); ax[0].set_ylabel("max dependent repos + 1 (ecosyste.ms)")
ax[0].legend(fontsize=7); ax[0].set_title(f"Spearman (≥20k) = {R['ge20k']['spearman_stars_D1']:.2f}")
labs = ["featured15", "top100", "top1000", "all_ge20k", "all_ge5k"]; x = np.arange(len(labs))
for j, lab in enumerate(("stars", "dep_D1", "dep_log1p_D1")):
    ax[1].bar(x + 0.27 * (j - 1), [R["share"][lab][s] for s in labs], 0.27, label=lab)
ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_ylabel("share of high-tier value"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_stars_vs_deps.png", dpi=120); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for j, el in enumerate(("stars", "dep_D1")):
    for sl, ls in (("stars", "--"), ("dep_D1", "-"), ("mix50", ":"), ("dep_log1p_D1", "-.")):
        ax[j].plot(BUD, [R["budget_c4"][str(B)][sl]["dq_by"][el] for B in BUD], ls, marker="o", label=f"selected by {sl}")
    ax[j].set_xscale("log"); ax[j].set_xlabel("budget (reels), c4 4KiB lc0"); ax[j].set_ylabel(f"q gain evaluated by {el}"); ax[j].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_budget_cross_eval.png", dpi=120); plt.close()
R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/" + os.environ.get("OUT029", "results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("data", "ge20k", "sample025", "share", "share_boot90", "as_archived_q", "sets_c4", "hypotheses")}, ensure_ascii=False, indent=1))
print(json.dumps({B: {sl: v["dq_by"] for sl, v in r.items() if isinstance(v, dict)} for B, r in R["budget_c4"].items()}, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
