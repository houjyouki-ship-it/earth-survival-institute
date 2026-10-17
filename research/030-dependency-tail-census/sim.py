"""研究030：依存の重い裾の全数調査（open_problems 90）。事前登録：PREREGISTRATION.md。
入力：data/pkgs.jsonl（collect.py）、data/repos_gh.jsonl（collect_gh.py）、029 の data/deps_raw.jsonl、
028 の sim.py（runpy で再利用：items・読者 M・dV・費用）、025 の data/sample.json。seed 20261030。numpy＋matplotlib のみ。"""
import os, json, time, runpy, contextlib, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P28, P29 = f"{HERE}/../028-important-repo-dualwrite", f"{HERE}/../029-dependency-weight"
os.environ["OUT_JSON"] = "_rep030_tmp.json"
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{P28}/sim.py", run_name="sim")
os.remove(f"{P28}/_rep030_tmp.json")
items28, w28, st28, tar28, M, dV, REEL, CFG, rho, S25 = (G[k] for k in "items w st tar M dV REEL CFG rho S25".split())
cost28 = {c: G["cost_bytes"](c) for c in CFG}
KiB, MiB = 1024, 1024 ** 2; S0_LC3 = G["S0_LC3"]; D_ARCH = G["D_ARCH"]
N = len(M); Ms = np.sort(M); pM = lambda x: np.searchsorted(Ms, x, side="right") / N
TMAIN = int(os.environ.get("TMAIN", 10000)); TS = [1000, 10000, 100000]
rng = np.random.default_rng(20261030)
R = {"T_main": TMAIN}

# ---- 029 の依存数 ----
raw29 = {}
for l in open(f"{P29}/data/deps_raw.jsonl"):
    d = json.loads(l)
    if d["status"] == "ok" or d["full_name"] not in raw29: raw29[d["full_name"]] = d
D1_29 = {k: float(v.get("D1_max_dep_repos") or 0) for k, v in raw29.items() if v["status"] == "ok"}

# ---- 収集データ ----
log = [json.loads(l) for l in open(f"{HERE}/data/collect_log.jsonl")]
fin = {}
for l in log:
    if l.get("final"): fin[l["registry"]] = l
R["collect"] = {"n_registries": len(fin), "n_pages": len(log), "n_errors": sum(1 for l in log if "error" in l),
                "stopped_early": [l["registry"] for l in fin.values() if l.get("stopped_early")],
                "max_last_value_of_finished": max([l.get("last_value") or 0 for l in fin.values() if not l.get("stopped_early")] + [0])}
pk = {}
for l in open(f"{HERE}/data/pkgs.jsonl"):
    p = json.loads(l); pk[(p["registry"], p["name"])] = p
R["collect"]["n_packages_ge1000"] = len(pk)
eco_tot = {}
for p in pk.values(): eco_tot[p["ecosystem"]] = eco_tot.get(p["ecosystem"], 0) + 1
R["collect"]["packages_by_ecosystem_top"] = dict(sorted(eco_tot.items(), key=lambda x: -x[1])[:12])
import re
GHRE = re.compile(r"github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$", re.I)
def repo_of(p):   # collect_gh.py と同じ規則
    m = GHRE.search(p.get("repository_url") or "")
    if m: return f"{m.group(1)}/{m.group(2)}"
    if p.get("rm_full_name") and (p.get("rm_host") or "").lower() == "github": return p["rm_full_name"]
    return None
aggD = {}
for p in pk.values():
    r = repo_of(p)
    if r:
        k = r.lower()
        if p["dep_repos"] > aggD.get(k, (0, None))[0]: aggD[k] = (p["dep_repos"], f'{p["registry"]}:{p["name"]}')
R["collect"]["n_repo_keys"] = len(aggD); R["collect"]["n_repo_keys_ge10000"] = sum(1 for v in aggD.values() if v[0] >= 10000)
gh = {}
for l in open(f"{HERE}/data/repos_gh.jsonl"):
    d = json.loads(l); gh[d["key"]] = d; d["D1_cen"], d["top"] = aggD.get(d["key"], (d["D1_cen"], d["top"]))
R["collect"]["n_repos_queried_gh"] = len(gh); R["collect"]["n_gh_not_found"] = sum(1 for d in gh.values() if d["status"] != "ok")
# 正規名でまとめ直す（改名で複数のキーが同じ repo を指す場合は D1 の最大）
canon = {}
for d in gh.values():
    g = d["gh"]
    if not g: continue
    k = g["nameWithOwner"]; c = canon.get(k)
    if c is None or d["D1_cen"] > c["D1_cen"]:
        canon[k] = {"name": k, "D1_cen": float(d["D1_cen"]), "top": d["top"], "stars": float(g["stargazerCount"]), "created": g["createdAt"],
                    "fork": g["isFork"], "disk_kb": float(g["diskUsage"] or 0)}
hi_ok = lambda c: c["stars"] >= 250 and c["created"] < "2020-02-02" and not c["fork"]
lname28 = {it["name"].lower(): k for k, it in enumerate(items28)}
# 028 の census 名 → canon 名の対応（大小文字・改名）
canon_l = {k.lower(): v for k, v in canon.items()}
alias = {d["key"]: d["gh"]["nameWithOwner"].lower() for d in gh.values() if d["gh"]}
def cen_D1(name):
    c = canon_l.get(name.lower()) or canon_l.get(alias.get(name.lower(), ""), None)
    return c["D1_cen"] if c else 0.0

# ---- TAR の回帰（025 の高層標本） ----
hi25 = [r for r in S25["repos"] if r.get("status") == "ok" and r["stratum"] == "high"]
x = np.log(np.maximum(1, np.array([r["api_size_kb_now"] for r in hi25], float)) * 1024); y = np.log(np.array([r["tar_bytes"] for r in hi25], float))
b1, b0 = np.polyfit(x, y, 1); res = y - (b0 + b1 * x)
R["tar_regression"] = {"n": len(hi25), "slope": float(b1), "intercept": float(b0), "resid_sd": float(res.std()),
                       "r2": float(1 - res.var() / y.var())}
s25 = {r["full_name"]: r for r in hi25}
bins = S25["bins"]; BINS = [b for b in bins if b != "20000-None"]
def bin_of(s):
    for b in BINS:
        lo, hi = b.split("-"); 
        if s >= float(lo) and s <= float(hi): return b
    return "20000-None"

def build(T, tar_mode="resid"):
    """母集団を作る：028 の census（≥20k）＋裾の全数（D1≥T）＋025 標本の残り（重み付け直し）"""
    it = []
    in28 = set()
    for k, i in enumerate(items28):
        if i["src"] != "census": continue
        d = max(D1_29.get(i["name"], 0.0), cen_D1(i["name"]))
        it.append({**i, "D1": d, "stratum": "ge20k", "k28": k}); in28.add(i["name"].lower())
    tail = []
    for c in canon.values():
        if c["name"].lower() in in28 or c["name"].lower() in {alias.get(n, n) for n in []}: continue
        if not hi_ok(c) or c["D1_cen"] < T: continue
        tail.append(c)
    tail_names = {c["name"].lower() for c in tail}
    # 028 census 名の改名照合
    cen_alias = {alias.get(n, n) for n in in28}
    tail = [c for c in tail if c["name"].lower() not in cen_alias]; tail_names = {c["name"].lower() for c in tail}
    nB = {}
    for c in tail:
        b = bin_of(c["stars"]); nB[b] = nB.get(b, 0) + 1
        s = s25.get(c["name"])
        if s is None:
            for nm, r in s25.items():
                if nm.lower() == c["name"].lower(): s = r
        if s is not None: tr = float(s["tar_bytes"])
        else:
            mu = b0 + b1 * np.log(max(1.0, c["disk_kb"]) * 1024)
            tr = float(np.exp(mu + (rng.choice(res) if tar_mode == "resid" else 0.0))) if tar_mode != "disk" else max(1.0, c["disk_kb"]) * 1024
        it.append({"name": c["name"], "src": "tail", "w": 1.0, "stars": c["stars"], "tar": tr, "featured": False, "ge20k": False,
                   "ge5k": c["stars"] >= 5000, "D1": c["D1_cen"], "stratum": "tail", "k28": None, "bin": b, "in025": s is not None})
    rest = {}
    for k, i in enumerate(items28):
        if i["src"] != "sample025": continue
        nm = i["name"].lower(); nm2 = alias.get(nm, nm)
        if nm in tail_names or nm2 in tail_names: continue
        b = next(r["bin"] for r in hi25 if r["full_name"] == i["name"]); rest.setdefault(b, []).append(k)
    for b, ks in rest.items():
        wb = (bins[b]["total_count_now"] - nB.get(b, 0)) / len(ks)
        for k in ks:
            i = items28[k]; it.append({**i, "w": wb, "D1": D1_29.get(i["name"], 0.0), "stratum": "sample", "k28": k, "bin": b})
    return it, nB
def arrays(it):
    w = np.array([i["w"] for i in it]); st = np.array([i["stars"] for i in it]); tar = np.array([i["tar"] for i in it])
    D1 = np.array([i["D1"] for i in it])
    return w, st, tar, D1
def need0_of(tar): return S0_LC3 + np.minimum(D_ARCH, tar)
def needc_of(c, tar): d, s0 = CFG[c]; return s0 + (np.minimum(d, tar) if d else 0)
def cost_of(c, it, tar):
    return np.array([cost28[c][i["k28"]] if i["k28"] is not None else tar[j] * rho[c] for j, i in enumerate(it)])
def q_vec(e, v):
    o = np.argsort(e); cw = np.cumsum(v[o]) / v.sum(); k = np.searchsorted(e[o], M, side="right")
    return np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
def greedy(it, w, tar, sv, c, budget):
    n0, nc = need0_of(tar), needc_of(c, tar); cb = w * cost_of(c, it, tar)
    gain = sv * np.clip(pM(n0 - 1) - pM(nc - 1), 0, None)
    od = np.argsort(-(gain / np.maximum(cb, 1)), kind="stable"); cc = np.cumsum(cb[od])
    fr = np.zeros(len(it)); k = int(np.searchsorted(cc, budget * REEL)); fr[od[:k]] = 1.0
    if k < len(od) and it[od[k]]["stratum"] == "sample": fr[od[k]] = (budget * REEL - (cc[k - 1] if k else 0)) / cb[od[k]]
    fr[gain <= 0] = 0.0
    return fr, n0, nc
def q_frac(fr, n0, nc, v):
    e1 = np.minimum(n0, nc); a = (v * fr).sum() / v.sum()
    q1 = q_vec(e1, v * fr) if (v * fr).sum() > 0 else np.zeros(N)
    q0 = q_vec(n0, v * (1 - fr)) if a < 1 - 1e-12 else np.zeros(N)
    return a * q1 + (1 - a) * q0

def analyze(T, tar_mode="resid", full=True):
    it, nB = build(T, tar_mode); w, st, tar, D1 = arrays(it)
    strat = np.array([i["stratum"] for i in it]); feat = np.array([bool(i["featured"]) for i in it])
    WT = {"stars": w * st, "dep_D1": w * D1, "dep_log1p_D1": w * np.log1p(D1)}
    out = {"n_items": len(it), "n_tail": int((strat == "tail").sum()), "n_sample_rest": int((strat == "sample").sum()),
           "tail_by_bin": nB, "tail_in025": int(sum(1 for i in it if i["stratum"] == "tail" and i.get("in025")))}
    sets = {"featured15": feat, "all_ge20k": strat == "ge20k", "tail": strat == "tail", "sample_rest": strat == "sample"}
    out["share"] = {lab: {s: float(v[m].sum() / v.sum()) for s, m in sets.items()} for lab, v in WT.items()}
    out["total_dep_weight"] = float(WT["dep_D1"].sum())
    # ブートストラップ（標本層のみ）
    bidx = {}
    for j, i in enumerate(it):
        if i["stratum"] == "sample": bidx.setdefault(i["bin"], []).append(j)
    B = 2000 if full else 300; sh = {s: [] for s in ("all_ge20k", "tail", "featured15")}
    for _ in range(B):
        cnt = np.where(strat == "sample", 0.0, 1.0)
        for b, ks in bidx.items(): np.add.at(cnt, rng.choice(ks, size=len(ks), replace=True), 1.0)
        vv = WT["dep_D1"] * cnt
        for s in sh: sh[s].append(vv[sets[s]].sum() / vv.sum())
    out["share_boot90_dep_D1"] = {s: [float(np.percentile(v, 5)), float(np.percentile(v, 50)), float(np.percentile(v, 95))] for s, v in sh.items()}
    top = np.argsort(-WT["dep_D1"])[:10]
    out["top_contributors_dep_D1"] = [{"name": it[k]["name"], "stratum": it[k]["stratum"], "w": float(w[k]), "D1": float(D1[k]),
                                       "stars": float(st[k]), "tar": float(tar[k]), "share": float(WT["dep_D1"][k] / WT["dep_D1"].sum())} for k in top]
    out["max_sample_contrib_dep_D1"] = float(max(WT["dep_D1"][strat == "sample"]) / WT["dep_D1"].sum())
    n0 = need0_of(tar)
    out["as_archived_q"] = {lab: float(q_vec(n0, v).mean()) for lab, v in WT.items()}
    out["tail_tar_median"] = float(np.median(tar[strat == "tail"])) if (strat == "tail").any() else None
    out["dep_weighted_median_tar"] = float(tar[np.argsort(tar)][np.searchsorted(np.cumsum(WT["dep_D1"][np.argsort(tar)]) / WT["dep_D1"].sum(), 0.5)])
    out["star_weighted_median_tar"] = float(tar[np.argsort(tar)][np.searchsorted(np.cumsum(WT["stars"][np.argsort(tar)]) / WT["stars"].sum(), 0.5)])
    if not full: return out, it
    c = "c4_4KiB_lc0"
    e = np.minimum(n0, needc_of(c, tar))
    out["all_high_c4"] = {lab: float(q_vec(e, v).mean()) for lab, v in WT.items()}
    out["all_high_c4"]["dV_dep"] = dV(q_vec(e, WT["dep_D1"])); out["all_high_c4"]["dV_stars"] = dV(q_vec(e, WT["stars"]))
    out["all_high_c4_cost_reels"] = float((w * cost_of(c, it, tar)).sum() / REEL)
    mix = 0.5 * WT["stars"] / WT["stars"].sum() + 0.5 * WT["dep_D1"] / WT["dep_D1"].sum()
    SEL = {"stars": WT["stars"], "dep_D1": WT["dep_D1"], "dep_log1p_D1": WT["dep_log1p_D1"], "mix50": mix}
    out["budget_c4"] = {}
    for Bd in (0.003, 0.01, 0.03, 0.1, 0.3):
        row = {}; FR = {}
        for sl, sv in SEL.items():
            fr, a0, ac = greedy(it, w, tar, sv, c, Bd); FR[sl] = fr
            row[sl] = {"n_repos": float((w * fr).sum()), "cost_reels": float((w * cost_of(c, it, tar) * fr).sum() / REEL),
                       "dq_by": {el: float(q_frac(fr, a0, ac, WT[el]).mean() - out["as_archived_q"][el]) for el in ("stars", "dep_D1")},
                       "top": [it[k]["name"] for k in np.argsort(-fr * sv)[:8] if fr[k] > 0]}
            if sl in ("dep_D1", "stars", "mix50"):
                row[sl]["dV_by"] = {el: dV(q_frac(fr, a0, ac, WT[el])) for el in ("stars", "dep_D1")}
        A = set(np.where(FR["stars"] > 0)[0]); Bs = set(np.where(FR["dep_D1"] > 0)[0])
        row["jaccard_stars_dep"] = len(A & Bs) / max(1, len(A | Bs))
        row["mix_vs_opt"] = {el: row["mix50"]["dq_by"][el] / row[el]["dq_by"][el] for el in ("stars", "dep_D1")}
        row["cross"] = {"star_sel_eval_dep_vs_opt": row["stars"]["dq_by"]["dep_D1"] / row["dep_D1"]["dq_by"]["dep_D1"],
                        "dep_sel_eval_star_vs_opt": row["dep_D1"]["dq_by"]["stars"] / row["stars"]["dq_by"]["stars"]}
        out["budget_c4"][str(Bd)] = row
    return out, it

R["by_T"] = {}
ITS = {}
for T in TS:
    R["by_T"][str(T)], ITS[T] = analyze(T, full=(T == TMAIN))
main = R["by_T"][str(TMAIN)]
R["tar_sensitivity"] = {m: {k: analyze(TMAIN, tar_mode=m, full=False)[0][k] for k in ("as_archived_q", "share")} for m in ("median", "disk")}

# ---- 方法の検証（H42a）と件数の推定 ----
it_m = ITS[TMAIN]; tailset = {i["name"].lower() for i in it_m if i["stratum"] == "tail"}
cen_found, cen_need, smp_found, smp_need, missing = 0, 0, 0, 0, []
for i in items28:
    d = D1_29.get(i["name"], 0.0)
    if d < TMAIN: continue
    if i["src"] == "census":
        cen_need += 1; ok = cen_D1(i["name"]) >= TMAIN; cen_found += ok
    else:
        smp_need += 1; nm = i["name"].lower(); ok = (nm in tailset) or (alias.get(nm, nm) in tailset); smp_found += ok
    if not ok: missing.append({"name": i["name"], "src": i["src"], "D1_029": d, "D1_cen": cen_D1(i["name"])})
R["validation"] = {"census_ge20k_found": cen_found, "census_ge20k_need": cen_need, "sample_found": smp_found, "sample_need": smp_need,
                   "recall": (cen_found + smp_found) / max(1, cen_need + smp_need), "missing": missing[:40]}
est = {}
for k, i in enumerate(items28):
    if i["src"] == "sample025" and D1_29.get(i["name"], 0) >= TMAIN:
        b = next(r["bin"] for r in hi25 if r["full_name"] == i["name"]); est[b] = est.get(b, 0) + w28[k]
R["count_check"] = {"sample_estimate_by_bin": est, "census_by_bin": main["tail_by_bin"],
                    "ratio_total": sum(est.values()) / max(1, sum(main["tail_by_bin"].values()))}

# ---- 仮説 ----
H = {}
H["H42a"] = {"value": R["validation"]["recall"], "rule": ">=0.9", "supported": R["validation"]["recall"] >= 0.9}
iv = main["share_boot90_dep_D1"]["all_ge20k"]; width = iv[2] - iv[0]
H["H42b"] = {"interval": iv, "width": width, "rule": "width<=0.107", "supported": width <= 0.107}
H["H42c"] = {"value": main["share"]["dep_D1"]["all_ge20k"], "rule": ">=0.15", "supported": main["share"]["dep_D1"]["all_ge20k"] >= 0.15}
H["H42d"] = {"value": main["share"]["dep_D1"]["tail"], "rule": ">=0.5", "supported": main["share"]["dep_D1"]["tail"] >= 0.5}
qa = main["as_archived_q"]
H["H42e"] = {"dep": qa["dep_D1"], "stars": qa["stars"], "ratio": qa["dep_D1"] / qa["stars"], "rule": "ratio>=2", "supported": qa["dep_D1"] / qa["stars"] >= 2}
b01 = main["budget_c4"]["0.01"]
H["H42f"] = {"value": b01["dep_D1"]["dq_by"]["dep_D1"], "rule": ">=0.5", "supported": b01["dep_D1"]["dq_by"]["dep_D1"] >= 0.5}
mv = {Bd: main["budget_c4"][Bd]["mix_vs_opt"] for Bd in ("0.01", "0.1")}
H["H42g"] = {"mix_vs_opt": mv, "rule": "all>=0.8", "supported": all(v >= 0.8 for d in mv.values() for v in d.values())}
R["hypotheses"] = H

# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
labs = ["029 (sample only)"] + [f"T={T:,}" for T in TS]
R29 = json.load(open(f"{P29}/results.json"))
pts = [R29["share"]["dep_D1"]["all_ge20k"]] + [R["by_T"][str(T)]["share"]["dep_D1"]["all_ge20k"] for T in TS]
lo = [R29["share_boot90"]["dep_D1"]["all_ge20k"][0]] + [R["by_T"][str(T)]["share_boot90_dep_D1"]["all_ge20k"][0] for T in TS]
hi_ = [R29["share_boot90"]["dep_D1"]["all_ge20k"][2]] + [R["by_T"][str(T)]["share_boot90_dep_D1"]["all_ge20k"][2] for T in TS]
xx = np.arange(len(labs))
ax[0].errorbar(xx, pts, yerr=[np.array(pts) - np.array(lo), np.array(hi_) - np.array(pts)], fmt="o", capsize=5, label="≥20k-star census")
tl = [np.nan] + [R["by_T"][str(T)]["share"]["dep_D1"]["tail"] for T in TS]
ax[0].plot(xx, tl, "s", c="C3", label="tail census (<20k stars, D1≥T)")
ax[0].set_xticks(xx); ax[0].set_xticklabels(labs); ax[0].set_ylabel("share of high-tier dependency weight (D1)"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
ax[0].set_title("dependency-weight shares: sample vs census (90% bootstrap)")
tt = [i for i in it_m if i["stratum"] in ("tail", "ge20k") and i["D1"] > 0]
ax[1].scatter([i["stars"] for i in tt if i["stratum"] == "ge20k"], [i["D1"] for i in tt if i["stratum"] == "ge20k"], s=5, alpha=0.5, label="≥20k census")
ax[1].scatter([i["stars"] for i in tt if i["stratum"] == "tail"], [i["D1"] for i in tt if i["stratum"] == "tail"], s=5, alpha=0.5, c="C3", label=f"tail census (T={TMAIN:,})")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("stars (2026)"); ax[1].set_ylabel("D1 (max dependent repos)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_shares_and_tail.png", dpi=120); plt.close()
BUD = (0.003, 0.01, 0.03, 0.1, 0.3)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for j, el in enumerate(("stars", "dep_D1")):
    for sl, ls in (("stars", "--"), ("dep_D1", "-"), ("mix50", ":"), ("dep_log1p_D1", "-.")):
        ax[j].plot(BUD, [main["budget_c4"][str(B)][sl]["dq_by"][el] for B in BUD], ls, marker="o", label=f"selected by {sl}")
    ax[j].set_xscale("log"); ax[j].set_xlabel("budget (reels), c4 4KiB lc0"); ax[j].set_ylabel(f"q gain evaluated by {el} (census-corrected)"); ax[j].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_budget_cross_eval_census.png", dpi=120); plt.close()
R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/" + os.environ.get("OUT030", "results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("collect", "tar_regression", "validation", "count_check", "hypotheses")}, ensure_ascii=False, indent=1))
for T in TS:
    r = R["by_T"][str(T)]; print("T", T, json.dumps({k: r[k] for k in ("n_tail", "share", "share_boot90_dep_D1", "as_archived_q", "max_sample_contrib_dep_D1")}, ensure_ascii=False))
print(json.dumps({k: main[k] for k in ("top_contributors_dep_D1", "all_high_c4", "all_high_c4_cost_reels", "tail_tar_median", "dep_weighted_median_tar", "star_weighted_median_tar")}, ensure_ascii=False, indent=1))
print(json.dumps({B: {sl: v["dq_by"] for sl, v in r.items() if isinstance(v, dict) and "dq_by" in v} | {"jacc": r["jaccard_stars_dep"], "mix": r["mix_vs_opt"], "cross": r["cross"]} for B, r in main["budget_c4"].items()}, indent=1))
print(json.dumps(R["tar_sensitivity"], indent=1))
print("runtime", round(R["runtime_seconds"], 1))
