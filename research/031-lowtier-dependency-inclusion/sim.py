"""研究031：高層の外にある依存の重み（open_problems 93）。事前登録：PREREGISTRATION.md（逸脱は report.md）。
入力：data/lowtier_gh.jsonl（collect.py）、030 の sim.py（runpy で再利用：高層の母集団・読者 M・dV・費用）、030 の data/tar_measured.jsonl。
seed 20261031。numpy＋matplotlib のみ。"""
import os, json, time, runpy, contextlib, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); P30 = f"{HERE}/../030-dependency-tail-census"
keep = {f: open(f"{P30}/{f}", "rb").read() for f in ("fig_shares_and_tail.png", "fig_budget_cross_eval_census.png")}
os.environ["OUT030"] = "_rep031_tmp.json"
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{P30}/sim.py", run_name="sim")
os.remove(f"{P30}/_rep031_tmp.json")
for f, b in keep.items(): open(f"{P30}/{f}", "wb").write(b)   # 030 の図を元のまま残す
it_hi = G["ITS"][10000]; M = G["M"]; dV = G["dV"]; REEL = G["REEL"]; rho = G["rho"]; pM = G["pM"]; q_vec = G["q_vec"]
need0_of, needc_of, cost_of = G["need0_of"], G["needc_of"], G["cost_of"]
C4 = "c4_4KiB_lc0"; INF = 1e30
rng = np.random.default_rng(20261031)
R = {}
# ---- 低層の TAR 回帰（030 の無作為150件の実測、030 explore と同じ） ----
canon = G["canon"]; meas = [json.loads(l) for l in open(f"{P30}/data/tar_measured.jsonl")]
rr = [(canon[v["name"]]["disk_kb"], v["tar_bytes"]) for v in meas if v["status"] == "ok" and v["group"] == "random150" and v["name"] in canon]
x = np.log(np.maximum(1, np.array([a for a, b in rr])) * 1024); y = np.log(np.array([b for a, b in rr], float))
b1, b0 = np.polyfit(x, y, 1); res = y - (b0 + b1 * x)
R["tar_regression"] = {"n": len(rr), "slope": float(b1), "intercept": float(b0), "resid_sd": float(res.std())}
# ---- 低層の分類 ----
CUT, A0, B0 = "2020-02-02", "2019-11-13", "2019-02-02"
def status(g, rule, forks):
    if g is None: return "not_found", False
    if g["createdAt"] >= CUT: return "created_after", False
    if g["isFork"] and not forks: return "fork", False
    if g["stargazerCount"] >= 250: return "now_ge250", True
    lc = g["last_commit_pre2020"]
    if lc is None: return "no_commit_pre2020", False
    if lc >= A0: return "A_80days", True
    if lc >= B0:
        if rule == "upper" or (rule == "main" and g["stargazerCount"] >= 1): return "B_year_star", True
        return "B_year_nostar", False
    return "last_commit_before_2019-02-02", False
low = [json.loads(l) for l in open(f"{HERE}/data/lowtier_gh.jsonl")]
R["collect"] = {"n": len(low), "n_ok": sum(d["status"] == "ok" for d in low), "n_not_found": sum(d["status"] == "not_found" for d in low),
                "n_query_failed": sum(d["status"] == "query_failed" for d in low), "t_first": low[0]["t"], "t_last": low[-1]["t"],
                "n_first_star_observed": sum(1 for d in low if d["gh"] and d["gh"]["first_star"])}
hi_names = {i["name"].lower() for i in it_hi} | {G["alias"].get(i["name"].lower(), i["name"].lower()) for i in it_hi}
# 030 の照会で作成日 ≥2020-02-02 または見つからなかった D1≥10,000 の repo（未収蔵）
extra = []; seen = set()
for k, d in G["gh"].items():
    if G["aggD"].get(k, (0,))[0] < 10000: continue
    g = d["gh"]
    if g is None: extra.append({"key": k, "D1_cen": float(G["aggD"][k][0]), "gh": None, "cls030": "not_found030"})
    elif g["createdAt"] >= CUT and not g["isFork"] and g["nameWithOwner"] not in seen:
        seen.add(g["nameWithOwner"]); extra.append({"key": k, "D1_cen": float(canon[g["nameWithOwner"]]["D1_cen"]), "cls030": "after030",
                                                    "gh": {**g, "last_commit_pre2020": None, "first_star": None}})
lowu = {}
for d in low + extra:
    nm = (d["gh"]["nameWithOwner"] if d["gh"] else d["key"]).lower()
    if nm in hi_names: continue
    if nm not in lowu or d["D1_cen"] > lowu[nm]["D1_cen"]: lowu[nm] = d
low = list(lowu.values())
R["n_low_unique"] = len(low)
disk = np.array([float((d["gh"] or {}).get("diskUsage") or d.get("rm_size_kb") or 0) for d in low])
def low_tar(mode):
    mu = b0 + b1 * np.log(np.maximum(1, disk) * 1024)
    return np.exp(mu + (rng.choice(res, size=len(mu)) if mode == "resid" else 0.0))
TAR_LOW = {"resid": low_tar("resid"), "median": low_tar("median")}

# ---- 母集団 U ----
w_h = np.array([i["w"] for i in it_hi]); D_h = np.array([i["D1"] for i in it_hi]); tar_h = np.array([i["tar"] for i in it_hi])
cost_h = cost_of(C4, it_hi, tar_h); n0_h = need0_of(tar_h); nc_h = needc_of(C4, tar_h)
strat_h = np.array([i["stratum"] for i in it_hi])
def build_U(rule="main", forks=False, tmode="resid"):
    st = [status(d["gh"], rule, forks) for d in low]
    arch_l = np.array([a for s, a in st]); tl = TAR_LOW[tmode]
    w = np.concatenate([w_h, np.ones(len(low))]); D = np.concatenate([D_h, [d["D1_cen"] for d in low]])
    tar = np.concatenate([tar_h, tl]); arch = np.concatenate([np.ones(len(it_hi), bool), arch_l])
    n0 = np.where(arch, need0_of(tar), INF); nc = np.where(arch, np.minimum(need0_of(tar), needc_of(C4, tar)), needc_of(C4, tar))
    cost = np.concatenate([cost_h, tl * rho[C4]]) * w
    tier = np.array(["high"] * len(it_hi) + ["low"] * len(low)); samp = np.concatenate([strat_h == "sample", np.zeros(len(low), bool)])
    cls = np.array(["high"] * len(it_hi) + [s for s, a in st])
    return dict(w=w, D=D, v=w * D, tar=tar, arch=arch, n0=n0, nc=nc, cost=cost, tier=tier, samp=samp, cls=cls,
                names=[i["name"] for i in it_hi] + [(d["gh"] or {}).get("nameWithOwner", d["key"]) for d in low])
Pge = lambda e: np.where(e >= INF, 0.0, 1.0 - pM(e - 1))
def qvec(U, fr):
    e = np.concatenate([U["nc"], U["n0"]]); vv = np.concatenate([U["v"] * fr, U["v"] * (1 - fr)])
    return q_vec(e, vv)
def greedy(U, cand, budget):
    gain = U["v"] * (Pge(U["nc"]) - Pge(U["n0"])); gain[~cand] = 0
    od = np.argsort(-(gain / np.maximum(U["cost"], 1)), kind="stable"); od = od[gain[od] > 0]
    cc = np.cumsum(U["cost"][od]); k = int(np.searchsorted(cc, budget * REEL)); fr = np.zeros(len(U["v"])); fr[od[:k]] = 1.0
    if k < len(od) and U["samp"][od[k]]: fr[od[k]] = (budget * REEL - (cc[k - 1] if k else 0)) / U["cost"][od[k]]
    return fr
def analyze(rule="main", forks=False, tmode="resid", full=True):
    U = build_U(rule, forks, tmode); v = U["v"]; V = v.sum(); out = {}
    lowm = U["tier"] == "low"
    out["share_U"] = {"high_tier": float(v[~lowm].sum() / V), "low_archived": float(v[lowm & U["arch"]].sum() / V),
                      "low_not_archived": float(v[lowm & ~U["arch"]].sum() / V)}
    out["archived_share_U"] = float(v[U["arch"]].sum() / V)
    out["low_archived_frac"] = float(v[lowm & U["arch"]].sum() / v[lowm].sum())
    out["low_by_class"] = {c: {"n": int((U["cls"] == c).sum()), "share_of_low": float(v[U["cls"] == c].sum() / v[lowm].sum())}
                           for c in sorted(set(U["cls"][lowm]))}
    out["n_low"] = int(lowm.sum()); out["n_low_archived"] = int((lowm & U["arch"]).sum())
    z = np.zeros(len(v)); q0 = qvec(U, z); out["q_as_archived_U"] = float(q0.mean()); out["dV0_U"] = dV(q0)
    out["q_as_archived_lowarch_only"] = float(q_vec(U["n0"][lowm & U["arch"]], v[lowm & U["arch"]]).mean()) if (lowm & U["arch"]).any() else None
    if not full: return out, U
    allU = np.ones(len(v)); qa = qvec(U, allU)
    out["all_U_c4"] = {"q": float(qa.mean()), "dq": float(qa.mean() - q0.mean()), "dV": dV(qa) - dV(q0), "dV_abs": dV(qa), "cost_reels": float(U["cost"].sum() / REEL)}
    hiall = (~lowm).astype(float); qh = qvec(U, hiall)
    out["all_high_c4_on_U"] = {"q": float(qh.mean()), "dq": float(qh.mean() - q0.mean()), "dV": dV(qh) - dV(q0), "cost_reels": float(U["cost"][~lowm].sum() / REEL)}
    lc = (lowm & U["arch"]).astype(float) + hiall
    CAND = {"S030_high_only": ~lowm, "S_snap_archived_only": U["arch"].copy(), "S_new_all_U": np.ones(len(v), bool)}
    out["budget"] = {}
    for Bd in (0.001, 0.003, 0.01, 0.03, 0.1):
        row = {}
        for s, cm in CAND.items():
            fr = greedy(U, cm, Bd); q1 = qvec(U, fr)
            row[s] = {"dq_U": float(q1.mean() - q0.mean()), "dV_U": dV(q1) - dV(q0), "n_repos": float((U["w"] * fr).sum()),
                      "n_low": float(fr[lowm].sum()), "n_low_unarchived": float(fr[lowm & ~U["arch"]].sum()),
                      "cost_reels": float((U["cost"] * fr).sum() / REEL),
                      "top": [U["names"][k] for k in np.argsort(-fr * v)[:8] if fr[k] > 0]}
        row["ratio_new_over_030"] = row["S_new_all_U"]["dq_U"] / row["S030_high_only"]["dq_U"]
        row["ratio_snap_over_new"] = row["S_snap_archived_only"]["dq_U"] / row["S_new_all_U"]["dq_U"]
        out["budget"][str(Bd)] = row
    # 030 の高層だけでの再現（検証）
    vh = v[~lowm]; z_h = np.zeros(len(vh))
    fr = greedy({k: (val[~lowm] if isinstance(val, np.ndarray) else [n for n, m in zip(val, ~lowm) if m]) for k, val in U.items()}, np.ones(len(vh), bool), 0.003)
    e = np.concatenate([U["nc"][~lowm], U["n0"][~lowm]]); q1 = q_vec(e, np.concatenate([vh * fr, vh * (1 - fr)])); q0h = q_vec(U["n0"][~lowm], vh)
    out["check030"] = {"q_as_archived_high": float(q0h.mean()), "dq_0.003_high": float(q1.mean() - q0h.mean())}
    # S-new の ΔV が全体 c4 の 90% に届く最小予算
    grid = np.geomspace(1e-4, 1.0, 41); dvs = []
    for Bd in grid:
        fr = greedy(U, CAND["S_new_all_U"], Bd); dvs.append(dV(qvec(U, fr)) - dV(q0))
    dvs = np.array(dvs); tgt = 0.9 * out["all_U_c4"]["dV"]
    out["min_budget_90pct_dV_S_new"] = float(grid[np.argmax(dvs >= tgt)]) if (dvs >= tgt).any() else None
    out["dV_curve"] = {"grid": grid.tolist(), "S_new": dvs.tolist()}
    dvh = []
    for Bd in grid:
        fr = greedy(U, CAND["S030_high_only"], Bd); dvh.append(dV(qvec(U, fr)) - dV(q0))
    out["dV_curve"]["S030"] = dvh
    dvsn = []
    for Bd in grid:
        fr = greedy(U, CAND["S_snap_archived_only"], Bd); dvsn.append(dV(qvec(U, fr)) - dV(q0))
    out["dV_curve"]["S_snap"] = dvsn
    # 低層の最後のコミット年（依存重み）
    yrs = {}
    for k in np.where(lowm)[0]:
        d = low[k - len(it_hi)]; g = d["gh"]; yv = (g or {}).get("last_commit_pre2020")
        yy = yv[:4] if yv else "none"; yrs[yy] = yrs.get(yy, 0.0) + float(v[k])
    out["low_last_commit_year_share"] = {k: yrs[k] / v[lowm].sum() for k in sorted(yrs)}
    top = [k for k in np.argsort(-v) if lowm[k]][:15]
    out["top_low"] = [{"name": U["names"][k], "D1": float(U["D"][k]), "class": str(U["cls"][k]), "tar_est": float(U["tar"][k]),
                       "last_commit_pre2020": (low[k - len(it_hi)]["gh"] or {}).get("last_commit_pre2020"),
                       "stars_now": (low[k - len(it_hi)]["gh"] or {}).get("stargazerCount")} for k in top]
    return out, U

main, Um = analyze("main")
R["main"] = main
R["sens"] = {}
for lab, kw in {"lower_A_only": dict(rule="lower"), "upper_any_star": dict(rule="upper"), "forks_by_rule": dict(forks=True),
                "tar_median": dict(tmode="median")}.items():
    o, _ = analyze(full=(lab == "tar_median"), **kw)
    R["sens"][lab] = {k: o[k] for k in ("share_U", "archived_share_U", "low_archived_frac", "q_as_archived_U")} | (
        {"ratio_new_over_030_0.003": o["budget"]["0.003"]["ratio_new_over_030"], "ratio_snap_over_new_0.003": o["budget"]["0.003"]["ratio_snap_over_new"],
         "min_budget_90pct_dV_S_new": o["min_budget_90pct_dV_S_new"]} if lab == "tar_median" else {})
# ---- 仮説 ----
H = {}; b3 = main["budget"]["0.003"]
H["H43a"] = {"value": main["low_archived_frac"], "rule": ">=0.5", "supported": main["low_archived_frac"] >= 0.5}
H["H43b"] = {"value": main["archived_share_U"], "rule": ">=0.70", "supported": main["archived_share_U"] >= 0.70}
H["H43c"] = {"value": main["q_as_archived_U"], "ref030": 0.07517046607589936, "rule": "<0.075", "supported": main["q_as_archived_U"] < 0.07517046607589936}
H["H43d"] = {"value": b3["ratio_new_over_030"], "rule": ">=1.5", "supported": b3["ratio_new_over_030"] >= 1.5}
H["H43e"] = {"value": b3["ratio_snap_over_new"], "rule": ">=0.8", "supported": b3["ratio_snap_over_new"] >= 0.8}
mb = main["min_budget_90pct_dV_S_new"]
H["H43f"] = {"value": mb, "rule": "<=0.01", "supported": mb is not None and mb <= 0.01}
vs = {lab: R["sens"][lab]["low_archived_frac"] >= 0.5 for lab in ("lower_A_only", "upper_any_star")}
H["H43g"] = {"verdicts": vs | {"main": H["H43a"]["supported"]}, "values": {lab: R["sens"][lab]["low_archived_frac"] for lab in vs},
             "rule": "same verdict", "supported": all(x == H["H43a"]["supported"] for x in vs.values())}
R["hypotheses"] = H
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
labs = ["lower (A only)", "main (A or B, stars now≥1)", "upper (A or B any)"]
keys = ["lower_A_only", None, "upper_any_star"]
vals = [R["sens"][k]["share_U"] if k else main["share_U"] for k in keys]
bt = np.zeros(3)
for part, col in (("high_tier", "C0"), ("low_archived", "C2"), ("low_not_archived", "C3")):
    h = [vv[part] for vv in vals]; ax[0].bar(labs, h, bottom=bt, color=col, label=part); bt += h
ax[0].set_ylabel("share of dependency weight in U (D1)"); ax[0].set_title("Was the dependency weight in the 2020 snapshot?"); ax[0].legend(fontsize=8)
ax[0].tick_params(axis="x", labelsize=8)
yl = main["low_last_commit_year_share"]; ax[1].bar(list(yl.keys()), list(yl.values()), color="C1")
ax[1].axvline(list(yl.keys()).index("2019") - 0.5 if "2019" in yl else 0, ls="--", c="k")
ax[1].set_ylabel("share of low-tier dependency weight"); ax[1].set_title("last default-branch commit before 2020-02-02 (low tier, D1≥10k)")
ax[1].tick_params(axis="x", rotation=60, labelsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_inclusion_shares.png", dpi=120); plt.close()
fig, ax = plt.subplots(figsize=(7, 4.6)); gr = main["dV_curve"]["grid"]
for s, ls in (("S_new", "-"), ("S_snap", "--"), ("S030", ":")): ax.plot(gr, main["dV_curve"][s], ls, label=s)
ax.axhline(main["all_U_c4"]["dV"], c="gray", lw=0.8, label="all of U in c4"); ax.axhline(0.9 * main["all_U_c4"]["dV"], c="gray", ls="--", lw=0.8, label="90%")
ax.set_xscale("log"); ax.set_xlabel("budget (reels), c4 4KiB lc0"); ax.set_ylabel("ΔV (dependency weight, U)"); ax.legend(fontsize=8)
ax.set_title("dual-write selection including the low tier"); plt.tight_layout(); plt.savefig(f"{HERE}/fig_budget_dV_U.png", dpi=120); plt.close()
R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/" + os.environ.get("OUT031", "results.json"), "w"), ensure_ascii=False, indent=1)
pr = {k: v for k, v in main.items() if k not in ("dV_curve",)}
print(json.dumps({"collect": R["collect"], "n_low_unique": R["n_low_unique"], "tar_regression": R["tar_regression"]}, indent=1))
print(json.dumps(pr, ensure_ascii=False, indent=1)); print(json.dumps(R["sens"], indent=1)); print(json.dumps(H, indent=1))
print("runtime", round(R["runtime_seconds"], 1))
