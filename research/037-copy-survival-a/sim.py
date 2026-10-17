"""研究037 sim.py：手元の写しが生き残り見つかる割合 a を、npm の公開 download 数から下から組み立て、035・036 の f_orph と 034 の順位に戻す。
036 の sim.py を一時ディレクトリで再実行し（中で 035 も再実行）、同じ係数 K=4,000 組を使う。追加の係数は seed 20261037。入力：data/npm_dl.json（collect.py）。"""
import json, os, sys, time, shutil, tempfile, runpy, io, contextlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(HERE)
R = {"seed": 20261037}
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
def wmed(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, 0.5 * c[-1])])

# ---------- 1. 036（と 035）の再実行 ----------
tmp = tempfile.mkdtemp(prefix="rep037_036_")
for d in ("024-content-value-reader-guide", "030-dependency-tail-census", "032-collapse-scenarios", "033-scenario-reader-q",
          "034-t3-vault-value", "035-orphan-share-distro", "036-star-repo-packaging"):
    shutil.copytree(f"{RES}/{d}", f"{tmp}/{d}")
cwd = os.getcwd(); os.chdir(f"{tmp}/036-star-repo-packaging"); sys.argv = ["sim.py"]
with contextlib.redirect_stdout(io.StringIO()):
    n36 = runpy.run_path("sim.py", run_name="__main__")
os.chdir(cwd)
n35 = n36["ns"]
P, P2, NREG, REGS, SCEN, K, by_scen, plug, summ = n36["P"], n36["P2"], n36["NREG"], n36["REGS"], n36["SCEN"], n36["K"], n36["by_scen"], n36["plug"], n36["summ"]
r36 = json.load(open(f"{RES}/036-star-repo-packaging/results.json"))
R["check_036"] = {a: (n36["R"]["plug_034"]["main"]["mix"][a]["P_S_gt_TC"], r36["plug_034"]["main"]["mix"][a]["P_S_gt_TC"]) for a in n35["WTS"]}
assert all(x == y for x, y in R["check_036"].values())
U, v, D1, dU = n35["U"], n35["v"], n35["D1"], n35["dU"]["main"]
S, st, w, d_deb, d_brew, inst = n36["S"], n36["st"], n36["w"], n36["d_deb"], n36["d_brew"], n36["inst"]
vn = v / v.sum()

# ---------- 2. npm の download（測定） ----------
DL = json.load(open(f"{HERE}/data/npm_dl.json")); DLT = float(DL["total"]["downloads"])
RP = DL["repo_pkgs"]; dl = DL["dl"]
def repo_dl(name, how):
    xs = [dl.get(p) for p in RP.get(name.lower(), [])]; xs = [x for x in xs if x is not None]
    if not xs: return np.nan
    return float(max(xs) if how == "max" else sum(xs))
def arr(items, how): return np.array([repo_dl(x["name"], how) for x in items])
DLU = {h: arr(U, h) for h in ("max", "sum")}; DLS = {h: arr(S, h) for h in ("max", "sum")}
mU = ~np.isnan(DLU["max"]); mS = ~np.isnan(DLS["max"])
n_pk = len(dl); n_none = sum(1 for x in dl.values() if x is None)
pos = mU & (D1 > 0)
ratio = DLU["max"][pos] / D1[pos]
R["measure"] = {"npm_total_last_month": DLT, "period": [DL["total"]["start"], DL["total"]["end"]], "n_packages": n_pk, "n_packages_no_data": n_none,
                "U_mapped_count": int(mU.sum()), "U_mapped_dep_weight": float(vn[mU].sum()),
                "star_mapped_count": int(mS.sum()), "star_mapped_star_weight": float(w[mS].sum()),
                "dl_per_D1_weighted_median": wmed(ratio, v[pos]), "dl_per_D1_count_median": float(np.median(ratio)),
                "dl_per_D1_p10_p90": [float(np.quantile(ratio, 0.1)), float(np.quantile(ratio, 0.9))],
                "spearman_dl_D1": spearman(DLU["max"][pos], D1[pos]),
                "dl_share_of_total_weighted_median": wmed(DLU["max"][pos] / DLT, v[pos]),
                "star_dl_per_star_median": float(np.median(DLS["max"][mS] / st[mS])) if mS.any() else None,
                "top_dl_U": sorted([(U[i]["name"], int(DLU["max"][i])) for i in np.where(mU)[0]], key=lambda x: -x[1])[:10]}

# ---------- 3. 下からの a ----------
g = np.random.default_rng(20261037)
NG = {"f_m": g.uniform(0.05, 0.5, K), "n_pkg": np.exp(g.uniform(np.log(300), np.log(3000), K)), "p_keep": g.uniform(0.2, 0.9, K),
      "t": g.uniform(1, 20, K), "p_f": np.exp(g.uniform(np.log(1e-3), np.log(0.3), K))}
AFR = 0.0157; GH = 1e8; JS = 0.623
def lam_base(DLx, NGx, sl=slice(None)):
    """地域の割合 s_r を掛ける前の λ（K 組の一部 sl, n）"""
    Nm = GH * JS * NGx["f_m"][sl]; ps = NGx["p_keep"][sl] * (1 - AFR) ** NGx["t"][sl]
    C = Nm[:, None] * np.minimum(1.0, np.nan_to_num(DLx)[None, :] * NGx["n_pkg"][sl][:, None] / DLT)
    return C * (ps * NGx["p_f"][sl])[:, None]
CH = 250
def fdep(r, DLx, NGx, m):
    Pdeb = 1 - (1 - P["p_m"]) ** NREG[r]; out = np.empty(K)
    for j0 in range(0, K, CH):
        sl = slice(j0, min(K, j0 + CH)); a = (P["a"][sl] * P["s_" + r][sl])[:, None]
        ins_old = (1 - P["g"][sl])[:, None] * (1 - np.exp(-D1[None, :] * a))
        ins_new = (1 - P["g"][sl])[:, None] * (1 - np.exp(-lam_base(DLx, NGx, sl) * P["s_" + r][sl][:, None]))
        ins = np.where(m[None, :], ins_new, ins_old)
        deb = (P["v"][sl] * Pdeb[sl])[:, None] * dU[None, :]
        out[sl] = ((1 - deb) * (1 - ins)) @ vn
    return out
def fstar(r, DLx, NGx, m):
    Pdeb = 1 - (1 - P["p_m"]) ** NREG[r]; out = np.empty(K)
    for j0 in range(0, K, CH):
        sl = slice(j0, min(K, j0 + CH)); a = (P["a"][sl] * P["s_" + r][sl])[:, None]
        deb = (P["v"][sl] * Pdeb[sl])[:, None] * d_deb[None, :]
        ins_old = (1 - P["g"][sl])[:, None] * (1 - np.exp(-st[None, :] * P["kappa"][sl][:, None] * a))
        ins_new = (1 - P["g"][sl])[:, None] * (1 - np.exp(-lam_base(DLx, NGx, sl) * P["s_" + r][sl][:, None]))
        ins = np.where(m[None, :], ins_new, ins_old)
        br = d_brew[None, :] * (1 - P2["g_b"][sl])[:, None] * (1 - np.exp(-inst[None, :] * P2["iota"][sl][:, None] * a))
        out[sl] = ((1 - deb) * (1 - br) * (1 - ins)) @ w
    return out
def model(how="max", NGx=NG, mUx=None, mSx=None, DLUx=None):
    mUx = mU if mUx is None else mUx; mSx = mS if mSx is None else mSx
    DU = DLU[how] if DLUx is None else DLUx
    fd = {r: fdep(r, DU, NGx, mUx) for r in REGS}; fs = {r: fstar(r, DLS[how], NGx, mSx) for r in REGS}
    fm = {r: 0.5 * fd[r] + 0.5 * fs[r] for r in REGS}
    return {"dep": by_scen(fd), "star": by_scen(fs), "mix": by_scen(fm)}
# 再現確認：置き換えなし（m を全 False）なら 036 の main と一致
nul = model(mUx=np.zeros_like(mU), mSx=np.zeros_like(mS))
chk = max(float(np.max(np.abs(nul["mix"][s] - n36["VAR"]["main"]["mix"][s]))) for s in SCEN)
chk_d = max(float(np.max(np.abs(nul["dep"][s] - n35["MAIN"]["dep"][s]))) for s in SCEN)
R["check_no_replacement_maxabs"] = {"mix_vs_036": chk, "dep_vs_035": chk_d}; assert chk < 1e-12 and chk_d < 1e-12
NG_low = dict(NG); NG_low["p_f"] = NG["p_f"] / 10
med_ratio = R["measure"]["dl_per_D1_count_median"]
imp = np.where(mU, DLU["max"], D1 * med_ratio); mImp = mU | (D1 > 0)
VAR = {"main": model("max"), "sum": model("sum"), "pf_low": model("max", NG_low),
       "explore_impute_nonnpm": model("max", mUx=mImp, DLUx=imp)}
R["f_orph"] = {k: {wk: {s: summ(x[wk][s]) for s in SCEN} for wk in ("dep", "star", "mix")} for k, x in VAR.items()}
R["plug_034"] = {k: {wk: plug(x[wk]) for wk in ("dep", "star", "mix")} for k, x in VAR.items()}
R["ref_036"] = {"mix_BIO": r36["f_orph"]["main"]["mix"]["BIO"]["median"],
                "star_BIO": r36["f_orph"]["main"]["star"]["BIO"]["median"], "dep_BIO_035": n35["R"]["f_orph"]["main"]["dep"]["BIO"]["median"],
                "plug_mix": {a: r36["plug_034"]["main"]["mix"][a]["P_S_gt_TC"] for a in n35["WTS"]}}
# 暗黙の a（EU、依存重みの中央値）
LU = lam_base(DLU["max"][pos], NG)          # (K, n_pos)
aimp = LU / D1[pos][None, :]          # λ/(D1 s_r)：s_r で割ったもの
amed = np.array([wmed(np.log10(np.maximum(aimp[k], 1e-300)), v[pos]) for k in range(K)])
R["a_implied"] = {"log10_weighted_median": summ(amed), "median": float(10 ** np.median(amed)),
                  "P_above_8.5e-5": float(np.mean(amed > np.log10(8.5e-5))), "P_above_2.9e-4": float(np.mean(amed > np.log10(2.9e-4))),
                  "prior_median": float(np.median(P["a"])), "prior_P_above_8.5e-5": float(np.mean(P["a"] > 8.5e-5))}
# 期待される写しの数（地域を掛ける前、依存重みの中央値）
R["copies_C_weighted_median"] = summ(np.array([wmed(LU[k] / (NG["p_keep"][k] * (1 - AFR) ** NG["t"][k] * NG["p_f"][k]), v[pos]) for k in range(K)]))
R["lambda_EU_weighted_median"] = summ(np.array([wmed(LU[k] * P["s_EU"][k], v[pos]) for k in range(K)]))
fm = VAR["main"]["mix"]["BIO"]
R["spearman_mix_BIO"] = {**{k: spearman(NG[k], fm) for k in NG}, **{k: spearman(P[k], fm) for k in ("a", "g", "kappa", "p_m")}}
R["spearman_dep_BIO"] = {**{k: spearman(NG[k], VAR["main"]["dep"]["BIO"]) for k in NG}, "g": spearman(P["g"], VAR["main"]["dep"]["BIO"]), "a": spearman(P["a"], VAR["main"]["dep"]["BIO"])}

# ---------- 3b. 探索（事前登録外、初回結果の後に追加。判定には使わない） ----------
pf_star = np.array([NG["p_f"][k] / max(wmed(LU[k] * P["s_EU"][k], v[pos]), 1e-300) for k in range(K)])   # λ_EU の中央値が 1 になる p_f
NG_inf = dict(NG); NG_inf["p_f"] = np.full(K, 1e12)
VINF = model("max", NG_inf)
unmS = ~mS & ~d_deb & ~d_brew
R["explore"] = {"p_f_for_lambda1_EU": summ(pf_star),
                "dep_floor_lambda_inf_BIO": summ(VINF["dep"]["BIO"]), "mix_lambda_inf_BIO": summ(VINF["mix"]["BIO"]),
                "plug_lambda_inf_mix": plug(VINF["mix"]),
                "star_weight_not_npm_deb_brew": float(w[unmS].sum()),
                "star_weight_npm_mapped_not_deb_brew": float(w[mS & ~d_deb & ~d_brew].sum()),
                "dep_weight_not_npm": float(vn[~mU].sum()),
                "spearman_star_BIO": {"a": spearman(P["a"], VAR["main"]["star"]["BIO"]), "kappa": spearman(P["kappa"], VAR["main"]["star"]["BIO"]), "g": spearman(P["g"], VAR["main"]["star"]["BIO"])}}

# ---------- 4. 仮説 ----------
FD = R["f_orph"]["main"]["dep"]["BIO"]["median"]; FM = R["f_orph"]["main"]["mix"]["BIO"]["median"]
PS = R["plug_034"]["main"]["mix"]["A_XPT_S"]["P_S_gt_TC"]; sp = R["spearman_mix_BIO"]
newk = ["f_m", "n_pkg", "p_keep", "t", "p_f"]
R["hypotheses"] = {
 "H49a": {"value": R["measure"]["dl_per_D1_weighted_median"], "supported": R["measure"]["dl_per_D1_weighted_median"] >= 100},
 "H49b": {"value": R["a_implied"]["median"], "supported": R["a_implied"]["median"] >= 8.5e-5},
 "H49c": {"value": FD, "supported": FD <= 0.11},
 "H49d": {"value": {"037": FM, "036": R["ref_036"]["mix_BIO"]}, "supported": FM <= 0.24},
 "H49e": {"value": PS, "supported": PS < 0.3},
 "H49f": {"value": {k: sp[k] for k in newk}, "supported": max(newk, key=lambda k: abs(sp[k])) == "p_f"},
}

# ---------- 5. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].scatter(D1[pos], DLU["max"][pos], s=4, alpha=0.4)
xx = np.logspace(np.log10(D1[pos].min()), np.log10(D1[pos].max()), 10)
for k in (1, 100, 1e4): ax[0].plot(xx, xx * k, ls="--", lw=0.8, label=f"{k:g} downloads/month per dependent")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("direct dependent repos D1 (031)"); ax[0].set_ylabel("npm downloads, last month (max over repo's packages)")
ax[0].set_title(f"npm downloads vs dependents (n={int(pos.sum())}, Spearman {R['measure']['spearman_dl_D1']:.2f})"); ax[0].legend(fontsize=7)
ax[1].hist(np.log10(P["a"]), bins=40, alpha=0.5, density=True, label="035 prior a ~ logU[1e-5,1e-2]")
ax[1].hist(amed, bins=40, alpha=0.5, density=True, label="bottom-up a (dep-weighted median, EU)")
ax[1].axvline(np.log10(8.5e-5), c="k", ls="--", lw=1, label="035 threshold 8.5e-5 (EU)")
ax[1].axvline(np.log10(2.9e-4), c="r", ls=":", lw=1, label="035 threshold 2.9e-4 (south)")
ax[1].set_xlabel("log10 a (surviving, found copies per dependent)"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=7)
ax[1].set_title("Prior vs bottom-up a")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_downloads_and_a.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
vs = ["main", "sum", "pf_low", "explore_impute_nonnpm"]; p = 0; ticks = []
ax[0].boxplot([n36["VAR"]["main"]["mix"]["BIO"]], positions=[p], widths=0.6, whis=(5, 95), showfliers=False); ticks.append("036 mix"); p += 1
for k in vs:
    for wk in ("dep", "mix"):
        ax[0].boxplot([VAR[k][wk]["BIO"]], positions=[p], widths=0.6, whis=(5, 95), showfliers=False); ticks.append(f"{k}\n{wk}"); p += 1
ax[0].axhline(n35["BRK"]["A_XPT_S"], ls="--", c="k", lw=1, label="034 break-even A_XPT_S"); ax[0].axhline(n35["BRK"]["A_XPT_E"], ls=":", c="r", lw=1, label="034 break-even A_XPT_E")
ax[0].set_yscale("log"); ax[0].set_xticks(range(p)); ax[0].set_xticklabels(ticks, fontsize=6); ax[0].set_ylabel("f_orph (avg region, 5-95%)"); ax[0].legend(fontsize=7)
ax[0].set_title("Unreplaceable share with bottom-up a")
xs = np.arange(3); bw = 0.16
ax[1].bar(xs, [R["ref_036"]["plug_mix"][a] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")], bw, label="036", color="grey")
for k, vv in enumerate(vs):
    ax[1].bar(xs + (k + 1) * bw, [R["plug_034"][vv]["mix"][a]["P_S_gt_TC"] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")], bw, label=vv)
ax[1].set_xticks(xs + 2 * bw); ax[1].set_xticklabels(["A_Ord", "A_XPT_S", "A_XPT_E"]); ax[1].set_ylim(0, 1); ax[1].set_ylabel("P(S > T+C), mix weight"); ax[1].legend(fontsize=7)
ax[1].set_title("Second site vs bundle")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_forph_plug.png", dpi=120); plt.close(fig)

R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("measure", "a_implied", "copies_C_weighted_median", "lambda_EU_weighted_median", "hypotheses", "spearman_mix_BIO", "spearman_dep_BIO")}, ensure_ascii=False, indent=1, default=float))
print("f_orph BIO", {k: {wk: round(R["f_orph"][k][wk]["BIO"]["median"], 4) for wk in ("dep", "star", "mix")} for k in VAR})
print("f_orph NUC_G", {k: {wk: round(R["f_orph"][k][wk]["NUC_G"]["median"], 4) for wk in ("dep", "star", "mix")} for k in VAR})
print("plug mix", {k: {a: round(R["plug_034"][k]["mix"][a]["P_S_gt_TC"], 3) for a in n35["WTS"]} for k in VAR})
print("ref 036", R["ref_036"]); print("explore", json.dumps(R["explore"], default=float)); print("runtime", R["runtime_s"])
