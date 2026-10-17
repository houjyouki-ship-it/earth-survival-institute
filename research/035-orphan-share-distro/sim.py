"""研究035 sim.py：金庫の中身のうち T3 社会の手元の写しで補えない割合 f_orph を、Debian の収録・鏡像の地理と手元の写しから見積もり、034 に戻す。
入力：data/match_trixie.json・match_sid.json（collect.py）、sources/debian_mirror_list_2026-10-17.html、data/south_src_check.json、
034 の sim.py（一時ディレクトリに 024・032・033・034 を写して runpy で再実行し、S_of と clone なしの u を得る）。seed 20261035。"""
import json, re, time, os, sys, shutil, tempfile, runpy, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def spearmanr(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return (float(np.corrcoef(rx, ry)[0, 1]),)
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(HERE)
R = {"seed": 20261035}
g = np.random.default_rng(20261035)

# ---------- 1. 収録（測定） ----------
def load(suite):
    m = json.load(open(f"{HERE}/data/match_{suite}.json"))
    U = m["U"]; S = m["star"]
    v = np.array([u["v"] for u in U]); D1 = None
    dU = {"M1": np.array([bool(u["m1"]) for u in U]), "M2": np.array([bool(u["m2"]) for u in U])}
    dU["main"] = dU["M1"] | dU["M2"]
    st = np.array([float(s["stars"]) for s in S])
    dS = {"M1": np.array([bool(s["m1"]) for s in S]), "M3": np.array([bool(s["m3"]) for s in S])}
    dS["main"] = dS["M1"] | dS["M3"]
    return m, U, S, v, dU, st, dS
COV = {}
for suite in ("trixie", "sid"):
    m, U, S, v, dU, st, dS = load(suite)
    tier = np.array([u["tier"] for u in U]); regs = [u["regs"] for u in U]
    c = {"n_U_archived": len(U), "n_star": len(S),
         "dep_share": {k: float(v[x].sum() / v.sum()) for k, x in dU.items()},
         "dep_count_share": {k: float(x.mean()) for k, x in dU.items()},
         "dep_share_by_tier": {t: float(v[(tier == t) & dU["main"]].sum() / v[tier == t].sum()) for t in ("high", "low")},
         "M2_recall_given_M1": float((dU["M1"] & dU["M2"]).sum() / max(1, dU["M1"].sum())),
         "star_share": {k: float(st[x].sum() / st.sum()) for k, x in dS.items()},
         "star_count_share": {k: float(x.mean()) for k, x in dS.items()}}
    eco = collections.defaultdict(lambda: [0.0, 0.0])
    for i, rr in enumerate(regs):
        key = "npmjs.org" if "npmjs.org" in rr else (rr[0] if rr else "none")
        eco[key][0] += v[i]; eco[key][1] += v[i] * dU["main"][i]
    c["dep_by_registry"] = {k: {"weight_share": a / v.sum(), "covered": b / a if a else None} for k, (a, b) in sorted(eco.items(), key=lambda x: -x[1][0])[:8]}
    COV[suite] = c
R["coverage"] = COV
m, U, S, v, dU, st, dS = load("trixie")
Ufull = {u["name"]: u for u in json.load(open(f"{HERE}/data/U_main.json")) if u["arch"]}
D1 = np.array([Ufull[u["name"]]["D1"] for u in U])
top_unc = sorted([(float(v[i] / v.sum()), U[i]["name"], U[i]["regs"]) for i in range(len(U)) if not dU["main"][i]], reverse=True)[:12]
R["top_uncovered_dep"] = top_unc
R["top_covered_dep"] = sorted([(float(v[i] / v.sum()), U[i]["name"], U[i]["m1"] + U[i]["m2"]) for i in range(len(U)) if dU["main"][i]], reverse=True)[:8]

# ---------- 2. 鏡像の地理（測定） ----------
h = open(f"{HERE}/sources/debian_mirror_list_2026-10-17.html", encoding="utf-8", errors="replace").read()
parts = re.split(r'<big><strong>([^<]+)</strong></big>', h)
cnt = {parts[i]: len(re.findall(r'<td valign="top"><a rel="nofollow" href', parts[i + 1])) for i in range(1, len(parts), 2)}
REGMAP = {"SOUTH": ["Argentina", "Australia", "Brazil", "Chile", "New Caledonia", "New Zealand", "Réunion", "South Africa", "Uruguay"],
          "NA": ["United States", "Canada", "Mexico", "Costa Rica"],
          "ASIA": ["Bangladesh", "Cambodia", "China", "Hong Kong", "India", "Indonesia", "Japan", "Korea", "Malaysia", "Singapore", "Taiwan", "Thailand", "Vietnam", "Kazakhstan"]}
other = [c for c in cnt if not any(c in L for L in REGMAP.values())]
AFR_ME = ["Burkina Faso", "Kenya", "Morocco", "Iran", "Israel", "Saudi Arabia"]
REGMAP["EU"] = [c for c in other if c not in AFR_ME]
n_reg = {r: sum(cnt[c] for c in L) for r, L in REGMAP.items()}
chk = json.load(open(f"{HERE}/data/south_src_check.json"))
f_src = sum(1 for x in chk if "200" in (x.get("https"), x.get("http"))) / len(chk)
R["mirrors"] = {"n_countries": len(cnt), "n_total": sum(cnt.values()), "by_region": n_reg, "south_checked": len(chk),
                "south_source_ok": int(round(f_src * len(chk))), "f_src": f_src, "excluded_afr_me": {c: cnt[c] for c in AFR_ME}}
NREG = {r: n_reg[r] * f_src for r in n_reg}   # source を持つ鏡像の期待数
REGS = ["NA", "EU", "ASIA", "SOUTH"]

# ---------- 3. f_orph の模型 ----------
K = 4000
def logu(lo, hi, n): return np.exp(g.uniform(np.log(lo), np.log(hi), n))
P = {"v": g.uniform(0.7, 1.0, K), "p_m": g.uniform(0.2, 0.9, K), "a": logu(1e-5, 1e-2, K), "g": g.uniform(0.0, 0.5, K),
     "kappa": logu(0.1, 1.0, K), "s_SOUTH": g.uniform(0.05, 0.10, K),
     "s_NA": g.uniform(0.15, 0.35, K), "s_EU": g.uniform(0.15, 0.35, K), "s_ASIA": g.uniform(0.15, 0.35, K)}
def forph(w, d, N, r, P, nreg=NREG, deb_on=True, inst_on=True):
    """係数 K 組ごとの f_orph（重み w、収録 d、写しの数の元 N、地域 r）"""
    Pdeb = 1 - (1 - P["p_m"]) ** nreg[r] if deb_on else np.zeros(K)
    out = np.empty(K); wn = w / w.sum()
    for j0 in range(0, K, 250):
        sl = slice(j0, min(K, j0 + 250))
        a = (P["a"][sl] * P["s_" + r][sl])[:, None]
        kap = P["kappa"][sl][:, None] if N is not None and N is st else 1.0
        inst = (1 - P["g"][sl])[:, None] * (1 - np.exp(-N[None, :] * kap * a)) if inst_on else 0.0
        deb = (P["v"][sl] * Pdeb[sl])[:, None] * d[None, :]
        out[sl] = ((1 - deb) * (1 - inst)) @ wn
    return out
SCEN = ["NUC_G", "NUC_R", "BIO", "AI", "CASC", "GEN"]
def by_scen(fr):   # fr: 地域 → (K,)
    o = {"NUC_G": fr["SOUTH"]}
    mix = np.mean([fr[r] for r in REGS], axis=0)
    for s in SCEN[1:]: o[s] = mix
    return o
def run_model(d_dep, d_star, P, deb_on=True, inst_on=True, nreg=NREG):
    fd = {r: forph(v, d_dep, D1, r, P, nreg, deb_on, inst_on) for r in REGS}
    fs = {r: forph(st, d_star, st, r, P, nreg, deb_on, inst_on) for r in REGS}
    fm = {r: 0.5 * fd[r] + 0.5 * fs[r] for r in REGS}
    return {"dep": by_scen(fd), "star": by_scen(fs), "mix": by_scen(fm), "region": {"dep": fd, "star": fs, "mix": fm}}
def summ(x): return {"median": float(np.median(x)), "mean": float(np.mean(x)), "p05": float(np.quantile(x, 0.05)), "p95": float(np.quantile(x, 0.95))}
MAIN = run_model(dU["main"], dS["main"], P)
VAR = {"main": MAIN,
       "M1_only": run_model(dU["M1"], dS["M1"], P),
       "no_debian": run_model(dU["main"], dS["main"], P, deb_on=False),
       "debian_only": run_model(dU["main"], dS["main"], P, inst_on=False)}
m_sid, U_sid, _, v_sid, dU_sid, _, dS_sid = load("sid")
assert [u["name"] for u in U_sid] == [u["name"] for u in U]
VAR["sid"] = run_model(dU_sid["main"], dS_sid["main"], P)
R["f_orph"] = {k: {w: {s: summ(x[w][s]) for s in SCEN} for w in ("dep", "star", "mix")} for k, x in VAR.items()}
R["f_orph_region_main"] = {w: {r: summ(MAIN["region"][w][r]) for r in REGS} for w in ("dep", "star", "mix")}

# 感度（Spearman、主の半々、地域を平均した値）
fmix = MAIN["mix"]["BIO"]
R["spearman_mix_avg_region"] = {k: float(spearmanr(P[k], fmix)[0]) for k in P}
R["spearman_mix_south"] = {k: float(spearmanr(P[k], MAIN["mix"]["NUC_G"])[0]) for k in P}

# ---------- 4. 034 に戻す ----------
tmp = tempfile.mkdtemp(prefix="rep035_034_")
for d in ("024-content-value-reader-guide", "032-collapse-scenarios", "033-scenario-reader-q", "034-t3-vault-value"):
    shutil.copytree(f"{RES}/{d}", f"{tmp}/{d}")
cwd = os.getcwd(); os.chdir(f"{tmp}/034-t3-vault-value"); sys.argv = ["sim.py"]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("sim.py", run_name="__main__")
os.chdir(cwd)
S_of, WTS, CW = ns["S_of"], ns["WTS"], ns["CROSS"]["weighted"]
MISS = ns["UEST"]["no_clone"]          # (K,) clone なしの u ＝ 拠点がすべて失われる確率の期待値
r34 = json.load(open(f"{RES}/034-t3-vault-value/results.json"))
R["check_034"] = {a: {"P_main_034": r34["S_vs_TC_under_u"]["main"][a]["P_S_gt_TC"], "P_rerun": ns["PSG"]["main"][a]["P_S_gt_TC"]} for a in WTS}
R["check_034_exact"] = all(abs(R["check_034"][a]["P_main_034"] - R["check_034"][a]["P_rerun"]) == 0 for a in WTS)
BRK = {a: r34["explore"]["f_orph_breakeven"][a] for a in WTS}
def plug(F):
    out = {}
    for a, w in WTS.items():
        Sk = sum(w[s] * (S_of(s, 0.0) + F[s] * MISS[s] * (S_of(s, 1.0) - S_of(s, 0.0))) for s in w)
        TC = CW[a]["TplusC"]
        out[a] = {"P_S_gt_TC": float(np.mean(Sk > TC)), "S_median": float(np.median(Sk)), "TplusC": TC, "S_over_TC_median": float(np.median(Sk) / TC)}
    return out
R["plug_034"] = {k: {w: plug(x[w]) for w in ("dep", "star", "mix")} for k, x in VAR.items()}
R["breakeven_034"] = BRK

# ---------- 5. 仮説 ----------
FM = R["f_orph"]["main"]["mix"]
medNG = FM["NUC_G"]["median"]; medO = np.mean([FM[s]["median"] for s in SCEN[1:]])
sp = R["spearman_mix_avg_region"]
R["hypotheses"] = {
 "H47a": {"value": COV["trixie"]["dep_share"]["main"], "supported": COV["trixie"]["dep_share"]["main"] >= 0.5},
 "H47b": {"value": {"dep": COV["trixie"]["dep_share"]["main"], "star": COV["trixie"]["star_share"]["main"]},
          "supported": COV["trixie"]["dep_share"]["main"] > COV["trixie"]["star_share"]["main"]},
 "H47c": {"value": {s: FM[s]["median"] for s in SCEN}, "supported": all(FM[s]["median"] < 0.087 for s in SCEN)},
 "H47d": {"value": {"NUC_G": medNG, "others_mean": float(medO), "ratio": float(medNG / medO)}, "supported": bool(medNG >= 2 * medO)},
 "H47e": {"value": {"a": sp["a"], "p_m": sp["p_m"]}, "supported": abs(sp["a"]) > abs(sp["p_m"])},
 "H47f": {"value": R["plug_034"]["main"]["mix"]["A_XPT_E"]["P_S_gt_TC"], "supported": R["plug_034"]["main"]["mix"]["A_XPT_E"]["P_S_gt_TC"] < 0.5},
}

# ---------- 6. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
labs = ["dep: M1", "dep: M1∪M2", "star: M1", "star: M1∪M3"]
for k, suite in enumerate(("trixie", "sid")):
    c = COV[suite]; vals = [c["dep_share"]["M1"], c["dep_share"]["main"], c["star_share"]["M1"], c["star_share"]["main"]]
    ax[0].bar(np.arange(4) + 0.38 * k, vals, 0.38, label=f"Debian {suite}")
ax[0].set_xticks(np.arange(4) + 0.19); ax[0].set_xticklabels(labs); ax[0].set_ylabel("share of vault value packaged in Debian main")
ax[0].set_title("Distro coverage of vault content (dependency vs star weight)"); ax[0].legend()
regs_lab = list(R["coverage"]["trixie"]["dep_by_registry"].items())[:6]
ax[1].bar([k for k, _ in regs_lab], [x["covered"] or 0 for _, x in regs_lab], color="C2")
for i, (k, x) in enumerate(regs_lab): ax[1].text(i, (x["covered"] or 0) + 0.01, f"w={x['weight_share']:.2f}", ha="center", fontsize=8)
ax[1].set_ylabel("covered share of dependency weight"); ax[1].set_title("Coverage by registry (trixie, w = share of U weight)")
ax[1].tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_distro_coverage.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
pos = 0; ticks = []
for w, col in (("dep", "C0"), ("star", "C1"), ("mix", "C3")):
    for sc in ("NUC_G", "BIO"):
        x = MAIN[w][sc]; ax[0].boxplot([x], positions=[pos], widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.5))
        ticks.append(f"{w}\n{'south' if sc=='NUC_G' else 'avg region'}"); pos += 1
ax[0].axhline(BRK["A_XPT_S"], ls="--", c="k", lw=1, label=f"034 break-even A_XPT_S {BRK['A_XPT_S']:.2f}")
ax[0].axhline(BRK["A_XPT_E"], ls=":", c="r", lw=1.2, label=f"034 break-even A_XPT_E {BRK['A_XPT_E']:.3f}")
ax[0].set_yscale("log"); ax[0].set_xticks(range(pos)); ax[0].set_xticklabels(ticks, fontsize=8); ax[0].set_ylabel("f_orph (5-95%)"); ax[0].legend(fontsize=8)
ax[0].set_title("Share of vault value not replaceable by local copies")
xs = np.arange(3); bw = 0.25
for k, var in enumerate(("main", "no_debian", "M1_only")):
    ax[1].bar(xs + k * bw, [R["plug_034"][var]["mix"][a]["P_S_gt_TC"] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")], bw, label=var)
ax[1].bar(xs + 3 * bw, [r34["S_vs_TC_under_u"]["main"][a]["P_S_gt_TC"] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")], bw, label="034 (f_orph prior)", color="grey")
ax[1].set_xticks(xs + 1.5 * bw); ax[1].set_xticklabels(["A_Ord", "A_XPT_S", "A_XPT_E"]); ax[1].set_ylabel("P(S > T+C)"); ax[1].set_ylim(0, 1)
ax[1].set_title("Second site vs bundle after plugging f_orph (mix weight)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_forph_and_plug.png", dpi=120); plt.close(fig)

R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("coverage", "mirrors", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("f_orph main mix", {s: round(FM[s]["median"], 4) for s in SCEN})
print("plug main mix", {a: R["plug_034"]["main"]["mix"][a]["P_S_gt_TC"] for a in WTS})
print("runtime", R["runtime_s"])
