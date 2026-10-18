"""研究040：重い崩壊で効く対策の比較（人口のまとまり vs 知識）と金庫に価値が出る最小のまとまり n*（open_problems 115・119）
事前登録：PREREGISTRATION.md。039 の sim.py を「仮説」節の前まで exec し、038・039 の乱数列・関数・成分をそのまま使う。
新しい乱数は seed 20261040。numpy＋matplotlib のみ、ネット接続なし。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P39 = os.path.abspath(os.path.join(HERE, "..", "039-gen-decomposition", "sim.py"))
src = open(P39, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P39, "__name__": "sim039"}
exec(compile(src[:cut], P39, "exec"), G)
ns = G["ns"]; N = G["N"]; POP_W = G["POP_W"]
assign = ns["assign"]; times = ns["times"]; V_gen = ns["V_gen"]; DRAWS = ns["DRAWS"]; SCEN = ns["SCEN"]
survivors = ns["survivors"]; HEAVY = ns["HEAVY"]; r_pop = ns["r_pop"]; d01 = ns["d01"]; d12 = ns["d12"]
U_lit = ns["U_lit"]; lit_mult = ns["lit_mult"]; f_primer = ns["f_primer"]; W = ns["W"]; T2s_032 = ns["T2s_032"]
U4 = G["U4"]; lu = G["lu"]; qn = G["qn"]; qo = G["qo"]; anchor_weights = G["anchor_weights"]; RES39 = G["RES"]
life = ns["life"]
r39 = json.load(open(os.path.join(HERE, "..", "039-gen-decomposition", "results.json")))
R = {"study": "040-cohesion-vs-knowledge", "seed": 20261040, "N": N}

# ---------- 新しい乱数 ----------
g5 = np.random.default_rng(20261040)
U5 = g5.uniform(size=(4, N))
p_ref = g5.uniform(0.3, 0.8, N)      # 避難所が働く確率【推測】
p_grow = g5.uniform(0.3, 0.8, N)     # 増殖の知識が r を開拓地水準にする確率【推測】
r_front = lu(U5[2], 0.012, 0.035)    # 開拓地水準の r【推測：Tristan 1.99%, Pitcairn 2.98%, Hutterite 3.45% を目安】

# ---------- 生存者の模型（038・039 と同じ式） ----------
def sv_of(s, nmax, mcap):
    return {"s": s, "ntot": s * POP_W, "nmax": np.minimum(nmax, s * POP_W), "mcap": mcap}
SRC = {}
for s in ("NUC_G", "NUC_R", "BIO", "CASC"):
    SRC[s] = dict(sv=survivors(s), sc=DRAWS[s], lit=SCEN[s]["lit"], ext=0.0, uext=None)
s_nat = G["s_nat"]; SRC["NAT"] = dict(sv=sv_of(s_nat, s_nat * lu(U4[1], 3e8, 8.232e9), np.ones(N)), sc=DRAWS["BIO"], lit=SCEN["BIO"]["lit"], ext=G["EXT_NAT"], uext=U4[2])
s_sv = G["s_sv"]; SRC["SV"] = dict(sv=sv_of(s_sv, s_sv * lu(U4[4], 3.4e8, 4e9), np.ones(N)), sc=DRAWS["GEN"], lit=0.3, ext=G["EXT_NON"], uext=U4[5])
s_as = G["s_as"]; SRC["AS"] = dict(sv=sv_of(s_as, s_as * lu(U4[7], 1e7, 4e9), 1 + 2 * U4[8]), sc=DRAWS["GEN"], lit=0.3, ext=G["EXT_NON"], uext=U4[5])
s_so = G["s_so"]; SRC["SO"] = dict(sv=sv_of(s_so, s_so * lu(U4[10], 3.4e8, 4e9), 1 + 2 * U4[11]), sc=DRAWS["GEN"], lit=0.3, ext=G["EXT_NON"], uext=U4[5])
s_u3 = G["s_u3"]; SRC["U3"] = dict(sv=sv_of(s_u3, s_u3 * POP_W * lu(U4[13], 1e-5, 1.0), 1 + 2 * U4[14]), sc=DRAWS["GEN"], lit=0.3, ext=0.0, uext=U4[15])

IV = ["BASE", "PRIM", "REF_1e3", "REF_1e4", "REF_1e5", "REF_1e6", "CON_0.25", "CON_0.5", "GROW"]
def run_src(d, iv, thr=1.0, pref_one=False, r_world=None):
    sv = dict(d["sv"]); primer = False
    rr = r_pop if r_world is None else r_world
    if iv == "PRIM": primer = True
    elif iv.startswith("REF_"):
        Nr = float(iv[4:]); works = U5[0] < (1.0 if pref_one else p_ref)
        sv["nmax"] = np.where(works, np.maximum(sv["nmax"], Nr), sv["nmax"]); sv["ntot"] = np.where(works, np.maximum(sv["ntot"], Nr), sv["ntot"])
    elif iv.startswith("CON_"):
        c = float(iv[4:]); sv["nmax"] = sv["nmax"] ** (1 - c) * sv["ntot"] ** c
    elif iv == "GROW":
        rr = np.where(U5[1] < p_grow, np.maximum(r_front, rr), rr)
    st = assign(sv, thr)
    if d["ext"] > 0: st = np.where(d["uext"] < d["ext"], 0, st)
    T2s = times(st, sv, d["lit"], primer=primer, r_mult=rr / r_pop, thr_mult=thr)[0]
    V0 = V_gen(st, T2s, d["sc"], 0.3); V2 = V_gen(st, T2s, d["sc"], 0.3, tablet=True)
    return {"V0": V0, "V2": V2, "st": st, "T2s": T2s}

def fixed_src(name, iv):   # AI と GEN（U1）：032 の状態。人口の対策は効かない、PRIM だけ段を縮める
    st = DRAWS[name]["st"]; lit = SCEN[name]["lit"]; sc = DRAWS[name]
    if iv == "PRIM":
        lit_lost = U_lit < lit
        T2s = np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, np.where(lit_lost, d01 * lit_mult, d01) * f_primer + d12, W], default=np.zeros(N))
    else:
        T2s = T2s_032(st, lit)
    return {"V0": V_gen(st, T2s, sc, 0.3), "V2": V_gen(st, T2s, sc, 0.3, tablet=True), "st": st, "T2s": T2s}

def all_arrays(thr=1.0, pref_one=False, r_world=None):
    out = {}
    for iv in IV:
        o = {k: run_src(SRC[k], iv, thr, pref_one, r_world) for k in SRC}
        for k in ("AI", "GEN"): o[k] = fixed_src(k, iv)
        out[iv] = o
    return out

def deltas(A):
    """各対策の増分（標本ごと）と重い崩壊の印（BASE の状態）"""
    B = A["BASE"]; D = {}
    D["TAB"] = {k: B[k]["V2"] - B[k]["V0"] for k in B}
    for iv in IV[1:]:
        D[iv] = {k: A[iv][k]["V2"] - B[k]["V2"] for k in B}
    heavy = {k: np.isin(B[k]["st"], HEAVY).astype(float) for k in B}
    return D, heavy

def comp_keys(Dx, hv):
    """039 の成分名に組み直す（NON・OTH は混合）"""
    c = {k: Dx[k] for k in ("AI", "BIO", "NUC_G", "NUC_R", "CASC", "NAT")}
    ch = {k: Dx[k] * hv[k] for k in c}
    c["NON"] = qn[:, 0] * Dx["SV"] + qn[:, 1] * Dx["AS"] + qn[:, 2] * Dx["SO"]
    ch["NON"] = qn[:, 0] * Dx["SV"] * hv["SV"] + qn[:, 1] * Dx["AS"] * hv["AS"] + qn[:, 2] * Dx["SO"] * hv["SO"]
    c["OTH"] = (qo[:, 1] + qo[:, 3]) * Dx["AI"] + qo[:, 2] * Dx["BIO"]
    ch["OTH"] = (qo[:, 1] + qo[:, 3]) * Dx["AI"] * hv["AI"] + qo[:, 2] * Dx["BIO"] * hv["BIO"]
    return c, ch

def weighted(D, hv, anchors=("A_Ord", "A_XPT_S", "A_XPT_E")):
    res = {}
    for a in anchors:
        w, _ = anchor_weights(a); res[a] = {}
        for u in ("U1", "U2", "U3"):
            res[a][u] = {}
            for x, Dx in D.items():
                c, ch = comp_keys(Dx, hv)
                kn = [k for k in w if k != "UNK"]
                if u == "U1": cu, chu = Dx["GEN"], Dx["GEN"] * hv["GEN"]
                elif u == "U3": cu, chu = Dx["U3"], Dx["U3"] * hv["U3"]
                else:
                    tw = sum(w[k] for k in kn); cu = sum(w[k] * c[k] for k in kn) / tw; chu = sum(w[k] * ch[k] for k in kn) / tw
                tot = sum(w[k] * c[k] for k in kn) + w["UNK"] * cu
                toth = sum(w[k] * ch[k] for k in kn) + w["UNK"] * chu
                res[a][u][x] = {"total": float(np.mean(tot)), "from_heavy": float(np.mean(toth))}
    return res

def per_heavy(D, hv, keys=("U3", "CASC", "BIO", "AS", "GEN")):
    out = {}
    for k in keys:
        m = hv[k] > 0
        out[k] = {"heavy_share": float(m.mean()), **({x: float(D[x][k][m].mean()) for x in D} if m.any() else {})}
    # 名前のつく原因の重い崩壊をまとめる（U3・GEN を除く、標本数で重み）
    pool = [k for k in ("NUC_G", "NUC_R", "BIO", "CASC", "NAT", "SV", "AS", "SO") if (hv[k] > 0).any()]
    out["named_pooled"] = {x: float(np.concatenate([D[x][k][hv[k] > 0] for k in pool]).mean()) for x in D}
    out["named_pooled"]["n"] = int(sum((hv[k] > 0).sum() for k in pool))
    return out

A = all_arrays(); D, HV = deltas(A)
# ---------- 検査：039 の重み付き R_tab（TAB）を再現 ----------
WB = weighted({"TAB": D["TAB"]}, HV)
chk = max(abs(WB[a][u]["TAB"]["total"] - r39["weighted_039"][a][u]["R_tab"]) for a in WB for u in WB[a])
chkh = max(abs(WB[a][u]["TAB"]["from_heavy"] - r39["weighted_039"][a][u]["R_tab_from_heavy"]) for a in WB for u in WB[a])
R["check_039"] = {"max_abs_diff_R_tab": chk, "max_abs_diff_R_tab_from_heavy": chkh, "reproduced": bool(chk < 1e-12 and chkh < 1e-12)}

R["weighted"] = weighted(D, HV)
R["per_heavy_event"] = per_heavy(D, HV)
R["heavy_nmax_quantiles"] = {k: [float(x) for x in np.percentile(SRC[k]["sv"]["nmax"][HV[k] > 0], [5, 50, 95])] for k in ("U3", "CASC", "BIO") if (HV[k] > 0).any()}

# ---------- 感度（判定に使わない） ----------
SENS = {"p_ref_1": dict(pref_one=True), "thr_x0.3": dict(thr=0.3), "thr_x3": dict(thr=3.0), "r_frontier_world": dict(r_world=r_front)}
R["sensitivity"] = {}
for k, kw in SENS.items():
    Ak = all_arrays(**kw); Dk, Hk = deltas(Ak)
    Wk = weighted(Dk, Hk, anchors=("A_Ord", "A_XPT_S"))
    R["sensitivity"][k] = {"per_heavy_event": per_heavy(Dk, Hk, keys=("U3", "CASC")),
                           "weighted": {a: {u: {x: Wk[a][u][x]["total"] for x in Wk[a][u]} for u in ("U2", "U3")} for a in Wk}}
    del Ak

# ---------- n* の走査（119） ----------
NG = np.logspace(np.log10(5e3), 9, 61)
REF_U1 = float((D["TAB"]["GEN"]).mean())     # U1（GEN、T0 再出発・人口待ちなし）の1件あたり R_tab
sc = DRAWS["GEN"]
def sweep(r_arr, thr=1.0, no_wait=False):
    rows = []
    for n in NG:
        sv = {"s": np.full(N, n / POP_W), "ntot": np.full(N, n), "nmax": np.full(N, n), "mcap": np.ones(N)}
        st = assign(sv, thr); T2s = times(st, sv, 0.3, r_mult=r_arr / r_pop, thr_mult=thr, no_wait=no_wait)[0]
        V0 = V_gen(st, T2s, sc, 0.3); V2 = V_gen(st, T2s, sc, 0.3, tablet=True)
        rows.append({"n": float(n), "R_tab": float((V2 - V0).mean()), "V2": float(V2.mean()), "heavy": float(np.isin(st, HEAVY).mean()),
                     "light": float(np.isin(st, (1, 2)).mean()), "T2s_median_alive": float(np.median(T2s[st != 0])) if (st != 0).any() else None})
    v = np.array([r["R_tab"] for r in rows]); ns_ = None
    idx = np.where(v >= 0.5 * REF_U1)[0]
    if idx.size:
        i = int(idx[0])
        if i == 0: ns_ = float(NG[0])
        else:
            x0, x1 = np.log10(NG[i - 1]), np.log10(NG[i]); y0, y1 = v[i - 1], v[i]
            ns_ = float(10 ** (x0 + (0.5 * REF_U1 - y0) / (y1 - y0) * (x1 - x0)))
    return {"rows": rows, "n_star": ns_, "max_R_tab": float(v.max()), "n_at_max": float(NG[int(v.argmax())])}
SW = {"r_blackdeath": sweep(r_pop), "r_frontier": sweep(r_front), "no_wait": sweep(r_pop, no_wait=True),
      "r_blackdeath_thr_x0.3": sweep(r_pop, thr=0.3), "r_blackdeath_thr_x3": sweep(r_pop, thr=3.0), "r_x3": sweep(3 * r_pop)}
R["n_star"] = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in SW.items()}
R["n_star_rows"] = {k: v["rows"] for k, v in SW.items()}
R["REF_U1_per_event"] = REF_U1
R["media_life_quantiles"] = [float(x) for x in np.percentile(life, [5, 50, 95])]
R["r_quantiles"] = {"blackdeath": [float(x) for x in np.percentile(r_pop, [5, 50, 95])], "frontier": [float(x) for x in np.percentile(r_front, [5, 50, 95])]}

# ---------- 仮説 ----------
PH = R["per_heavy_event"]; WT_ = R["weighted"]
pop_iv = [x for x in IV if x.startswith(("REF", "CON")) or x == "GROW"]
nb, nf = R["n_star"]["r_blackdeath"]["n_star"], R["n_star"]["r_frontier"]["n_star"]
xs = WT_["A_XPT_S"]["U2"]; xo = WT_["A_Ord"]["U3"]
R["hypotheses"] = {
 "H52a": {"value": nb, "rule": "n*(Black Death r) >= 1e5", "supported": bool(nb is not None and nb >= 1e5)},
 "H52b": {"value": [nb, nf], "rule": "n*(frontier r) <= n*(BD r)/10", "supported": bool(nb is not None and nf is not None and nf <= nb / 10)},
 "H52c": {"value": [PH["U3"]["REF_1e5"], PH["U3"]["PRIM"]], "rule": "U3 heavy: REF_1e5 > PRIM", "supported": bool(PH["U3"]["REF_1e5"] > PH["U3"]["PRIM"])},
 "H52d": {"value": [PH["U3"]["REF_1e4"], PH["U3"]["PRIM"]], "rule": "U3 heavy: REF_1e4 < PRIM", "supported": bool(PH["U3"]["REF_1e4"] < PH["U3"]["PRIM"])},
 "H52e": {"value": [PH["CASC"].get("CON_0.5"), PH["CASC"].get("REF_1e5")], "rule": "CASC heavy: CON_0.5 > REF_1e5",
          "supported": bool(PH["CASC"].get("CON_0.5", -1) > PH["CASC"].get("REF_1e5", 0))},
 "H52f": {"value": {x: xs[x]["total"] / xs["TAB"]["total"] for x in pop_iv}, "rule": "A_XPT_S U2: every population intervention < 0.3 x TAB",
          "supported": bool(all(xs[x]["total"] < 0.3 * xs["TAB"]["total"] for x in pop_iv))},
 "H52g": {"value": [max(xo[x]["total"] for x in pop_iv), xo["PRIM"]["total"]], "rule": "A_Ord U3: max population intervention > PRIM",
          "supported": bool(max(xo[x]["total"] for x in pop_iv) > xo["PRIM"]["total"])},
 "H52h": {"value": [PH["U3"]["GROW"], PH["U3"]["REF_1e5"]], "rule": "U3 heavy: GROW > REF_1e5", "supported": bool(PH["U3"]["GROW"] > PH["U3"]["REF_1e5"])},
}

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for k, c, lab in (("r_blackdeath", "C3", "r: Black Death recovery (0.14-0.61%/yr)"), ("r_frontier", "C2", "r: frontier (1.2-3.5%/yr)"),
                  ("r_x3", "C1", "r x3"), ("no_wait", "k", "no population wait")):
    rows = SW[k]["rows"]
    ax[0].semilogx([r["n"] for r in rows], [r["R_tab"] for r in rows], c, label=lab)
    if SW[k]["n_star"]: ax[0].axvline(SW[k]["n_star"], color=c, ls=":", lw=0.8)
ax[0].axhline(0.5 * REF_U1, color="gray", ls="--", lw=0.8, label="half of U1 value")
ax[0].set_xlabel("size of the single connected surviving group n"); ax[0].set_ylabel("tablet-guide value per event (R_tab)")
ax[0].set_title("119: minimum cohesive group for the vault to matter (dotted = n*)"); ax[0].legend(fontsize=8)
rows = SW["r_blackdeath"]["rows"]
ax[1].semilogx([r["n"] for r in rows], [r["heavy"] for r in rows], "C3", label="heavy state (T1/T0/T00)")
ax[1].semilogx([r["n"] for r in rows], [r["light"] for r in rows], "C0", label="computer society kept (T3)")
ax2 = ax[1].twinx(); ax2.loglog([r["n"] for r in rows], [max(r["T2s_median_alive"] or 1e-1, 1e-1) for r in rows], "C3--")
ax2.loglog([r["n"] for r in SW["r_frontier"]["rows"]], [max(r["T2s_median_alive"] or 1e-1, 1e-1) for r in SW["r_frontier"]["rows"]], "C2--")
ax2.axhline(R["media_life_quantiles"][1], color="gray", ls=":"); ax2.set_ylabel("median years to T2 (dashed); dotted = median media life")
ax[1].set_xlabel("n"); ax[1].set_ylabel("share of outcomes"); ax[1].legend(fontsize=8, loc="center left")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_n_star.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
xsl = ["TAB"] + IV[1:]; xx = np.arange(len(xsl))
for j, (k, c) in enumerate((("U3", "C3"), ("CASC", "C1"), ("named_pooled", "C0"))):
    ax[0].bar(xx + (j - 1) * 0.27, [PH[k].get(x, 0.0) for x in xsl], 0.26, color=c, label=f"{k} heavy events")
ax[0].set_xticks(xx); ax[0].set_xticklabels(xsl, rotation=30, fontsize=8); ax[0].set_ylabel("value gain per heavy-collapse event")
ax[0].set_title("115: intervention gain inside heavy collapse (TAB vs none; others vs TAB)"); ax[0].legend(fontsize=8)
combos = [("A_Ord", "U1"), ("A_Ord", "U3"), ("A_XPT_S", "U2"), ("A_XPT_S", "U3")]
for j, (a, u) in enumerate(combos):
    ax[1].bar(xx + (j - 1.5) * 0.2, [WT_[a][u][x]["total"] for x in xsl], 0.19, label=f"{a} {u}")
ax[1].set_xticks(xx); ax[1].set_xticklabels(xsl, rotation=30, fontsize=8); ax[1].set_ylabel("weighted value gain (all collapse outcomes)")
ax[1].set_title("same interventions, anchor-weighted"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_interventions.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"check": R["check_039"], "per_heavy": PH, "heavy_nmax": R["heavy_nmax_quantiles"],
                  "weighted": {a: {u: {x: round(v["total"], 6) for x, v in R["weighted"][a][u].items()} for u in R["weighted"][a]} for a in R["weighted"]},
                  "n_star": R["n_star"], "REF_U1": REF_U1, "life": R["media_life_quantiles"], "r": R["r_quantiles"],
                  "sens": R["sensitivity"], "hyp": R["hypotheses"]}, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
