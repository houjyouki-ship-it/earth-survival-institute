"""研究039：GEN（未知・その他）の分解（open_problems 113）
事前登録：PREREGISTRATION.md。038 の sim.py の Kline & Boyd より前を exec して 038 の乱数列・関数をそのまま使い、
GEN を名前のつく成分（XPT 表2・Ord 表6.1 の内訳）と本当に未知の残り（UNK）に分けて重い崩壊の割合と価値を置き直す。
numpy＋matplotlib のみ、ネット接続なし。新しい乱数は seed 20261390。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P38 = os.path.abspath(os.path.join(HERE, "..", "038-restart-state-evidence", "sim.py"))
src = open(P38, encoding="utf8").read(); cut = src.index("# ---------- Kline & Boyd")
ns = {"__file__": P38, "__name__": "sim038"}
exec(compile(src[:cut], P38, "exec"), ns)
N = ns["N"]; POP_W = ns["POP_W"]; logu = ns["logu"]; survivors = ns["survivors"]; assign = ns["assign"]; times = ns["times"]
recs_gen = ns["recs_gen"]; T2s_032 = ns["T2s_032"]; DRAWS = ns["DRAWS"]; SCEN = ns["SCEN"]; WT = ns["WT"]; HEAVY = ns["HEAVY"]
BASE = ns["BASE"]; BASE_W = ns["BASE_W"]
r38 = json.load(open(os.path.join(HERE, "..", "038-restart-state-evidence", "results.json")))
R = {"study": "039-gen-decomposition", "seed": 20261390, "N": N}

# ---------- 検査：038 の再現 ----------
d1 = max(abs(BASE_W[a][k] - r38["weighted_038"][a][k]) for a in BASE_W for k in BASE_W[a])
d2 = max(abs(BASE[s]["heavy_share"] - r38["scenarios_038"][s]["heavy_share"]) for s in BASE)
R["check_038"] = {"max_abs_diff_weighted": d1, "max_abs_diff_heavy_share": d2, "reproduced_exactly": bool(d1 < 1e-12 and d2 < 1e-12)}

# ---------- 名前のついたシナリオの標本ごとの値（038 と同じ） ----------
KEYS = ("R_tab", "R_c4", "R_dep")
def pack(st, rc):
    o = {k: rc[k] for k in KEYS}; o["heavy"] = np.isin(st, HEAVY).astype(float); o["light"] = np.isin(st, (1, 2)).astype(float)
    o["ext"] = (st == 0).astype(float); return o
ARR = {}
for s in ("NUC_G", "NUC_R", "BIO", "CASC"):
    sv = survivors(s); st = assign(sv); T2s = times(st, sv, SCEN[s]["lit"])[0]
    ARR[s] = pack(st, recs_gen(st, T2s, DRAWS[s]))
for s in ("AI", "GEN"):
    st = DRAWS[s]["st"]; ARR[s] = pack(st, recs_gen(st, T2s_032(st, SCEN[s]["lit"]), DRAWS[s]))
R["check_arrays_vs_038"] = max(abs(ARR[s]["R_tab"].mean() - BASE[s]["R_tab"]) for s in ARR)

# ---------- 新しい乱数 ----------
g4 = np.random.default_rng(20261390)
U4 = g4.uniform(size=(16, N))
def lu(u, lo, hi): return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))
def comp(s, nmax, mcap, sc, lit, ext_frac, u_ext):
    sv = {"s": s, "ntot": s * POP_W, "nmax": np.minimum(nmax, s * POP_W), "mcap": mcap}
    st = assign(sv); st = np.where(u_ext < ext_frac, 0, st)
    T2s = times(st, sv, lit)[0]
    o = pack(st, recs_gen(st, T2s, sc)); o["nmax_q"] = [float(x) for x in np.percentile(sv["nmax"], [5, 50, 95])]
    o["states"] = {ns["NAMES7"][i]: float(np.mean(st == i)) for i in range(7)}
    return o
EXT_NAT = 0.0018 / 1.0      # 【事実】XPT 表3/表2 超予測者 自然病原体 絶滅/大災害
EXT_NON = 0.0043 / 0.05     # 【事実】XPT 表3/表2 超予測者 非人為 絶滅/大災害
# 自然のパンデミック：死亡 logU(0.10,0.60)【推測、上端は黒死病の欧州 30〜60%】。設備は無傷（BIO の金庫の係数）
s_nat = 1 - lu(U4[0], 0.10, 0.60)
NAT = comp(s_nat, s_nat * lu(U4[1], 3e8, 8.232e9), np.ones(N), DRAWS["BIO"], SCEN["BIO"]["lit"], EXT_NAT, U4[2])
# 非人為：超巨大噴火・2 km 級衝突・太陽嵐【死亡の幅はすべて推測】
s_sv = 1 - lu(U4[3], 0.10, 0.37); SVc = comp(s_sv, s_sv * lu(U4[4], 3.4e8, 4e9), np.ones(N), DRAWS["GEN"], 0.3, EXT_NON, U4[5])
s_as = 1 - (0.10 + 0.80 * U4[6]); ASc = comp(s_as, s_as * lu(U4[7], 1e7, 4e9), 1 + 2 * U4[8], DRAWS["GEN"], 0.3, EXT_NON, U4[5])
s_so = 1 - lu(U4[9], 0.10, 0.40); SOc = comp(s_so, s_so * lu(U4[10], 3.4e8, 4e9), 1 + 2 * U4[11], DRAWS["GEN"], 0.3, EXT_NON, U4[5])
qn = g4.dirichlet(np.ones(3), size=N)
def mix(parts, wts):
    o = {}
    for k in KEYS + ("heavy", "light", "ext"):
        o[k] = sum(w * p[k] for p, w in zip(parts, wts))
    return o
NON = mix([SVc, ASc, SOc], [qn[:, 0], qn[:, 1], qn[:, 2]])
# Ord のその他の人為：ディストピア（T3 で閉じる＝金庫の価値0）、ナノテク→AI、逆汚染→BIO、異星人→AI、実験→絶滅
Z0 = {k: np.zeros(N) for k in KEYS + ("heavy", "light", "ext")}
DYS = dict(Z0); DYS["light"] = np.ones(N)
EXP = dict(Z0); EXP["ext"] = np.ones(N)
qo = g4.dirichlet(np.ones(5), size=N)
OTH = mix([DYS, ARR["AI"], ARR["BIO"], ARR["AI"], EXP], [qo[:, i] for i in range(5)])
# UNK の事前分布 U3：無知（生存割合 logU(1e-7,1)、つながった割合 logU(1e-5,1)）
s_u3 = lu(U4[12], 1e-7, 1.0)
U3 = comp(s_u3, s_u3 * POP_W * lu(U4[13], 1e-5, 1.0), 1 + 2 * U4[14], DRAWS["GEN"], 0.3, 0.0, U4[15])
R["components"] = {n: {"heavy_share": float(c["heavy"].mean()), "light_share": float(c["light"].mean()), "ext_share": float(c["ext"].mean()),
                       "R_tab": float(c["R_tab"].mean()), **({"states": c["states"], "nmax_q": c["nmax_q"]} if "states" in c else {})}
                   for n, c in (("NAT_PAND", NAT), ("SUPERVOLCANO", SVc), ("ASTEROID_2km", ASc), ("SOLAR", SOc), ("NONANTH_mix", NON),
                                ("OTH_ord", OTH), ("UNK_U1_T0", ARR["GEN"]), ("UNK_U3_agnostic", U3))}

# ---------- 錨ごとの重み ----------
r_g = ns["r_g"]
def anchor_weights(name):
    """返り値：名前→標本ごとの重み（和1）、GEN の名前のある成分の名前の集合"""
    if name in ("A_Ord", "A_XPT_S", "A_XPT_E"):
        w = {k: v for k, v in WT[name]["_w"].items()}
        wG = w.pop("GEN")
        if name == "A_Ord":
            u, o = 1 / 30, 1 / 50
            w["UNK"] = wG * u / (u + o); w["OTH"] = wG * o / (u + o); named = ("OTH",)
        elif name == "A_XPT_S":
            bG = WT[name]["base"]["GEN"]
            w["NAT"] = wG * 1.0 / bG; w["NON"] = wG * 0.05 / bG; w["UNK"] = wG * (bG - 1.05) / bG; named = ("NAT", "NON")
        else:
            w["UNK"] = wG; named = ()
        return w, named
    if name in ("A_XPT_S_D2", "A_XPT_E_D2"):
        t = {"A_XPT_S_D2": (2.13, 0.8, 1.0, 4.0, 0.05, 9.05), "A_XPT_E_D2": (12, 3, 0.85, 8, 0.09, 20)}[name]
        ai, bio, nat, nuc, non, tot = t; rest = max(0.0, tot - ai - bio - nat - nuc - non)
        base = {"AI": ai, "BIO": bio, "NAT": nat, "NON": non, "CASC": rest / 2, "UNK": rest / 2}
        T = sum(base.values()) + nuc
        w = {k: v / T * np.ones(N) for k, v in base.items()}
        w["NUC_G"] = nuc / T * r_g; w["NUC_R"] = nuc / T * (1 - r_g)
        return w, ("NAT", "NON")
COMP = {"NAT": NAT, "NON": NON, "OTH": OTH, **{s: ARR[s] for s in ("AI", "BIO", "NUC_G", "NUC_R", "CASC")}}
def evaluate(name, unk):
    w, named = anchor_weights(name)
    if unk == "U1": U = ARR["GEN"]
    elif unk == "U3": U = U3
    elif unk == "U2":
        kn = [k for k in w if k != "UNK"]; tw = sum(w[k] for k in kn)
        U = {m: sum(w[k] * COMP[k][m] for k in kn) / tw for m in KEYS + ("heavy", "light", "ext")}
    elif isinstance(unk, tuple):   # ("h", h)：U1 の割合 h、残りは U2
        h = unk[1]; a = evaluate(name, "U1")["_U"]; b = evaluate(name, "U2")["_U"]
        U = {m: h * a[m] + (1 - h) * b[m] for m in KEYS + ("heavy", "light", "ext")}
    C = dict(COMP); C["UNK"] = U
    out = {}
    for m in KEYS:
        out[m] = float(np.mean(sum(w[k] * C[k][m] for k in w)))
    # 重い崩壊からの価値：混合の成分（NON・OTH・U2）では heavy が割合になるので、HV で成分ごとに分解して足す。
    hv = 0.0
    for k in w:
        hv = hv + w[k] * HVAL[k](C, U)
    out["R_tab_from_heavy"] = float(np.mean(hv))
    out["heavy_value_share"] = out["R_tab_from_heavy"] / out["R_tab"]
    out["heavy_share_weighted"] = float(np.mean(sum(w[k] * C[k]["heavy"] for k in w)))
    out["ext_share_weighted"] = float(np.mean(sum(w[k] * C[k]["ext"] for k in w)))
    out["UNK_weight"] = float(np.mean(w.get("UNK", 0.0)))
    out["UNK_heavy_value_share_of_heavy"] = float(np.mean(w["UNK"] * HVAL["UNK"](C, U)) / out["R_tab_from_heavy"]) if "UNK" in w and out["R_tab_from_heavy"] > 0 else None
    out["weights_mean"] = {k: float(np.mean(v)) for k, v in w.items()}
    out["_U"] = U
    return out
# 重い崩壊からの価値（正確に分解）
HV = {s: ARR[s]["R_tab"] * ARR[s]["heavy"] for s in ARR}
HV["NAT"] = NAT["R_tab"] * NAT["heavy"]
HV["NON"] = qn[:, 0] * SVc["R_tab"] * SVc["heavy"] + qn[:, 1] * ASc["R_tab"] * ASc["heavy"] + qn[:, 2] * SOc["R_tab"] * SOc["heavy"]
HV["OTH"] = (qo[:, 1] + qo[:, 3]) * HV["AI"] + qo[:, 2] * HV["BIO"]
HV["U3"] = U3["R_tab"] * U3["heavy"]
HVAL = {k: (lambda C, U, k=k: HV[k]) for k in ("NAT", "NON", "OTH", "AI", "BIO", "NUC_G", "NUC_R", "CASC")}
CUR = {}
def hval_unk(C, U):
    return CUR["unk_hv"]
HVAL["UNK"] = hval_unk
def run(name, unk):
    w, named = anchor_weights(name)
    if unk == "U1": CUR["unk_hv"] = HV["GEN"]
    elif unk == "U3": CUR["unk_hv"] = HV["U3"]
    elif unk == "U2":
        kn = [k for k in w if k != "UNK"]; tw = sum(w[k] for k in kn); CUR["unk_hv"] = sum(w[k] * HV[k] for k in kn) / tw
    else:
        h = unk[1]; kn = [k for k in w if k != "UNK"]; tw = sum(w[k] for k in kn)
        CUR["unk_hv"] = h * HV["GEN"] + (1 - h) * sum(w[k] * HV[k] for k in kn) / tw
    return evaluate(name, unk)

ANCH = ("A_Ord", "A_XPT_S", "A_XPT_E", "A_XPT_S_D2", "A_XPT_E_D2")
RES = {}
for a in ANCH:
    RES[a] = {}
    for u in ("U1", "U2", "U3"):
        o = run(a, u); o.pop("_U"); RES[a][u] = o
R["weighted_039"] = RES
# 038 の値（GEN を分けない）
R["ref_038"] = {a: {"heavy_share_weighted": r38["weighted_038"][a]["heavy_share_weighted"], "R_tab": r38["weighted_038"][a]["R_tab"],
                    "R_tab_from_heavy": r38["weighted_038"][a]["R_tab_from_heavy"],
                    "heavy_value_share": r38["weighted_038"][a]["R_tab_from_heavy"] / r38["weighted_038"][a]["R_tab"]} for a in ("A_Ord", "A_XPT_S", "A_XPT_E")}

# ---------- h の走査 ----------
HS = np.linspace(0, 1, 21); SCAN = {}
for a in ("A_Ord", "A_XPT_S", "A_XPT_S_D2"):
    rows = []
    for h in HS:
        o = run(a, ("h", float(h))); o.pop("_U")
        rows.append({"h": float(h), "heavy_share_weighted": o["heavy_share_weighted"], "heavy_value_share": o["heavy_value_share"], "R_tab": o["R_tab"]})
    v = np.array([r["heavy_value_share"] for r in rows])
    hstar = None
    if v[0] < 0.5 <= v[-1]:
        i = int(np.argmax(v >= 0.5)); hstar = float(HS[i - 1] + (0.5 - v[i - 1]) / (v[i] - v[i - 1]) * (HS[i] - HS[i - 1]))
    elif v[0] >= 0.5: hstar = 0.0
    SCAN[a] = {"rows": rows, "h_star_value_share_0.5": hstar}
R["h_scan"] = SCAN

# ---------- 仮説 ----------
wO, _ = anchor_weights("A_Ord"); wS, _ = anchor_weights("A_XPT_S")
named_heavy_Ord = float(OTH["heavy"].mean())
named_heavy_S = float(np.mean((wS["NAT"] * NAT["heavy"] + wS["NON"] * NON["heavy"]) / (wS["NAT"] + wS["NON"])))
R["hypotheses"] = {
 "H51a": {"value": {"A_Ord_OTH": named_heavy_Ord, "A_XPT_S_named": named_heavy_S}, "rule": "named part of GEN heavy share < 0.05 (A_Ord, A_XPT_S D1)",
          "supported": bool(named_heavy_Ord < 0.05 and named_heavy_S < 0.05)},
 "H51b": {"value": {"A_XPT_S": RES["A_XPT_S"]["U1"]["heavy_share_weighted"], "A_Ord": RES["A_Ord"]["U1"]["heavy_share_weighted"]},
          "rule": "U1: A_XPT_S heavy < 0.192/2=0.096 and A_Ord heavy > 0.302/2=0.151",
          "supported": bool(RES["A_XPT_S"]["U1"]["heavy_share_weighted"] < r38["weighted_038"]["A_XPT_S"]["heavy_share_weighted"] / 2 and
                            RES["A_Ord"]["U1"]["heavy_share_weighted"] > r38["weighted_038"]["A_Ord"]["heavy_share_weighted"] / 2)},
 "H51c": {"value": {a: [RES[a]["U2"]["heavy_share_weighted"], RES[a]["U2"]["heavy_value_share"]] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")},
          "rule": "U2: heavy share < 0.05 and heavy value share < 0.5 for all three anchors",
          "supported": bool(all(RES[a]["U2"]["heavy_share_weighted"] < 0.05 and RES[a]["U2"]["heavy_value_share"] < 0.5 for a in ("A_Ord", "A_XPT_S", "A_XPT_E")))},
 "H51d": {"value": float(U3["heavy"].mean()), "rule": "0.2 <= U3 heavy share <= 0.8", "supported": bool(0.2 <= U3["heavy"].mean() <= 0.8)},
 "H51e": {"value": SCAN["A_Ord"]["h_star_value_share_0.5"], "rule": "A_Ord h* > 0.5", "supported": bool(SCAN["A_Ord"]["h_star_value_share_0.5"] is not None and SCAN["A_Ord"]["h_star_value_share_0.5"] > 0.5)},
 "H51f": {"value": RES["A_XPT_S_D2"]["U1"]["R_tab"] / RES["A_XPT_S"]["U1"]["R_tab"], "rule": "|D2/D1 - 1| <= 0.2 (U1)",
          "supported": bool(abs(RES["A_XPT_S_D2"]["U1"]["R_tab"] / RES["A_XPT_S"]["U1"]["R_tab"] - 1) <= 0.2)},
}

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
labs = ["A_Ord", "A_XPT_S", "A_XPT_S_D2"]; cols = {"UNK": "#7f0000", "OTH": "#ff7f0e", "NAT": "#2ca02c", "NON": "#9467bd"}
for i, a in enumerate(labs):
    w = RES[a]["U1"]["weights_mean"]; parts = {k: w.get(k, 0.0) for k in ("UNK", "OTH", "NAT", "NON")}
    tot = sum(parts.values()); b = 0
    for k, v in parts.items():
        ax[0].bar(i, v / tot if tot else 0, bottom=b, color=cols[k], label=k if i == 0 else None); b += v / tot if tot else 0
ax[0].set_xticks(range(len(labs))); ax[0].set_xticklabels(labs); ax[0].set_ylabel("share of the 'unknown/other' weight")
ax[0].set_title("what GEN contains (XPT Table 2, Ord Table 6.1)"); ax[0].legend(fontsize=8)
cn = ["NAT_PAND", "SUPERVOLCANO", "ASTEROID_2km", "SOLAR", "OTH_ord", "UNK_U1_T0", "UNK_U3_agnostic"]
hvals = [R["components"][c]["heavy_share"] for c in cn]; evals = [R["components"][c]["ext_share"] for c in cn]
xx = np.arange(len(cn)); ax[1].bar(xx - 0.2, hvals, 0.4, color="#d62728", label="heavy (T1/T0/T00)"); ax[1].bar(xx + 0.2, evals, 0.4, color="#555", label="extinct")
ax[1].set_xticks(xx); ax[1].set_xticklabels(cn, rotation=25, fontsize=8); ax[1].set_ylabel("share of outcomes"); ax[1].legend(fontsize=8)
ax[1].set_title("restart depth by GEN component (038 model)")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_gen_components.png"), dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for a, c in zip(("A_Ord", "A_XPT_S", "A_XPT_S_D2"), ("C0", "C1", "C2")):
    rows = SCAN[a]["rows"]
    ax[0].plot(HS, [r["heavy_share_weighted"] for r in rows], c, label=a); ax[1].plot(HS, [r["heavy_value_share"] for r in rows], c, label=a)
    if a in R["ref_038"]:
        ax[0].axhline(R["ref_038"][a]["heavy_share_weighted"], color=c, ls=":"); ax[1].axhline(R["ref_038"][a]["heavy_value_share"], color=c, ls=":")
ax[1].axhline(0.5, color="k", lw=0.8)
ax[0].set_xlabel("h = share of truly-unknown risk that restarts from T0 (rest behaves like named risks)"); ax[0].set_ylabel("weighted heavy-collapse share")
ax[1].set_xlabel("h"); ax[1].set_ylabel("share of tablet-guide value from heavy collapse"); ax[0].legend(); ax[1].legend()
ax[0].set_title("dotted = 038 (GEN undecomposed)"); ax[1].set_title("heavy-collapse priority needs h above the 0.5 crossing")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_h_scan.png"), dpi=120); plt.close()

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"check_038": R["check_038"], "check_arrays": R["check_arrays_vs_038"], "components": R["components"],
                  "weighted": {a: {u: {k: v for k, v in RES[a][u].items() if k != "weights_mean"} for u in RES[a]} for a in RES},
                  "hstar": {a: SCAN[a]["h_star_value_share_0.5"] for a in SCAN}, "hyp": R["hypotheses"]}, ensure_ascii=False, indent=1, default=float))
print("runtime", round(R["runtime_seconds"], 1))
