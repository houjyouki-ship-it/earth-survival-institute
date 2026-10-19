"""研究057：読取装置より前の段の手引きは重い崩壊で T0→T2 をどれだけ速め、T＋C の価値を何倍にするか（open_problems 153）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
056 の sim.py を「# ---------- 仮説」の前まで exec し（056→055→054→040→039→038 の乱数をそのまま使う）、
054 の BASE の T2 到達時刻の式を写して段の年数の倍率 f と人口の待ちの閾値の倍率 thr を入れる。新しい乱数は seed 20261057。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P56 = os.path.abspath(os.path.join(HERE, "..", "056-manual-char-count", "sim.py"))
src = open(P56, encoding="utf8").read(); cut = src.index("# ---------- 仮説 ----------")
G56 = {"__file__": P56, "__name__": "sim056"}
exec(compile(src[:cut], P56, "exec"), G56)
G55 = G56["G"]; G54 = G55["G"]; ns = G54["ns"]
r56 = json.load(open(os.path.join(HERE, "..", "056-manual-char-count", "results.json")))
R = {"study": "057-pre-reader-knowledge", "seed": 20261057}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p + f"[{i}]")
    else: yield p, x
a_ = dict(leaves(r56["runs"])); b_ = dict(leaves(json.loads(json.dumps(G56["RUN"], default=float))))
R["check_056_runs"] = {"n_leaves": len(a_), "max_abs_rel_diff": max(abs(a_[k] - b_[k]) / max(1e-300, abs(a_[k])) for k in a_ if a_[k] != 0)}

# ---------- 事例（sources/cases.json） ----------
CASES = json.load(open(os.path.join(HERE, "sources", "cases.json")))["cases"]
def t_work(c): return c["t_reliable"] if c["t_reliable"] is not None else c["t_first"]
tb = [t_work(c) for c in CASES if c["mode"] == "book" and t_work(c) is not None]
tp = [t_work(c) for c in CASES if c["mode"] == "person" and t_work(c) is not None]
th = [t_work(c) for c in CASES if c["mode"] == "hybrid" and t_work(c) is not None]
book = [c for c in CASES if c["mode"] == "book"]
R["cases"] = {"t_book": tb, "t_person": tp, "t_hybrid": th, "median_book": float(np.median(tb)), "median_person": float(np.median(tp)),
              "ratio_book_over_person": float(np.median(tb) / np.median(tp)),
              "book_outcomes": {o: sum(c["outcome"] == o for c in book) for o in ("full", "partial", "fail")},
              "book_full_share": sum(c["outcome"] == "full" for c in book) / len(book),
              "person_full_share": sum(c["outcome"] == "full" for c in CASES if c["mode"] == "person") / sum(c["mode"] == "person" for c in CASES)}

# ---------- 054 の状態と T2 到達時刻 ----------
N = G54["N"]; B = G54["B"]; SC = G54["SC"]; COMPS = G54["COMPS"]; SRC = G54["SRC"]; agg = G54["agg"]; only = G54["only"]
V_ext = G54["V_ext"]; U_T3 = G54["U_T3"]; q_TC = G54["q_TC"]
SCEN = ns["SCEN"]; U_lit, d01, d12, lit_mult, d_Fe, W = (ns[k] for k in ("U_lit", "d01", "d12", "lit_mult", "d_Fe", "W"))
N_Fe, N_lens, N_T2e, r_pop = (ns[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop"))
U_lost, U_find, Z, life, H = (ns[k] for k in ("U_lost", "U_find", "Z", "life", "H"))
U_sep, p_sep, t_build, pop_ok, U_t, P_max, kappa, U_acc, p_acc = (ns[k] for k in ("U_sep", "p_sep", "t_build", "pop_ok", "U_t", "P_max", "kappa", "U_acc", "p_acc"))
FIXED = ("AI", "GEN")

def T2s_m(k, fFe=1.0, f01=1.0, f12=1.0, thr=1.0):
    st = B[k]["st"]
    if k in FIXED:
        lit = SCEN[k]["lit"]; d01e = np.where(U_lit < lit, d01 * lit_mult, d01) * f01
        return np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12 * f12, d01e + d12 * f12, W], default=np.zeros(N))
    sv = SRC[k]["sv"]; lit = SRC[k]["lit"]
    n = np.maximum(sv["nmax"], 1.0); r = r_pop * (r_pop / r_pop)
    wait = lambda Nk: np.maximum(0.0, np.log(thr * Nk / n) / r)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01) * f01
    dFe = d_Fe * fFe
    tFe = np.where(st == 6, np.maximum(dFe, wait(N_Fe)), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12 * f12, wait(N_T2e)), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))

R["check_T2s_reproduced"] = max(float(np.abs(T2s_m(k) - B[k]["T2s"]).max()) for k in COMPS)

def V_m(k, qv, tablet, T2s):
    """054 の V_ext（S なし）の写しで、T2 到達時刻だけを外から与える"""
    st, sc = B[k]["st"], SC[k]
    alive = st != 0
    intact1 = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
    T3t = T2s + 943 * sc["M"] * np.exp(0.8 * Z[2])
    gen = (T3t < life) & (T3t <= H)
    cov = np.zeros(N); avail = np.zeros(N, bool)
    if tablet:
        tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, cov); avail |= tab_ok
    Tt = T2s + t_build
    tgt = avail & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    frag_acc = (st == 2) & (U_acc < p_acc)
    v = np.maximum(qv * tgt, gen.astype(float))
    v = np.where(st == 1, U_T3[k], v); v = np.where(frag_acc, 1.0, v)
    return np.where(alive & intact1, v, 0.0)
R["check_V_m_eq_V_ext"] = max(max(float(np.abs(V_m(k, q_TC, True, B[k]["T2s"]) - V_ext(k, q_TC, True)).max()),
                                  float(np.abs(V_m(k, 0.3, False, B[k]["T2s"]) - V_ext(k, 0.3, False)).max())) for k in COMPS)

# ---------- 新しい乱数：早く見つかるか、事例からの f_ev ----------
gq = np.random.default_rng(20261057)
U_early = gq.uniform(size=N); p_early_main = gq.uniform(0.1, 0.6, N)
def lu(lo, hi): return np.exp(gq.uniform(np.log(lo), np.log(hi), N))
tab_ok = U_sep > p_sep
def fev(d_stage):
    return np.minimum(1.0, lu(min(tb), max(tb)) * lu(1, 5) * lu(1, 4) / d_stage)
d01e_any = d01   # f_ev の分母：識字喪失のない d01（喪失時はさらに小さな f になるので保守側＝手引きに有利でない方は避け、別に下で確かめる）
F_EV = {"fFe": fev(d_Fe), "f01": fev(d01e_any), "f12": fev(d12)}
R["f_ev_quantiles"] = {k: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)] for k, v in F_EV.items()}

V0 = {k: V_ext(k, 0.3, False) for k in COMPS}
VTC = {k: V_ext(k, q_TC, True) for k in COMPS}
D_TC = {k: VTC[k] - V0[k] for k in COMPS}
KEYS = [("A_Ord", "U3"), ("A_XPT_S", "U2")]
base_val = agg({"TC": D_TC}, anchors=("A_Ord", "A_XPT_S"))

def F_of(fFe, f01, f12, thr, p_early, decompose=False):
    man = tab_ok & (U_early < p_early)
    D = {}
    for k in COMPS:
        T2f = T2s_m(k, fFe, f01, f12, thr)
        T2 = np.where(man, T2f, B[k]["T2s"])
        D[k] = V_m(k, q_TC, True, T2) - V0[k]
    val = agg({"TCpre": D}, anchors=("A_Ord", "A_XPT_S"))
    out = {}
    for a, u in KEYS:
        b = base_val[a][u]["TC"]; v = val[a][u]["TCpre"]
        out[f"{a}/{u}"] = {"F_total": v["total"] / b["total"], "F_heavy": (v["heavy"] / b["heavy"]) if b["heavy"] > 0 else None,
                           "heavy_share_base": b["heavy"] / b["total"], "gain_total": v["total"] - b["total"]}
    if decompose:
        for a, u in KEYS:
            gain = out[f"{a}/{u}"]["gain_total"]; sh = {}
            for k in COMPS:
                Dk = {"x": {kk: (D[kk] if kk == k else VTC[kk] - V0[kk]) for kk in COMPS}}
                vk = agg(Dk, anchors=(a,))[a][u]["x"]["total"]
                sh[k] = (vk - base_val[a][u]["TC"]["total"]) / gain if gain > 0 else None
            out[f"{a}/{u}"]["gain_share_by_component"] = sh
    return out

FS = [1.0, 0.8, 0.5, 0.3, 0.1, 0.03, 0.0]
VARS = {"pre": dict(f12=False, thr=1.0), "pre12": dict(f12=True, thr=1.0), "pre12_thr0.3": dict(f12=True, thr=0.3), "pre12_thr0.1": dict(f12=True, thr=0.1)}
PE = {"main": p_early_main, "one": 1.0}
SCAN = {}
for vn, vv in VARS.items():
    for pn, pv in PE.items():
        rows = []
        for f in FS:
            o = F_of(f, f, f if vv["f12"] else 1.0, vv["thr"], pv)
            rows.append({"f": f, **o})
        SCAN[f"{vn}/{pn}"] = rows
R["scan"] = SCAN
R["F_ev"] = {"pre12/main": F_of(F_EV["fFe"], F_EV["f01"], F_EV["f12"], 1.0, p_early_main),
             "pre12/one": F_of(F_EV["fFe"], F_EV["f01"], F_EV["f12"], 1.0, 1.0)}
R["extreme_f0_thr1_one"] = F_of(0.0, 0.0, 0.0, 1.0, 1.0, decompose=True)
R["extreme_f0_thr0.1_one"] = F_of(0.0, 0.0, 0.0, 0.1, 1.0)
R["extreme_f0_thr0.01_one"] = F_of(0.0, 0.0, 0.0, 0.01, 1.0)   # 事前登録の外の探索

# ---------- 費用あたり（056 の CH_heavy の T＋C の CE に F を掛ける） ----------
RUN = G56["RUN"]; BE = G56["R"]["breakeven_value_factor_heavy_vs_reader"]
R["breakeven_056"] = BE
def ce(ch, key, x, p="total", ver="V_mix"): return RUN[ch][ver]["CE"][key][p][x]["CE_q"][1]
CEC = {}
for a, u in KEYS:
    key = f"{a}/{u}"
    Fev = R["F_ev"]["pre12/main"][key]["F_total"]; Fx = R["extreme_f0_thr1_one"][key]["F_total"]
    CEC[key] = {"CH_reader_TC": ce("CH_reader", key, "TC"), "CH_heavy_TC_054value": ce("CH_heavy", key, "TC"),
                "CH_heavy_TC_x_F_ev": ce("CH_heavy", key, "TC") * Fev, "CH_heavy_TC_x_F_extreme": ce("CH_heavy", key, "TC") * Fx,
                "CH_heavy_STC": ce("CH_heavy", key, "S_TC_given_TC"), "CH_reader_STC": ce("CH_reader", key, "S_TC_given_TC")}
R["CE_compare_median"] = CEC

# ---------- 仮説 ----------
Hh = {}
Hh["H69a"] = {"value": R["cases"]["ratio_book_over_person"], "rule": "median t_book / median t_person >= 2", "supported": R["cases"]["ratio_book_over_person"] >= 2}
Hh["H69b"] = {"value": R["cases"]["book_full_share"], "rule": "book-only full-success share < 0.75", "supported": R["cases"]["book_full_share"] < 0.75}
v = {k: R["extreme_f0_thr1_one"][k]["F_total"] for k in R["extreme_f0_thr1_one"]}
Hh["H69c"] = {"value": v, "rule": "f=0 (dFe,d01,d12), thr=1, p_early=1: F < 5 both", "supported": all(x < 5 for x in v.values())}
v = {k: R["F_ev"]["pre12/main"][k]["F_total"] for k in R["F_ev"]["pre12/main"]}
Hh["H69d"] = {"value": v, "rule": "F_ev (pre12, thr=1, p_early main) < 1.2 both", "supported": all(x < 1.2 for x in v.values())}
v = R["extreme_f0_thr1_one"]["A_Ord/U3"]["F_heavy"]
Hh["H69e"] = {"value": v, "rule": "f=0, thr=1, p_early=1: heavy-part ratio < 2 (A_Ord/U3)", "supported": v < 2}
mx = max(r["A_Ord/U3"]["F_total"] for rows in SCAN.values() for r in rows)
mx = max(mx, R["extreme_f0_thr0.1_one"]["A_Ord/U3"]["F_total"])
Hh["H69f"] = {"value": mx, "rule": "A_Ord/U3: no scanned combination reaches F >= 5", "supported": mx < 5}
sh = R["extreme_f0_thr1_one"]["A_Ord/U3"]["gain_share_by_component"]
fx = sum(sh[k] for k in FIXED if sh[k] is not None)
Hh["H69g"] = {"value": {"fixed_share": fx, "by_comp": sh}, "rule": "A_Ord/U3 f=0: >=50% of F-1 from AI/GEN fixed components", "supported": fx >= 0.5}
R["hypotheses"] = Hh
R["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, (a, u) in zip(axs, KEYS):
    key = f"{a}/{u}"
    for (vn, pn), col in zip([(v, p) for v in VARS for p in PE], ["C0", "C0", "C1", "C1", "C2", "C2", "C3", "C3"]):
        rows = SCAN[f"{vn}/{pn}"]
        ax.plot([r["f"] for r in rows], [r[key]["F_total"] for r in rows], "-" if pn == "one" else "--", color=col, marker="o", ms=3, label=f"{vn}, p_early={pn}")
    ax.axhline(BE["V_mix"], color="k", lw=1); ax.text(0.5, BE["V_mix"] * 1.03, "break-even 5.0 (V_mix, 056)", fontsize=7, ha="center")
    ax.axhline(1, color="gray", lw=0.6); ax.set_xlabel("stage-duration factor f with the pre-reader manual"); ax.set_title(key, fontsize=9)
    ax.set_yscale("log")
axs[0].set_ylabel("F = value(T+C with pre-reader stages) / value(T+C, 054)"); axs[0].legend(fontsize=6)
fig.suptitle("057: does the pre-reader part of the manual pay for itself? (heavy-collapse stage shortening)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_F_vs_f.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
cs = [c for c in CASES if t_work(c) is not None]
cols = {"book": "C3", "hybrid": "C1", "person": "C0"}
for i, c in enumerate(cs):
    ax.barh(i, t_work(c), color=cols[c["mode"]], alpha=0.7 if c["outcome"] == "full" else 0.35, hatch=None if c["outcome"] == "full" else "//")
    if c["t_first"] is not None and c["t_reliable"] is not None: ax.plot(c["t_first"], i, "k|", ms=12)
ax.set_yticks(range(len(cs))); ax.set_yticklabels([c["id"] for c in cs], fontsize=7)
ax.set_xlabel("years from start of effort to working output (hatched = partial)"); ax.set_title("Book-only (red), book+experienced people (orange), people/memory (blue)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases.png"), dpi=120); plt.close(fig)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in ("check_056_runs", "check_T2s_reproduced", "check_V_m_eq_V_ext", "cases", "f_ev_quantiles", "F_ev",
                                    "extreme_f0_thr1_one", "extreme_f0_thr0.1_one", "extreme_f0_thr0.01_one", "CE_compare_median", "breakeven_056")}, indent=1, ensure_ascii=False, default=float))
for k, rows in SCAN.items():
    print(k, [(r["f"], round(r["A_Ord/U3"]["F_total"], 4), round(r["A_XPT_S/U2"]["F_total"], 4)) for r in rows])
for k, h in Hh.items(): print(k, h["supported"], json.dumps(h["value"], default=float)[:300])
print("runtime", round(R["runtime_s"], 1))
