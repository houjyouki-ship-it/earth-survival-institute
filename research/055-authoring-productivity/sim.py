"""研究055：執筆の生産性の公開アンカーと、束・南半球の写し・法定納本の費用あたりの順位（open_problems 146＋147）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
054 の sim.py を「# ---------- 費用（2025年ドル）」の前まで exec し、054 の価値を差0で再現してから使う。新しい乱数は seed 20261055。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P54 = os.path.abspath(os.path.join(HERE, "..", "054-site2-vs-bundle-cost", "sim.py"))
src = open(P54, encoding="utf8").read(); cut = src.index("# ---------- 費用（2025年ドル）")
G = {"__file__": P54, "__name__": "sim054"}
exec(compile(src[:cut], P54, "exec"), G)
N = G["N"]; COMPS = G["COMPS"]; B = G["B"]; SC = G["SC"]; VAL54 = G["VAL"]; agg = G["agg"]
r54 = json.load(open(os.path.join(HERE, "..", "054-site2-vs-bundle-cost", "results.json")))
r40 = json.load(open(os.path.join(HERE, "..", "040-cohesion-vs-knowledge", "results.json")))
ox = json.load(open(os.path.join(HERE, "sources", "openstax_counts.json")))
R = {"study": "055-authoring-productivity", "seed": 20261055, "N_value": N}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + "/" + k)
    else: yield p, x
a_ = dict(leaves(r54["values_main"])); b_ = dict(leaves(json.loads(json.dumps(VAL54, default=float))))
R["check_054_values"] = {"n_leaves": len(a_), "max_abs_diff": max(abs(a_[k] - b_[k]) for k in a_)}
R["check_054_values"]["reproduced"] = R["check_054_values"]["max_abs_diff"] < 1e-12

# ---------- S_LD：印刷した T だけを南半球の図書館に（054 の V_ext の写し） ----------
gv = np.random.default_rng(20261055)
U_lost3 = gv.uniform(size=N); U_find3 = gv.uniform(size=N); U_acc3 = gv.uniform(size=N)
p_accept = gv.uniform(0.5, 1.0, N)
lost_ld_main = gv.uniform(0.3, 0.8, N); lost_ld_multi = gv.uniform(0.15, 0.6, N)
U_lost, U_find, Z, life, H = (G["ns"][k] for k in ("U_lost", "U_find", "Z", "life", "H"))
U_sep, p_sep, t_build, pop_ok, U_t, P_max, kappa, U_acc, p_acc = (G["ns"][k] for k in ("U_sep", "p_sep", "t_build", "pop_ok", "U_t", "P_max", "kappa", "U_acc", "p_acc"))
U_T3, q_B, q_TC, find2, V_ext, HV, LT, T2M, weighted = G["U_T3"], G["q_B"], G["q_TC"], G["find2"], G["V_ext"], G["HV"], G["LT"], G["T2M"], G["weighted"]

def V_ld(k, lost):
    st, T2s, sc = B[k]["st"], B[k]["T2s"], SC[k]
    alive = st != 0
    intact1 = (U_lost > sc["p_lost"]) & (U_find < sc["p_find"])
    T3t = T2s + 943 * sc["M"] * np.exp(0.8 * Z[2]); gen = (T3t < life) & (T3t <= H)
    tab_ok = U_sep > p_sep; cov = np.where(tab_ok, 1.0, 0.0)
    Tt = T2s + t_build
    tgt = tab_ok & pop_ok & (U_t < P_max * cov ** kappa) & (Tt < life) & (Tt <= H)
    frag_acc = (st == 2) & (U_acc < p_acc)
    v1 = np.maximum(q_TC * tgt, gen.astype(float)); v1 = np.where(st == 1, U_T3[k], v1); v1 = np.where(frag_acc, 1.0, v1)
    v2 = q_B * tgt; v2 = np.where(st == 1, 0.0, v2); v2 = np.where(frag_acc, 0.0, v2)   # 紙の T だけ：T3 の社会には足さない
    i3 = (U_lost3 > lost) & (U_acc3 < p_accept) & (U_find3 < find2(k, {}))
    return np.where(alive & intact1, v1, np.where(alive & i3, v2, 0.0))

chk_tc = max(float(np.abs(V_ld(k, np.ones(N)) - V_ext(k, q_TC, True)).max()) for k in COMPS)
R["check_V_ld_equals_TC_when_lost1"] = chk_tc
D_ld = {"S_LD_given_TC": {k: V_ld(k, lost_ld_main) - V_ext(k, q_TC, True) for k in COMPS},
        "S_LD_multi_given_TC": {k: V_ld(k, lost_ld_multi) - V_ext(k, q_TC, True) for k in COMPS}}
VLD = agg(D_ld, anchors=("A_Ord", "A_XPT_S"))
R["values_S_LD"] = VLD

# ---------- 費用（2025年ドル） ----------
CPI = {2013: 233.0, 2023: 304.7, 2025: 321.9}
NC = 200_000
gc = np.random.default_rng(20261055 + 1)
def lu(lo, hi, n=NC): return np.exp(gc.uniform(np.log(lo), np.log(hi), n))
def q(x, ps=(0.05, 0.5, 0.95)): return [float(np.quantile(x, p)) for p in ps]
wage = lu(1e5, 2e5)
BOOKS = {"biology-2e": "biology-2e.collection.xml", "concepts-biology": "concepts-biology.collection.xml",
         "chemistry-2e": "chemistry-2e.collection.xml", "college-physics-2e": "college-physics-2e.collection.xml"}
book_lo = np.array([ox[v]["chars_text_excl_math"] for v in BOOKS.values()], float)
book_hi = np.array([ox[v]["chars_text_incl_math"] for v in BOOKS.values()], float)
def osx_usd_per_char(n=NC):
    i = gc.integers(0, 4, n); ch = gc.uniform(book_lo[i], book_hi[i])
    y13 = gc.uniform(size=n) < 0.5
    cost = np.where(y13, 5e5 * CPI[2025] / CPI[2013], np.exp(gc.uniform(np.log(5e5), np.log(1e6), n)) * CPI[2025] / CPI[2023])
    return cost / ch, ch
def prod_wp(n=NC):
    return 2.5e9 * gc.uniform(5.5, 6.5, n) / (41_018_804 * gc.uniform(1, 2, n) / gc.uniform(1800, 2000, n))
def prod_jau(n=NC):
    return 4.7e6 * gc.uniform(5.5, 6.5, n) / (gc.uniform(6, 7, n) * (1 + gc.uniform(1, 4, n)))
def prod_sme(n=NC):
    return 2391 * gc.uniform(800, 1500, n) * gc.uniform(5.5, 6.5, n) / (gc.uniform(2.7, 4, n) * (1 + gc.uniform(0, 0.5, n)))
OED_KEY = 350e6 / 120; OED_KEYPROOF = 350e6 / 180
def prod_comp(n=NC):
    p = np.where(gc.uniform(size=n) < 0.5, prod_jau(n), prod_sme(n))
    return np.minimum(p, OED_KEY)
upc_osx, ch_osx = osx_usd_per_char()
ANCH = {"OSX_as_prod": wage / upc_osx, "WP": prod_wp(), "JAU": prod_jau(), "SME": prod_sme()}
R["anchors_chars_per_person_year"] = {k: q(v) for k, v in ANCH.items()}
R["anchors_chars_per_person_year"]["OED_keying_only"] = OED_KEY
R["anchors_chars_per_person_year"]["OED_keying_plus_proof"] = OED_KEYPROOF
R["OSX_usd_per_char"] = q(upc_osx)
meds = [float(np.median(v)) for v in ANCH.values()]
R["anchor_median_span"] = max(meds) / min(meds)

def usd_per_char(version, n=NC):
    w = lu(1e5, 2e5, n)
    if version == "V_prior": return w / lu(1e5, 1e7, n)
    if version == "V_orig": return osx_usd_per_char(n)[0]
    if version == "V_wiki": return w / prod_wp(n)
    if version == "V_comp": return w / prod_comp(n)
    if version == "V_mix":
        f = gc.uniform(0.2, 0.8, n); return f * osx_usd_per_char(n)[0] + (1 - f) * w / prod_comp(n)
    raise ValueError(version)
VERS = ["V_prior", "V_orig", "V_wiki", "V_comp", "V_mix"]
eur = gc.uniform(1.0, 1.2, NC)
def cost_T(version, ch_lo=1e7, ch_hi=1e8):
    chars = lu(ch_lo, ch_hi); per_tab = lu(5e4, 5e6); n_tab = np.ceil(chars / per_tab)
    auth = chars * usd_per_char(version)
    return auth + n_tab * 195 * eur * 3, chars, n_tab, auth
# C、S_TC は 054 と同じ式（新しい乱数）
p_reel = lu(8900 / 4, 8900 * 4) * eur; n_C = lu(4, 10)
C_C = n_C * p_reel + lu(0.5, 3) * lu(1e5, 2e5)
def cost_STC(n_tab): return n_tab * gc.uniform(195, 260, NC) * eur + n_C * p_reel + lu(1e4, 3e5) + lu(1e3, 5e4) * gc.uniform(20, 100, NC)
def cost_SLD(chars, copies):
    i = gc.integers(0, 4, NC); per_book = gc.uniform(book_lo[i], book_hi[i])
    books = np.ceil(chars / per_book)
    return books * gc.uniform(25, 50, NC) * copies * gc.uniform(1, 3, NC) + lu(1e3, 1e4) + lu(1e2, 1e3) * copies / 2
# GROW の価値の比（053 と同じ）
def grow_ratio(a, u):
    hi = r40["weighted"][a][u]["GROW"]["total"] / r40["weighted"][a][u]["TAB"]["total"]
    lo_, hi_ = min(0.13, hi), max(0.131, hi)
    return np.exp(gc.uniform(np.log(lo_), np.log(hi_), NC))

KEYS = ["A_Ord/U3", "A_XPT_S/U2"]
def vals(key):
    a, u = key.split("/"); v = dict(VAL54[a][u]); v["S_LD_given_TC"] = VLD[a][u]["S_LD_given_TC"]; v["S_LD_multi_given_TC"] = VLD[a][u]["S_LD_multi_given_TC"]
    return v
def run(version, ch_lo=1e7, ch_hi=1e8):
    C_T, chars, n_tab, auth = cost_T(version, ch_lo, ch_hi)
    C_G = cost_T(version, ch_lo, ch_hi)[0]
    C_TC = C_T + C_C; C_STC = cost_STC(n_tab)
    C_LD = cost_SLD(chars, 2); C_LDm = cost_SLD(chars, 9)
    out = {"costs": {"C_T": q(C_T), "author_share_median": float(np.median(auth / C_T)), "C_TC": q(C_TC), "C_STC": q(C_STC),
                     "C_SLD_AU": q(C_LD), "C_SLD_multi": q(C_LDm), "usd_per_char": q(auth / chars)}, "CE": {}}
    cost = {"TC": C_TC, "S_TC_given_TC": C_STC, "S_LD_given_TC": C_LD, "S_LD_multi_given_TC": C_LDm}
    for key in KEYS:
        v = vals(key); a, u = key.split("/"); out["CE"][key] = {}
        for p in ("total", "heavy", "light"):
            vt = v["TAB"][p]; d = {}
            if vt <= 0: continue
            for x, c in cost.items():
                if v[x][p] <= 0: d[x] = {"value_ratio": v[x][p] / vt, "CE_q": [0.0, 0.0, 0.0]}; continue
                ce = v[x][p] / vt * C_T / c
                d[x] = {"value_ratio": v[x][p] / vt, "CE_q": q(ce)}
            if p == "total":
                ce = grow_ratio(a, u) * C_T / C_G; d["GROW"] = {"CE_q": q(ce), "P_gt_1": float((ce > 1).mean())}
            d["P_SLD_gt_STC"] = float(((v["S_LD_given_TC"][p] / C_LD) > (v["S_TC_given_TC"][p] / C_STC)).mean()) if v["S_LD_given_TC"][p] > 0 else 0.0
            out["CE"][key][p] = d
    return out
R["main_chars"] = {ver: run(ver) for ver in VERS}
R["greq_chars"] = {ver: run(ver, 0.725e6, 8.9e6) for ver in VERS}

# ---------- 仮説 ----------
Hh = {}
m_orig = R["main_chars"]["V_orig"]["costs"]["usd_per_char"][1]; m_prior = R["main_chars"]["V_prior"]["costs"]["usd_per_char"][1]
Hh["H67a"] = {"value": [m_orig, m_prior], "rule": "median $/char V_orig > V_prior", "supported": m_orig > m_prior}
Hh["H67b"] = {"value": R["anchor_median_span"], "rule": "max/min of anchor medians < 10", "supported": R["anchor_median_span"] < 10}
def med(block, key, x, p="total"): return block["CE"][key][p][x]["CE_q"][1]
v = {}
ok = True
for key in KEYS:
    b = R["main_chars"]["V_mix"]; s, t, g = med(b, key, "S_TC_given_TC"), med(b, key, "TC"), med(b, key, "GROW")
    v[key] = [s, t, 1.0, g]; ok &= (s > t > 1.0 > g)
Hh["H67c"] = {"value": v, "rule": "V_mix medians: S_TC|TC > TC > 1 > GROW, both", "supported": bool(ok)}
v = {key: med(R["greq_chars"]["V_mix"], key, "S_TC_given_TC") for key in KEYS}
Hh["H67d"] = {"value": v, "rule": "V_mix, g_req chars: median CE(S_TC|TC) > 1, both", "supported": all(x > 1 for x in v.values())}
b = R["main_chars"]["V_mix"]; v = [med(b, "A_Ord/U3", "S_LD_given_TC"), med(b, "A_Ord/U3", "S_TC_given_TC")]
Hh["H67e"] = {"value": v, "rule": "A_Ord/U3 V_mix: median CE(S_LD|TC) > CE(S_TC|TC)", "supported": v[0] > v[1]}
v = {key: med(b, key, "S_LD_given_TC", "heavy") for key in KEYS}
Hh["H67f"] = {"value": v, "rule": "heavy only V_mix: median CE(S_LD|TC) > 1, both", "supported": all(x > 1 for x in v.values())}
cc, co = R["main_chars"]["V_comp"]["costs"]["C_T"][1], R["main_chars"]["V_orig"]["costs"]["C_T"][1]
Hh["H67g"] = {"value": cc / co, "rule": "median C_T V_comp <= 1/3 V_orig", "supported": cc / co <= 1 / 3}
cm = R["main_chars"]["V_mix"]["costs"]["C_T"][1]
Hh["H67h"] = {"value": cm, "rule": "1e6 <= median C_T (V_mix) <= 3e7", "supported": 1e6 <= cm <= 3e7}
R["hypotheses"] = Hh

# ---------- 図 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
lab = list(ANCH) + ["V_prior", "V_mix(equiv)"]
data = [np.log10(ANCH[k]) for k in ANCH] + [np.log10(lu(1e5, 1e7)), np.log10(wage / usd_per_char("V_mix"))]
ax.boxplot(data, whis=(5, 95), showfliers=False)
ax.set_xticks(range(1, len(lab) + 1)); ax.set_xticklabels(["OpenStax\n(as prod)", "Wikipedia\n(en, 2012)", "Jaucourt\n(compiled)", "Smellie\n(compiled)", "053 prior", "V_mix\n(main)"], fontsize=8)
ax.axhline(np.log10(OED_KEY), color="r", ls="--", lw=0.8, label="OED2 keying only (2.9M/py)")
ax.axhline(np.log10(OED_KEYPROOF), color="m", ls=":", lw=0.8, label="OED2 keying+proof (1.9M/py)")
ax.set_ylabel("log10 characters per person-year"); ax.legend(fontsize=7)
ax.set_title("Public anchors of authoring productivity (box 25-75%, whiskers 5-95%)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_anchors.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
opts = ["TC", "S_TC_given_TC", "S_LD_given_TC", "S_LD_multi_given_TC", "GROW"]
for ax, key in zip(axs, KEYS):
    for j, (blk, mk) in enumerate(((R["main_chars"], "o"), (R["greq_chars"], "s"))):
        for i, ver in enumerate(VERS):
            for k, x in enumerate(opts):
                c = blk[ver]["CE"][key]["total"][x]["CE_q"]
                if c[1] <= 0: continue
                xx = k + (i - 2) * 0.13 + j * 0.05
                ax.errorbar(xx, c[1], yerr=[[c[1] - max(c[0], 1e-6)], [c[2] - c[1]]], fmt=mk, color=f"C{i}", ms=4, lw=0.8,
                            label=(f"{ver} ({'1e7-1e8 chars' if j == 0 else 'g_req chars'})" if k == 0 else None))
    ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(len(opts))); ax.set_xticklabels(["T+C", "S_TC|T+C", "S_LD|T+C\n(AU)", "S_LD|T+C\n(multi)", "GROW"], fontsize=8)
    ax.set_title(f"{key}: CE vs tablet guide (median, 5-95%)", fontsize=9)
axs[0].set_ylabel("CE (TAB=1)"); axs[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ce_by_version.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
b = R["main_chars"]["V_mix"]
for i, key in enumerate(KEYS):
    for k, p in enumerate(("heavy", "light")):
        for m, x in enumerate(("TC", "S_TC_given_TC", "S_LD_given_TC")):
            c = b["CE"][key][p][x]["CE_q"]
            if c[1] <= 0: continue
            ax.errorbar(m + k * 0.2 + i * 0.08, c[1], yerr=[[c[1] - max(c[0], 1e-6)], [c[2] - c[1]]], fmt="o" if i == 0 else "s",
                        color="C3" if p == "heavy" else "C0", ms=4, label=f"{key} {p}" if m == 0 else None)
ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.8); ax.set_xticks([0.15, 1.15, 2.15]); ax.set_xticklabels(["T+C", "S_TC|T+C", "S_LD|T+C (AU)"])
ax.set_ylabel("CE (TAB=1, same part)"); ax.legend(fontsize=7); ax.set_title("V_mix: heavy (red) vs light (blue) collapse", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_heavy_light.png"), dpi=130); plt.close(fig)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"check54": R["check_054_values"], "check_ld": chk_tc, "anchors": R["anchors_chars_per_person_year"], "span": R["anchor_median_span"],
                  "OSX_usd_per_char": R["OSX_usd_per_char"]}, indent=1))
print("values_S_LD", json.dumps(VLD, default=float)[:800])
for blkn in ("main_chars", "greq_chars"):
    for ver in VERS:
        b = R[blkn][ver]; print("==", blkn, ver, "C_T", [round(x) for x in b["costs"]["C_T"]], "STC", [round(x) for x in b["costs"]["C_STC"]], "SLD", [round(x) for x in b["costs"]["C_SLD_AU"]])
        for key in KEYS:
            print("  ", key, {p: {x: round(d[x]["CE_q"][1], 3) for x in d if isinstance(d[x], dict)} for p, d in b["CE"][key].items()},
                  "P_SLD>STC", b["CE"][key]["total"]["P_SLD_gt_STC"])
for k, v in Hh.items(): print(k, v["supported"], json.dumps(v["value"], default=float)[:200])
print("runtime", round(R["runtime_seconds"], 1))
