"""研究123：手作りの継電器の乾いた接点の寿命の分布（open_problems 289）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 11:35 JST）。
122 の sim.py を「# ---------- 仮説」の前まで exec し、122 の p_relay・p_vac を差 0 で再現（assert）。
事実（sources/）：点火の接点（タングステン合金、コンデンサで弧を抑える、負荷を切る）は 6,000 mile か 10,000 km で交換（howacarworks）、
10,000〜12,000 mile で交換（Moss Motors）、2,500 rpm の 4 気筒で毎分 5,000 回・6,000 mile で 3,000 万回（Powerspark の計算）。
やすりで面を戻すのは一時しのぎ（howacarworks）。データシートの機械的寿命 10^7・定格負荷 10^5（120 の sources）。
鋼の疲労限度は理論上あるが、高サイクルでは金属に疲労限度はないとの研究もある（Fatigue (material)）。
寿命の分布の形・手作りの倍率・磨き直しの倍率は【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P122 = os.path.abspath(os.path.join(HERE, "..", "122-min-ops-reader", "sim.py"))
src = open(P122, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P122, "__name__": "sim122"}
exec(compile(src[:cut], P122, "exec"), G)
R122 = json.load(open(os.path.join(os.path.dirname(P122), "results.json"), encoding="utf8"))
dR = max(abs(G["PR"][k]["p_relay"] - R122["p_relay_versions"][k]["p_relay"]) for k in R122["p_relay_versions"])
dV = max(abs(G["PV"][k] - R122["p_vac_versions"][k]) for k in R122["p_vac_versions"])
assert dR < 1e-12 and dV < 1e-12, (dR, dV)
prelay, from_design, DES, P_TUBE = G["prelay"], G["from_design"], G["DES"], G["P_TUBE"]
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC"))
OUT = {"study": "123-dry-contact-life", "check_122_prelay_maxdiff": dR, "check_122_pvac_maxdiff": dV}

# ---------- 1. 点火の接点から工場の品の寿命の錨 ----------
def ign_ops(miles, mph, rpm=2500, sparks_per_rev=2):
    return rpm * sparks_per_rev * 60 * miles / mph
IGN = {}
for miles, src_ in ((6000, "howacarworks/Powerspark"), (10000 / 1.609344, "howacarworks 10,000 km"), (12000, "Moss upper")):
    for mph in (30, 45, 60):
        IGN[f"{miles:.0f}mi|{mph}mph"] = {"source": src_, "ops": ign_ops(miles, mph)}
ign = np.array([v["ops"] for v in IGN.values()])
OUT["ignition_ops"] = IGN
OUT["ignition_ops_range"] = [float(ign.min()), float(ign.max())]
OUT["check_powerspark_30M"] = ign_ops(6000, 60)            # Powerspark の『3,000 万回』は 60 mph 相当
# 工場の品の乾いた接点の錨：データシートの機械 1e7 と点火の接点（負荷を切るのに）3e7〜1.2e8 の間の幾何平均【推測】
anchor_lo, anchor_hi = 1e7, float(ign.max())
ANCH = 10 ** ((np.log10(anchor_lo) + np.log10(anchor_hi)) / 2)
OUT["factory_anchor"] = {"lo": anchor_lo, "hi": anchor_hi, "geo_mid": ANCH}

# ---------- 2. 手作りの寿命の分布（log10 L ~ 正規）で p_relay を積分 ----------
GH = np.polynomial.hermite_e.hermegauss(41)                 # 標準正規のガウス・エルミート
def expect(design, mu, sig, **kw):
    xs, ws = GH; ws = ws / ws.sum(); vals = []
    for x in xs:
        L = 10 ** (mu + sig * x)
        vals.append(prelay(from_design(design, L_arm=L, **kw))["p_relay"])
    vals = np.array(vals)
    cdf = np.cumsum(ws)
    q = lambda p: float(vals[np.searchsorted(cdf, p)])
    return {"mean": float((ws * vals).sum()), "p10": q(0.10), "p50": q(0.50), "p90": q(0.90)}
# 主：工場の錨 ×手作りの倍率 h（中央 0.1、log10 の幅 0.5）、錨の幅 0.3【推測】
def dist(h_med=0.1, h_sd=0.5, anch=ANCH, a_sd=0.3):
    return float(np.log10(anch * h_med)), float(np.hypot(h_sd, a_sd))
DV = {"main": dist(), "h0.03": dist(h_med=0.03), "h0.3": dist(h_med=0.3), "wide": dist(h_sd=1.0),
      "silver_only_anchor1e7": dist(anch=1e7), "pess": dist(h_med=0.03, h_sd=1.0, anch=1e7)}
OUT["dist_versions"] = {k: {"mu_log10": m, "sigma": s, "L_median": 10 ** m} for k, (m, s) in DV.items()}
DESN = {"D1": "D1_min|rand", "D4": "D4_stepper|rand", "D3": "D3_realistic|rand"}
EXP = {}
for dn, dk in DESN.items():
    for vn, (m, s) in DV.items():
        for rf in (1.0, 3.0, 10.0):
            EXP[f"{dn}|{vn}|refurb{rf:g}"] = expect(dk, m, s, refurb=rf)
OUT["expected_prelay"] = EXP
# 点の値（比較用）：122 の L 1e6・1e7
PT = {f"{dn}|L{L:.0e}|refurb{rf:g}": prelay(from_design(dk, L_arm=L, refurb=rf))["p_relay"]
      for dn, dk in DESN.items() for L in (1e5, 1e6, 1e7) for rf in (1.0, 10.0)}
OUT["point_prelay"] = PT
# 走査：L と p_relay（D1・D3）
Ls = np.logspace(4, 9, 41)
SC = {dn: [prelay(from_design(dk, L_arm=L))["p_relay"] for L in Ls] for dn, dk in DESN.items()}
OUT["scan_L"] = list(map(float, Ls)); OUT["scan_prelay"] = SC

# ---------- 3. p_vac と F_heavy ----------
PVN = {}
for key in ("D3|main|refurb1", "D1|main|refurb1", "D4|main|refurb1", "D3|pess|refurb1", "D3|main|refurb10"):
    PVN[key] = 1 - (1 - EXP[key]["mean"]) * (1 - P_TUBE)
OUT["p_vac"] = PVN; OUT["p_tube"] = P_TUBE
TAB = {}
for vn in ("D3|main|refurb1", "D1|main|refurb1", "D3|pess|refurb1"):
    fT = t2e_two(100.0, L_bulk, PPREC * PVN[vn])
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB

# ---------- 仮説 ----------
H = {}
H["H135a"] = {"class": "confirmatory", "supported": False, "value": "Contact breaker の記事に交換・点検の間隔の数値なし（『regular adjustment』『subject to wear』のみ）。間隔は howacarworks・Moss Motors で得た"}
H["H135b"] = {"class": "confirmatory", "supported": False, "value": "Contact breaker の記事に材料の記述なし（タングステンは Powerspark の頁にあり）"}
H["H135c"] = {"class": "confirmatory", "supported": True, "value": "『some steel and titanium alloys exhibit a theoretical fatigue limit』。ただし『fatigue limits do not exist for any metals』とする研究もあると併記"}
H["H135d"] = {"class": "confirmatory", "supported": True, "strength": "weak", "value": "Electrical contact：継電器を含む接点の典型の材料に Platinum-group metals、Platinum：用途に electrical contacts。電信の接点という記述はどれにもない"}
lo, hi = OUT["ignition_ops_range"]
H["H135e"] = {"class": "prediction", "value": [lo, hi], "rule": "2e7<=ops<=1e9", "supported": bool(2e7 <= lo and hi <= 1e9),
  "note": "登録は Contact breaker の記事の間隔の予定だったが記事に値がなく、別の頁の値で計算（逸脱）"}
d1 = EXP["D1|main|refurb1"]["mean"]; d3 = EXP["D3|main|refurb1"]["mean"]
H["H135f"] = {"class": "prediction", "value": {"D1": d1, "D3": d3}, "rule": "0.60<=D1<=0.80 かつ 0.42<=D3<=0.62",
  "supported": bool(0.60 <= d1 <= 0.80 and 0.42 <= d3 <= 0.62)}
dref = PT["D3|L1e+06|refurb10"] - PT["D3|L1e+06|refurb1"]; dL = PT["D3|L1e+07|refurb1"] - PT["D3|L1e+06|refurb1"]
H["H135g"] = {"class": "prediction", "value": {"refurb_1to10": dref, "L_1e6to1e7": dL}, "rule": "refurb の差 > L の差", "supported": bool(dref > dL),
  "note": "120 の式では refurb は交換の数を割るだけで L を同じ倍率で上げるのとほぼ同じ働き（初めの一揃いの分だけ違う）"}
H["H135h"] = {"class": "descriptive", "supported": None, "value": "分布の版と p_relay の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for dn, c in (("D1", "g"), ("D4", "m"), ("D3", "b")):
    axs[0].plot(Ls, SC[dn], c, label=f"{dn} p_relay")
m, s = DV["main"]; x = np.linspace(4, 9, 300); pdf = np.exp(-0.5 * ((x - m) / s) ** 2)
ax2 = axs[0].twinx(); ax2.fill_between(10 ** x, pdf, color="orange", alpha=0.25); ax2.set_yticks([]); ax2.set_ylim(0, 3)
axs[0].axvspan(lo, hi, color="red", alpha=0.1); axs[0].text(lo, 0.40, "ignition points\n(factory, load-breaking)", fontsize=7, color="r")
axs[0].axvline(1e7, color="k", ls=":", lw=0.8); axs[0].text(1.1e7, 0.9, "datasheet mech 1e7", fontsize=7)
axs[0].set_xscale("log"); axs[0].set_xlabel("dry-contact life L_arm (operations)"); axs[0].set_ylabel("p_relay"); axs[0].legend(fontsize=7, loc="upper left")
axs[0].set_title("123: handmade dry-contact life (orange = main distribution)", fontsize=9)
vns = list(DV.keys()); xx = np.arange(len(vns))
for j, dn in enumerate(("D1", "D4", "D3")):
    mean = [EXP[f"{dn}|{v}|refurb1"]["mean"] for v in vns]
    err = [[mean[i] - EXP[f"{dn}|{v}|refurb1"]["p10"] for i, v in enumerate(vns)], [EXP[f"{dn}|{v}|refurb1"]["p90"] - mean[i] for i, v in enumerate(vns)]]
    axs[1].bar(xx + (j - 1) * 0.27, mean, 0.27, yerr=np.clip(err, 0, None), capsize=2, label=dn)
axs[1].set_xticks(xx); axs[1].set_xticklabels(vns, fontsize=7, rotation=20); axs[1].set_ylabel("E[p_relay] (bars: p10-p90 over L)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_dry_contact.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_122", dR, dV)
print("ignition", OUT["ignition_ops_range"], "powerspark60", OUT["check_powerspark_30M"], "anchor", ANCH)
for k, v in OUT["dist_versions"].items(): print("dist", k, v)
for k, v in EXP.items(): print("E", k, {kk: round(vv, 4) for kk, vv in v.items()})
for k, v in PT.items(): print("pt", k, round(v, 4))
for k, v in PVN.items(): print("p_vac", k, round(v, 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:200])
print("runtime", round(OUT["runtime_s"], 1))
