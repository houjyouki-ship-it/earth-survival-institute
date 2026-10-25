"""研究125：大きな人口損失の後の回復年数と長い後退（open_problems 57・116・266、固定の最優先指示）。
計算のみ、ネット接続なし（資料は sources/ に保存済み、SHA256SUMS）。事前登録：PREREGISTRATION.md（2026-10-25 15:39 JST）。
038 の sim.py を「# ---------- 感度」の前まで exec し、038 の H50e（0.8631415436656069）・BIO/CASC の T2 中央値・
重み付き R_tab を差 1e-9 未満で再現（assert）。その後、038 の人口の増え方 r_pop（logU(0.0014,0.0061)、停滞なし）を
史実の事例の分布で置き換え、重い崩壊の人口待ちの割合・T2 までの年数・重い崩壊分の値 R_tab_from_heavy の比 F_r を出す。
指標は生存確率や復興確率ではない（模型の中の相対値）。
"""
import json, os, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P038 = os.path.abspath(os.path.join(HERE, "..", "038-restart-state-evidence", "sim.py"))
src = open(P038, encoding="utf8").read()
cut = src.index("\n# ---------- 感度")
G = {"__file__": P038, "__name__": "sim038"}
exec(compile(src[:cut], P038, "exec"), G)
R038 = json.load(open(os.path.join(os.path.dirname(P038), "results.json"), encoding="utf8"))
EVID, HEAVY, N = G["EVID"], G["HEAVY"], G["N"]

def pooled_share(out):
    return float(np.median(np.concatenate([out[s]["_share"][out[s]["_heavy_mask"]] for s in EVID])))

BASE, BASE_W = G["BASE"], G["BASE_W"]
chk = {"H50e": (pooled_share(BASE), R038["hypotheses"]["H50e"]["value"])}
for s in ("BIO", "CASC"):
    for k in ("t2_median_heavy", "wait_share_median_heavy", "R_tab_from_heavy"):
        chk[f"{s}.{k}"] = (BASE[s][k], R038["scenarios_038"][s][k])
for a in BASE_W:
    for k in ("R_tab", "R_tab_from_heavy"):
        chk[f"W.{a}.{k}"] = (BASE_W[a][k], R038["weighted_038"][a][k])
maxdiff = max(abs(x - y) for x, y in chk.values())
assert maxdiff < 1e-9, maxdiff
OUT = {"study": "125-pop-recovery-history", "check_038_maxdiff": maxdiff, "check_038": {k: v[0] for k, v in chk.items()}}
print(f"[再現] 038 の H50e={chk['H50e'][0]:.10f}（038: {chk['H50e'][1]:.10f}）、BIO t2={chk['BIO.t2_median_heavy'][0]:.2f}、"
      f"CASC t2={chk['CASC.t2_median_heavy'][0]:.2f}、最大差 {maxdiff:.2e}")

R_POP0 = G["r_pop"].copy()
r_med038 = float(np.median(R_POP0))
LO, HI = 0.0014, 0.0061

# ---------- 1. 事例の表（数値は sources/ の頁から。年は西暦、紀元前は負） ----------
W = "https://en.wikipedia.org/wiki/"
CASES = [
 dict(key="England_BlackDeath", cls="S", P0=4.81e6, Pmin=1.90e6, t_start=1348, t_min=1450, t_rec=1631, status="rec", immig=False,
      note="Broadberry et al. 2010 表（1348 4.81M、1450 1.90M、1600 4.11M、1650 5.31M）。回復年は 1600–1650 の指数補間（推測）",
      url=[W + "Demographics_of_England"]),
 dict(key="Europe_BlackDeath", cls="S", P0=1.0, Pmin=0.5, L_range=(0.45, 0.60), t_start=1347, t_min=None, t_rec=1550, t_rec_range=(1500, 1600),
      status="rec", immig=False, note="死亡 45–50%（Daileader）〜60%（Benedictow）。『16世紀まで 14 世紀の水準に戻らなかった』→回復年は 16 世紀の中点 1550（推測）",
      url=[W + "Black_Death"]),
 dict(key="Greece_LBA", cls="C", P0=600e3, Pmin=300e3, L_range=(0.40, 0.60), t_start=-1200, t_min=-1000, t_rec=-700, t_rec_range=(-800, -700),
      status="rec", immig=False, note="Murray 2017 ch.5：LBA 50–70万、後宮殿期 25–35万（減少 40–60%）、幾何学様式期の終わり（前700頃）までに LBA の水準に回復",
      url=["https://www.cambridge.org/core/books/collapse-of-the-mycenaean-economy/demographic-and-domestic-economic-change-in-early-greece-factors-of-supply-and-demand/9AA296081882FB44BCBC905BA0F51DD3", W + "Greek_Dark_Ages"]),
 dict(key="Britain_postRoman", cls="C", P0=3.0e6, Pmin=1.0e6, L_range=(0.50, 0.75), t_start=400, t_min=600, t_rec=1184, status="rec", immig=False,
      note="4世紀 2–4M、1M へ減った可能性（Anglo-Saxon settlement 頁）。谷の年 600 は推測。回復年は England の 1086 1.71M・1190 3.10M の補間（England と Britain の範囲の違いあり）",
      url=[W + "Anglo-Saxon_settlement_of_Britain", W + "Roman_Britain", W + "Demographics_of_England"]),
 dict(key="Maya_Peten", cls="C", P0=2.87e6, Pmin=6752, t_start=800, t_min=None, t_rec=None, censor=2000, status="cens", immig=False,
      note="Late Classic 80–120 人/km²（Canuto 2018）× Petén 35,854 km²＝2.87–4.30M（面積に一様に当てるのは推測）。Petén の国勢調査 1893 6,752、2002 366,735、2024 640,151 → 2024 でも未回復。規則により 2000 で打ち切り",
      url=["https://www.science.org/doi/10.1126/science.aau0137", "https://es.wikipedia.org/wiki/Departamento_de_Pet%C3%A9n", W + "Pet%C3%A9n_Department"]),
 dict(key="Norse_Greenland", cls="C", P0=2500, Pmin=0, t_start=1350, t_min=None, t_rec=None, censor=2000, status="cens", immig=False,
      note="最盛期 2,000–2,500（改訂推定）、1350–1500 に放棄（西の入植地は 1350–1400、最後の記録 1408）。消滅＝回復しない",
      url=[W + "Norse_settlements_in_Greenland"]),
 dict(key="Hawaii_contact", cls="S", P0=300e3, Pmin=56897, t_start=1778, t_min=1872, t_rec=1924, status="rec", immig=True,
      note="1778 推定 300,000（Schmitt 1968）、1872 56,897、1920 255,912、1930 368,336（補間で 1924）。回復は移民による（先住ハワイ人は 1920 に 24,000 で底）",
      url=[W + "Hawaii", W + "Hawaiian_Kingdom"]),
 dict(key="LatinAmerica_conquest", cls="S", P0=37e6, Pmin=9e6, t_start=1500, t_min=1650, t_rec=1865, t_rec_range=(1810, 1865), status="rec", immig=True,
      note="1492 約 37M、1650 9M（最低の推定で 80% 減）、1800 17M、1850 30M、1900 61M（37M は補間で 1865。本文は『19世紀初め』と書き数値と食い違う）。移民・奴隷の流入あり",
      url=[W + "Population_history_of_the_Indigenous_peoples_of_the_Americas"]),
 dict(key="Easter_Island", cls="C", P0=2500, Pmin=111, t_start=1722, t_min=1877, t_rec=None, t_rec_upper=2017, status="interval", immig=True,
      note="1722 2,000–3,000、1877 111。2017 の国勢調査 7,750（Rapa Nui 3,512）→ 1877〜2017 のどこかで回復。年は取れず（区間）",
      url=[W + "Easter_Island"]),
]
SUPP = [  # L < 0.5（感度の集合のみ）と、主の集合の地域内の重複
 dict(key="ThirtyYearsWar_Germany", cls="S", P0=19e6, Pmin=12e6, t_start=1618, t_min=1648, t_rec=1750, status="rec", immig=False,
      note="1600 18–20M → 1650 11–13M（約40%減）、1750 まで戦前に戻らず", url=[W + "Thirty_Years%27_War"]),
 dict(key="Ireland_famine", cls="S", P0=8.2e6, Pmin=4.5e6, t_start=1845, t_min=1961, t_rec=None, censor=2000, status="cens", immig=False,
      note="1841 8.2M → 1961 4.5M、島全体 2022 で 7M 強 → 未回復", url=[W + "Great_Famine_(Ireland)", W + "Ireland"]),
 dict(key="Cambodia_KhmerRouge", cls="S", P0=7.3e6, Pmin=5.961e6, t_start=1975, t_min=1978, t_rec=1984, status="rec", immig=False,
      note="1975 7.3M、1978 5.961M、1981 6.682M、1991 8.81M（回復は補間で約1984）", url=[W + "Demographics_of_Cambodia", W + "Cambodian_genocide"]),
 dict(key="Copan_valley", cls="C", P0=28000, Pmin=1000, t_start=775, t_min=1200, t_rec=None, censor=1200, status="cens", immig=False,
      note="Copán 谷 最盛 28,000、900 に 15,000、1200 に 1,000 未満（記録の終わり 1200 で打ち切り）。Petén と同じマヤ低地で独立でない",
      url=[W + "Classic_Maya_collapse"]),
]
NOT_OBTAINED = {
 "Egypt_BlackDeath": "Population_history_of_Egypt は遺伝の頁で国全体の人口の数値なし。Black_Death_in_the_Middle_East はカイロ 20万死亡（市の 1/3）と Asyut の納税者 6,000→116 のみで、回復の年の数値なし",
 "Iraq_Mongol1258": "Siege_of_Baghdad は市の死者の推定（80万〜200万、後世の誇張とされる）のみで、地域の人口と回復の年の数値なし",
 "Iceland_1402": "Iceland 頁は 1402–04 の死亡 50–60% と 1494–95 の 30–50% のみ。前後の人口・回復の年の数値なし",
}

def finalize(c):
    c["L"] = 1 - c["Pmin"] / c["P0"]
    c["lnratio"] = math.log(c["P0"] / c["Pmin"]) if c["Pmin"] > 0 else math.inf
    if c["status"] == "rec":
        c["T_rec"] = c["t_rec"] - c["t_start"]; c["event"] = 1
    elif c["status"] == "cens":
        c["T_rec"] = c["censor"] - c["t_start"]; c["event"] = 0
    else:
        c["T_rec"] = None; c["T_upper"] = c["t_rec_upper"] - c["t_start"]; c["T_lower"] = c["t_min"] - c["t_start"]; c["event"] = None
    c["r_eff"] = (c["lnratio"] / (c["t_rec"] - c["t_min"])) if (c["status"] == "rec" and c["t_min"] is not None) else None
    c["r_all"] = (c["lnratio"] / c["T_rec"]) if c["status"] == "rec" else (
        (c["lnratio"] / c["T_rec"] if math.isfinite(c["lnratio"]) else 0.0) if c["status"] == "cens" else None)
    c["decline_years"] = (c["t_min"] - c["t_start"]) if c["t_min"] is not None else None
    # 038 の模型（谷から r で指数成長、停滞なし）が予測する回復年数の中央値と、200 年で戻らない確率（その事例の L で）
    if math.isfinite(c["lnratio"]):
        c["pred038_median_years"] = c["lnratio"] / r_med038
        rc = c["lnratio"] / 200.0
        c["pred038_P_notrec200"] = float(np.clip(math.log(rc / LO) / math.log(HI / LO), 0, 1))
        c["excess_vs038_median"] = (c["T_rec"] - c["pred038_median_years"]) if c["T_rec"] is not None else None
    else:
        c["pred038_median_years"] = math.inf; c["pred038_P_notrec200"] = 1.0; c["excess_vs038_median"] = None
    return c

for c in CASES + SUPP: finalize(c)
MAIN = [c for c in CASES if c["L"] >= 0.5 - 1e-12]

print("\n[事例] key / 類 / L / 回復年数（打ち切り）/ r_eff（谷から）/ r_all（始まりから）/ 038 予測中央値 / 超過")
for c in CASES + SUPP:
    T = c["T_rec"]
    Ts = (f"{T:.0f}" + ("" if c["event"] else "+")) if T is not None else f"({c['T_lower']},{c['T_upper']}]"
    fx = lambda v, f="{:.5f}": "—" if v is None else f.format(v)
    print(f"  {c['key']:24s} {c['cls']} L={c['L']:.3f} T={Ts:>9s} r_eff={fx(c['r_eff'])} r_all={fx(c['r_all'])} "
          f"pred038={c['pred038_median_years']:.0f} excess={fx(c['excess_vs038_median'], '{:+.0f}')}{' 移民あり' if c['immig'] else ''}")

# ---------- 2. 確認的仮説 ----------
def km_median(T, E):
    """Kaplan–Meier の中央値（S(t) ≤ 0.5 になる最小の t）。未到達なら None と、到達した最大の観測年数。"""
    T = np.asarray(T, float); E = np.asarray(E, int); o = np.lexsort((-E, T)); T, E = T[o], E[o]
    S = 1.0; n = len(T)
    for i in range(len(T)):
        if E[i]:
            S *= (n - 1) / n
            if S <= 0.5 + 1e-12: return float(T[i]), S
        n -= 1
    return None, float(T.max())

def h137a(cs):
    obs = [c for c in cs if c["status"] in ("rec", "cens") and (c["event"] == 1 or c["T_rec"] >= 200)]
    k = sum(1 for c in obs if c["T_rec"] > 200)
    return k, len(obs)
k, n = h137a(MAIN)
pa = k / n
binom_p = sum(math.comb(n, j) * 0.616 ** j * 0.384 ** (n - j) for j in range(k, n + 1))
# 頑健性：Europe の回復年の幅 1500–1600（153〜253 年）→ 戻ったと数える側；移民の事例を除く
k2, n2 = h137a([c for c in MAIN if c["key"] != "Europe_BlackDeath"])
k3, n3 = h137a([c for c in MAIN if not c["immig"]])
pred038_mean = float(np.mean([c["pred038_P_notrec200"] for c in MAIN if c["status"] != "interval"]))
OUT["H137a"] = {"k_not_recovered_200": k, "n": n, "share": pa, "threshold_038_halving": 0.616, "supported": bool(n >= 6 and pa > 0.616),
                "binom_p_vs_0616": binom_p, "excl_Europe": [k2, n2, k2 / n2], "excl_immigration": [k3, n3, k3 / n3],
                "038_predicted_share_for_each_case_own_L_mean": pred038_mean,
                "excluded": "Easter_Island（回復年が区間で 200 年の判定不能）"}
print(f"\n[H137a] L≥0.5 で 200 年以内に戻らなかった割合 {k}/{n} = {pa:.3f}（038 の半減の暗の値 0.616）→ "
      f"{'支持' if OUT['H137a']['supported'] else '不支持'}。二項 p={binom_p:.3f}。Europe 除く {k2}/{n2}、移民除く {k3}/{n3}。"
      f"ただし各事例の L で 038 が予測する割合の平均は {pred038_mean:.3f}")

def km_for(cs, easter_mode="upper"):
    T, E = [], []
    for c in cs:
        if c["status"] == "interval":
            if easter_mode == "drop": continue
            T.append(c["T_upper"]); E.append(1)   # 上限で戻ったとみなす＝C を短く見る保守側
        else:
            T.append(c["T_rec"]); E.append(c["event"])
    return km_median(T, E)
res_b = {}
for lab, filt in (("main", lambda c: True), ("excl_immigration", lambda c: not c["immig"])):
    Cm = km_for([c for c in MAIN if c["cls"] == "C" and filt(c)]); Sm = km_for([c for c in MAIN if c["cls"] == "S" and filt(c)])
    ratio = (Cm[0] / Sm[0]) if (Cm[0] and Sm[0]) else ((Cm[1] / Sm[0]) if Sm[0] else None)
    res_b[lab] = {"C_median": Cm[0], "C_max_obs_if_unreached": None if Cm[0] else Cm[1], "S_median": Sm[0], "ratio_C_over_S": ratio}
# 損失の大きさで割った年数（ln(P0/Pmin) あたり）での比較（入力判明後の記述）
per_log = {c["key"]: (c["T_rec"] / c["lnratio"]) if (c["T_rec"] is not None and math.isfinite(c["lnratio"])) else None for c in MAIN}
OUT["H137b"] = {**res_b, "supported": bool(res_b["main"]["ratio_C_over_S"] is not None and res_b["main"]["ratio_C_over_S"] >= 2),
                "years_per_unit_lnloss_descriptive": per_log}
print(f"[H137b] KM 中央値 C={res_b['main']['C_median']} 年、S={res_b['main']['S_median']} 年、比 {res_b['main']['ratio_C_over_S']:.2f} → "
      f"{'支持' if OUT['H137b']['supported'] else '不支持'}（移民除く：比 {res_b['excl_immigration']['ratio_C_over_S']:.2f}）")

reff = [c["r_eff"] for c in MAIN if c["r_eff"] is not None]
reff_ni = [c["r_eff"] for c in MAIN if c["r_eff"] is not None and not c["immig"]]
med_c = float(np.median(reff))
OUT["H137c"] = {"r_eff": {c["key"]: c["r_eff"] for c in MAIN if c["r_eff"] is not None}, "median": med_c, "median_excl_immigration": float(np.median(reff_ni)),
                "range_038": [LO, HI], "supported": bool(LO <= med_c <= HI),
                "decline_years_before_trough": {c["key"]: c["decline_years"] for c in MAIN if c["decline_years"] is not None}}
print(f"[H137c] 回復した事例の r_eff 中央値 {med_c:.5f}（移民除く {np.median(reff_ni):.5f}）、038 の範囲 [{LO}, {HI}] → "
      f"{'支持' if OUT['H137c']['supported'] else '不支持'}。谷までの下り坂の年数 {OUT['H137c']['decline_years_before_trough']}")

# ---------- 3. 038 の模型に入れ直す ----------
tsrc = src[src.index("def times("):src.index("\nEVID = ")]
tsrc = tsrc.replace("np.maximum(0.0, np.log(k * Nk / n) / r))", "np.where(k * Nk > n, STAG + np.log(k * Nk / n) / r, 0.0))")
assert "STAG" in tsrc
G["STAG"] = np.zeros(N)
exec(compile(tsrc, P038 + "<times+STAG>", "exec"), G)
g5 = np.random.default_rng(20261125)
U_case = g5.uniform(size=N); U_stag = g5.uniform(size=N); U_pick = g5.uniform(size=N)

def from_list(vals):
    vals = np.asarray(vals, float); return vals[np.minimum((U_case * len(vals)).astype(int), len(vals) - 1)]

def run_variant(r_arr=None, stag=None):
    G["r_pop"] = R_POP0 if r_arr is None else np.maximum(r_arr, 1e-9)
    G["STAG"] = np.zeros(N) if stag is None else stag
    out, wt = G["run_all"]()
    G["r_pop"] = R_POP0; G["STAG"] = np.zeros(N)
    o = {"pooled_wait_share_median_heavy": pooled_share(out)}
    for s in ("BIO", "CASC"):
        o[s] = {k: out[s][k] for k in ("wait_share_median_heavy", "t2_median_heavy", "stage_median_heavy", "R_tab_from_heavy")}
        o[s]["t2_ratio_vs_038"] = out[s]["t2_median_heavy"] / BASE[s]["t2_median_heavy"]
        o[s]["F_r"] = out[s]["R_tab_from_heavy"] / BASE[s]["R_tab_from_heavy"]
    o["weighted"] = {a: {"R_tab": wt[a]["R_tab"], "R_tab_from_heavy": wt[a]["R_tab_from_heavy"],
                         "F_r": wt[a]["R_tab_from_heavy"] / BASE_W[a]["R_tab_from_heavy"]} for a in wt}
    return o

V = {}
V["V0_038_check"] = run_variant()
assert abs(V["V0_038_check"]["pooled_wait_share_median_heavy"] - chk["H50e"][1]) < 1e-9
# V1（登録どおり）：r を回復した事例の r_eff の経験分布で（打ち切りの事例は谷の年が不明で r_eff を作れず＝楽観側）
V["V1a_reff_all"] = run_variant(from_list(reff))
V["V1b_reff_no_immig"] = run_variant(from_list(reff_ni))
# V2（登録どおりの形、細部は入力判明後に固定）：038 の r に、確率 p_nr で停滞年数を前置き。
# 停滞年数＝事例の回復年数 − 038 の r の中央値での予測年数（超過）が正の事例の値。消滅・打ち切りは観測年数を下限として使う。
pos = []; neg = 0; n_det = 0
for c in MAIN:
    if c["key"] == "Norse_Greenland":
        pos.append(c["T_rec"]); n_det += 1; continue      # 消滅：超過は無限、下限として 650 年
    if c["status"] == "interval":
        ex_hi = c["T_upper"] - c["pred038_median_years"]
        if ex_hi < 0: neg += 1; n_det += 1
        continue
    ex = c["excess_vs038_median"]
    if c["status"] == "cens" and ex < 0: continue          # 打ち切りで超過が負なら判定不能（Petén）
    n_det += 1
    if ex > 0: pos.append(ex)
    else: neg += 1
p_nr = len(pos) / n_det
pos_ni = [c["T_rec"] if c["key"] == "Norse_Greenland" else c["excess_vs038_median"] for c in MAIN
          if not c["immig"] and c["status"] != "interval" and (c["key"] == "Norse_Greenland" or (c["excess_vs038_median"] is not None and c["excess_vs038_median"] > 0))]
n_det_ni = sum(1 for c in MAIN if not c["immig"] and c["status"] != "interval" and not (c["status"] == "cens" and c["key"] != "Norse_Greenland"))
p_nr_ni = len(pos_ni) / n_det_ni
def stag_draw(p, vals):
    vals = np.asarray(vals, float); pick = vals[np.minimum((U_pick * len(vals)).astype(int), len(vals) - 1)]
    return np.where(U_stag < p, pick, 0.0)
V["V2a_stag_all"] = run_variant(None, stag_draw(p_nr, pos))
V["V2b_stag_no_immig"] = run_variant(None, stag_draw(p_nr_ni, pos_ni))
OUT["V2_inputs"] = {"p_nr": p_nr, "stag_years": pos, "n_determinable": n_det, "p_nr_no_immig": p_nr_ni, "stag_years_no_immig": pos_ni, "n_det_no_immig": n_det_ni}
# V3（入力判明後）：始まりからの実効 r_all（停滞・下り坂を含む）。消滅は r≈0（実質「戻らない」）、打ち切りは上限
rall = [c["r_all"] for c in MAIN if c["r_all"] is not None]
rall_ni = [c["r_all"] for c in MAIN if c["r_all"] is not None and not c["immig"]]
V["V3a_rall_all"] = run_variant(from_list(rall))
V["V3b_rall_no_immig"] = run_variant(from_list(rall_ni))
# 感度：038 の r の下限だけ史実の最遅（Britain・Greece の r_all≈0.0014）に合わせ、上限を r_eff の最速（移民なし England 0.0051）に
V["V4_logU_0.0014_0.0051"] = run_variant(np.exp(np.log(0.0014) + U_case * (np.log(0.0051) - np.log(0.0014))))
OUT["variants"] = V
OUT["inputs"] = {"r_eff_list": reff, "r_eff_list_no_immig": reff_ni, "r_all_list": rall, "r_all_list_no_immig": rall_ni, "r_median_038": r_med038}

print("\n[入れ直し] 変種 / 重い崩壊の人口待ちの割合の中央値 / BIO・CASC の T2 中央値（038 比）/ F_r（BIO, CASC, A_Ord, A_XPT_S, A_XPT_E）")
for k_, o in V.items():
    w = o["weighted"]
    print(f"  {k_:24s} share={o['pooled_wait_share_median_heavy']:.4f}  BIO t2={o['BIO']['t2_median_heavy']:.1f}（×{o['BIO']['t2_ratio_vs_038']:.2f}）"
          f" CASC t2={o['CASC']['t2_median_heavy']:.1f}（×{o['CASC']['t2_ratio_vs_038']:.2f}）  F_r BIO={o['BIO']['F_r']:.3f} CASC={o['CASC']['F_r']:.3f}"
          f"  W: {w['A_Ord']['F_r']:.3f} {w['A_XPT_S']['F_r']:.3f} {w['A_XPT_E']['F_r']:.3f}")

v2 = V["V2a_stag_all"]
OUT["H137d"] = {"V2_share": v2["pooled_wait_share_median_heavy"], "V2_BIO_t2_ratio": v2["BIO"]["t2_ratio_vs_038"], "V2_CASC_t2_ratio": v2["CASC"]["t2_ratio_vs_038"],
                "V1_share": V["V1a_reff_all"]["pooled_wait_share_median_heavy"],
                "supported": bool(v2["pooled_wait_share_median_heavy"] > 0.8631415436656069 and v2["BIO"]["t2_ratio_vs_038"] >= 1.2 and v2["CASC"]["t2_ratio_vs_038"] >= 1.2)}
print(f"[H137d] V2：割合 {v2['pooled_wait_share_median_heavy']:.4f}（>0.8631?）、T2 比 BIO {v2['BIO']['t2_ratio_vs_038']:.2f}・CASC {v2['CASC']['t2_ratio_vs_038']:.2f}（≥1.2?）→ "
      f"{'支持' if OUT['H137d']['supported'] else '不支持'}")
OUT["H137e"] = {"reproduced": True, "maxdiff": maxdiff, "counted_as_support": False}

# ---------- 4. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
allc = MAIN + [c for c in SUPP]
ys = np.arange(len(allc))
for i, c in enumerate(allc):
    col = "tab:red" if c["cls"] == "C" else "tab:blue"
    if c["status"] == "interval":
        ax[0].plot([c["T_lower"], c["T_upper"]], [i, i], color=col, lw=6, alpha=0.4)
    else:
        ax[0].barh(i, c["T_rec"], color=col, alpha=0.8 if c["L"] >= 0.5 else 0.35, hatch=None if c["event"] else "//")
    if math.isfinite(c["pred038_median_years"]):
        ax[0].plot(c["pred038_median_years"], i, "k|", ms=14)
ax[0].axvline(200, color="gray", ls=":")
ax[0].set_yticks(ys); ax[0].set_yticklabels([f"{c['key']} (L={c['L']:.2f})" for c in allc], fontsize=7)
ax[0].set_xscale("log"); ax[0].set_xlabel("years from loss start to regain P0 (hatched = not recovered / censored)")
ax[0].set_title("Historical recovery (red C: state+trade collapsed, blue S)\nblack tick = 038 median prediction from own L", fontsize=9)
names = list(V.keys()); xs = np.arange(len(names))
ax[1].bar(xs - 0.2, [V[k]["BIO"]["t2_ratio_vs_038"] for k in names], 0.4, label="BIO T2 median / 038")
ax[1].bar(xs + 0.2, [V[k]["CASC"]["t2_ratio_vs_038"] for k in names], 0.4, label="CASC T2 median / 038")
ax2 = ax[1].twinx(); ax2.plot(xs, [V[k]["pooled_wait_share_median_heavy"] for k in names], "ko-", label="pooled wait share (right)")
ax2.set_ylim(0.5, 1.0); ax[1].axhline(1.2, color="gray", ls=":")
ax[1].set_xticks(xs); ax[1].set_xticklabels(names, rotation=35, ha="right", fontsize=7); ax[1].set_yscale("log")
ax[1].set_title("Replacing 038 r_pop with historical distributions (model-internal)", fontsize=9)
ax[1].legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_recovery.png"), dpi=130)

OUT["cases"] = [{k: v for k, v in c.items()} for c in CASES]; OUT["cases_supplementary"] = SUPP; OUT["not_obtained"] = NOT_OBTAINED
OUT["seed"] = 20261125; OUT["runtime_seconds"] = time.time() - t0
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x): return str(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    return x
json.dump(clean(OUT), open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(f"\n実行時間 {OUT['runtime_seconds']:.1f} 秒。results.json と fig_recovery.png を書いた。")
