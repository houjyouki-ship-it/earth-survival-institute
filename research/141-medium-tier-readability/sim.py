"""研究141：媒体と書き方ごとの「読むのに要る最低の技術段 × 劣化 × 段に届く時刻」（next_plan 固定の節「記録の単純さ」、open_problems 324）
注：P_read・量×読める確率は模型の中の量で、生存確率でも復興確率でもない。
- 段に届く時刻：038 の sim.py を一時の写し（/tmp/rep141）で実行し、重い崩壊（T1・T0・T00）の標本の t1（レンズ）・t2 を取り出す。
  GEN・AI は 032 の時刻（d01・d12、識字喪失の倍率つき）、BIO・CASC は 038 の人口待ちつきの時刻。重みは 038 の mean_weights × 重い崩壊の割合。
- 媒体の寿命：Weibull(k=2)、中央値＝主張の寿命×保管の割引 f（logU 0.3〜1、012 と同じ）。
- 発見の確率と識字は全媒体に共通なので比に効かない（ここでは 1 とする）。
"""
import json, runpy, time, sys, platform, hashlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
T0 = time.time()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
ns = runpy.run_path("/tmp/rep141/038-restart-state-evidence/sim.py", run_name="m038")
# 再現の確認：038 の公開済み results.json の重い崩壊の割合と差 0
r38 = json.load(open(f"{HERE}/../038-restart-state-evidence/results.json"))
for s in ("BIO", "CASC"):
    assert abs(ns["BASE"][s]["heavy_share"] - r38["scenarios_038"][s]["heavy_share"]) == 0.0, s
    assert abs(ns["BASE"][s]["t2_median_heavy"] - r38["scenarios_038"][s]["t2_median_heavy"]) == 0.0, s
N = ns["N"]; H = 1000.0
d01, d12, U_lit, lit_mult, t_build = ns["d01"], ns["d12"], ns["U_lit"], ns["lit_mult"], ns["t_build"]
SC = ns["SCEN"]

def tier_times(s):
    """重い崩壊の標本について (t1, t2) を返す。"""
    lit = SC[s]["lit"]
    if s in ("BIO", "CASC"):
        sv = ns["survivors"](s); st = ns["assign"](sv)
        n = np.maximum(sv["nmax"], 1.0); r = ns["r_pop"]
        wait = lambda Nk: np.maximum(0.0, np.log(Nk / n) / r)
        d01e = np.where(U_lit < lit, d01 * lit_mult, d01); dFe = ns["d_Fe"]
        tFe = np.where(st == 6, np.maximum(dFe, wait(ns["N_Fe"])), 0.0)
        t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(ns["N_lens"])), 0.0)
        t2 = np.where(st >= 4, np.maximum(t1 + d12, wait(ns["N_T2e"])), 0.0)
        heavy = np.isin(st, (4, 5, 6))
        # 038 の times() と t2 が一致することを確認
        _, t2_ref, _ = ns["times"](st, sv, lit)
        assert np.array_equal(t2, t2_ref)
    else:
        st = ns["DRAWS"][s]["st"]
        d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
        t1 = np.where(st == 5, d01e, 0.0); t2 = t1 + d12
        heavy = np.isin(st, (4, 5))
    return t1[heavy], t2[heavy], heavy.mean()

HSC = ["GEN", "AI", "BIO", "CASC"]
TT = {s: tier_times(s) for s in HSC}
g = np.random.default_rng(20261141)

# ---- 媒体の表（単位、段、寿命の主張の範囲、単位あたりの字数）----
# 字数：1 頁 3,000 字【推測】。1 字 = 1 バイトとして機械の容量と並べる【推測：英文の UTF-8】。
PAGE = 3000
MEDIA = [
 # key, 表示名, 段, 寿命 lo, hi, 単位, 単位の字数, 根拠
 ("stone",   "石の刻字（1 m² の面、肉眼）", "R0", 2500, 2e4, "1 m²", 1.25e5, "Behistun は前 522〜486 年の刻字で現存（下限）。字の密さは MOM の板の 1/10【推測】"),
 ("mom_eye", "MOM のセラミック板（肉眼）", "R0", 1e4, 1e6, "板 1 枚 20×20 cm", 5e4, "MOM：文字だけで最大 5 万字・100 万年の主張（012・053 で確認）。下端は【推測】"),
 ("paper",   "アルカリ紙の本（肉眼）", "R0", 500, 1000, "本 1 冊 500 頁", 500 * PAGE, "ASTM D3290：最良 1,000 年超・平均 500 年（WP Acid-free paper）"),
 ("mom_mf",  "MOM の極小文字セラミック（10 倍レンズ）", "R1", 1e4, 1e6, "板 1 枚", 5e6, "MOM：最大 500 万字（012 で確認）"),
 ("fiche24", "銀塩マイクロフィッシュ 24 倍（強い拡大鏡）", "R1", 250, 500, "A6 1 枚", 98 * PAGE, "98 コマ（NMA Type I）、強い拡大鏡で読める、寿命 約 500 年（WP Microform、Saffady 2000）"),
 ("fiche48", "COM マイクロフィッシュ 48 倍（顕微鏡）", "R2", 250, 500, "4×6 in 1 枚", 240 * PAGE, "240 頁（WP Microform）。48 倍は顕微鏡の段と置く【推測】"),
 ("ultra",   "ウルトラフィッシュ 150 倍（顕微鏡）", "R2", 250, 500, "1 枚", 2380 * PAGE, "BNB 150 倍で 2,380 コマ（WP Microform）"),
 ("rosetta", "Rosetta ディスク（ニッケル、顕微鏡）", "R2", 2000, 1e4, "直径 3 in 1 枚", 13000 * PAGE, "13,000 頁超・650 倍（012・019 で確認）。寿命の幅は【推測】"),
 ("piql_hand","Piql の手読み QR 層（リールの 1 %）", "R2", 500, 2000, "リールの 1 %", 1.9e6, "021 の D_max 1.9 MB（Leeuwenhoek 級、1e-6）、寿命 500〜2,000 年（012）"),
 ("piql",    "Piql の機械読みフィルム（1 リール）", "R2e", 500, 2000, "1 リール", 113e9, "約 113 GB/リール（012）。R2e の自作読取装置では q(M) の割合だけ、R3 で全量"),
 ("silica",  "Project Silica（石英ガラス）", "R3", 1e4, 1e5, "板 1 枚", 7e12, "7 TB 超・数万〜数十万年（012）"),
 ("mdisc",   "M-DISC BDXL", "R3d", 100, 500, "1 枚", 100e9, "NIST IR 8387 で 100 年超、Verbatim の他地域は『数百年』（WP M-DISC）"),
 ("lto",     "LTO テープ", "R3d", 10, 30, "1 巻", 30e12, "空調下で最大 30 年（WP Optical disc の LTO の節、要出典の記載）"),
 ("ssd",     "SSD（無通電）", "R3d", 1, 10, "1 台 1 TB", 1e12, "リフレッシュなしで室温 1 年程度（140 で確認の WP Flash memory）。上端は【推測】"),
]

def logu(lo, hi, n): return np.exp(g.uniform(np.log(lo), np.log(hi), n))

def run(var):
    """var：m23 中央値、R3d の追加、廃墟のレンズの確率、停滞の版。戻り値は媒体ごとの量。"""
    out = {k[0]: {"P_read_H": 0.0, "P_K01": 0.0, "P_K12": 0.0, "P_K2e": 0.0} for k in MEDIA}
    tiers_med = {}
    wsum = 0.0
    for s in HSC:
        t1, t2, hs = TT[s]
        n = len(t1)
        if n == 0: continue
        w = ns["WT"][var.get("anchor", "A_XPT_S")]["mean_weights"][s] * hs
        wsum += w
        rs = np.random.default_rng(20261141 + HSC.index(s))
        L = lambda lo, hi: np.exp(rs.uniform(np.log(lo), np.log(hi), n))
        stag = np.where(rs.uniform(size=n) < var.get("p_stag", 0.0), L(*var.get("stag", (300, 900))), 0.0)
        t1s, t2s = t1 + stag, t2 + stag
        tb = t_build[:n]
        m23 = var.get("m23", 300.0) * np.exp(0.8 * rs.standard_normal(n))
        salv = rs.uniform(size=n) < var.get("p_salv", 0.0)       # 廃墟の 10 倍級の拡大鏡が早く使える
        t_salv = L(1, 20); life_salv = np.where(rs.uniform(size=n) < 0.5, L(500, 5000), L(30, 300))  # 104 のガラス／樹脂の寿命の幅
        R1 = np.where(salv, np.minimum(t_salv, t1s), t1s)
        T = {"R0": np.zeros(n), "R1": R1, "R2": t2s, "R2e": t2s + tb, "R3": t2s + m23}
        T["R3d"] = T["R3"] + (L(*var["drive"]) if var.get("drive") else 0.0)
        qM = L(0.03, 0.43)   # 033・025 の q(M)：自作の小さい RAM で読める高層の割合
        for (k, nm, tier, lo, hi, unit, cap, src) in MEDIA:
            life = L(lo, hi) * L(0.3, 1.0) / np.log(2) ** 0.5 * (-np.log(rs.uniform(size=n))) ** 0.5
            ta = T[tier]
            ok = (ta <= H) & (life > ta)
            frac = np.ones(n)
            if k == "piql":   # R2e では q(M) だけ、R3 に届けば全量
                okR3 = (T["R3"] <= H) & (life > T["R3"])
                frac = np.where(okR3, 1.0, np.where(ok, qM, 0.0)); ok = ok | okR3
            o = out[k]
            o["P_read_H"] += w * np.mean(ok)
            o["E_frac_H"] = o.get("E_frac_H", 0.0) + w * np.mean(ok * frac)
            # 中身が役に立つ時刻までに読めるか（循環の検査）
            o["P_K01"] += w * np.mean((ta < t1s) & (life > ta))                    # 製鉄・ガラス・レンズの知識：レンズの段の前に
            o["P_K12"] += w * np.mean((ta < t2s) & (life > np.maximum(ta, 0)))       # 顕微鏡・T2 までの知識
            o["P_K2e"] += w * np.mean((ta < t2s + tb) & (life > ta))                # 読取装置を作る知識
            if s == "GEN" and k == "stone":
                tiers_med = {kk: float(np.median(v)) for kk, v in T.items()}
    for k in out:
        for kk in list(out[k]): out[k][kk] /= wsum
        cap = [m[6] for m in MEDIA if m[0] == k][0]
        out[k]["amount_x_P"] = cap * out[k]["E_frac_H"]
    return out, tiers_med

VARS = {
 "main": {},
 "A_Ord": {"anchor": "A_Ord"},
 "m23_100": {"m23": 100.0}, "m23_800": {"m23": 800.0},
 "drive_10_100": {"drive": (10, 100)},
 "salv_0.5": {"p_salv": 0.5}, "salv_0.9": {"p_salv": 0.9},
 "stag_0.9": {"p_stag": 0.9},
}
RES = {}
for vn, v in VARS.items():
    o, tm = run(v); RES[vn] = {"media": o, "GEN_tier_medians": tm}
# 重い崩壊の段の到達の分布（主、重み付きの混合の分位）
def tier_quant():
    rows = {}
    for s in HSC:
        t1, t2, hs = TT[s]
        rows[s] = {"heavy_share": float(hs), "n": int(len(t1)),
                   "t1_q": [float(x) for x in np.percentile(t1, [10, 50, 90])] if len(t1) else None,
                   "t2_q": [float(x) for x in np.percentile(t2, [10, 50, 90])] if len(t2) else None}
    return rows
TQ = tier_quant()

# ---- 二段構えの検査：要点（1e7〜1e8 字）と大量（1e10〜1e12 字）を置く最良の媒体 ----
def best(o, need_K):
    return max(o, key=lambda k: o[k][need_K])
main = RES["main"]["media"]
lowtier = [m[0] for m in MEDIA if m[2] in ("R0", "R1")]
hightier = [m[0] for m in MEDIA if m[2] in ("R2e", "R3", "R3d")]
best_low_amt = max(lowtier, key=lambda k: main[k]["amount_x_P"])
best_high_amt = max(hightier, key=lambda k: main[k]["amount_x_P"])
H153 = {
 "a_R0R1_long_min": min(main[k]["P_read_H"] for k in ("stone", "mom_eye", "mom_mf")),
 "a_R3_max": max(main[k]["P_read_H"] for k in ("mdisc", "lto", "ssd")),
 "b_best_low": best_low_amt, "b_best_high": best_high_amt,
 "b_ratio": main[best_high_amt]["amount_x_P"] / main[best_low_amt]["amount_x_P"],
 "b_argmax_all": max(main, key=lambda k: main[k]["amount_x_P"]),
 "c_K01_R1_over_R0_main": main["mom_mf"]["P_K01"] / main["mom_eye"]["P_K01"],
 "c_K01_R1_over_R0_salv05": RES["salv_0.5"]["media"]["mom_mf"]["P_K01"] / RES["salv_0.5"]["media"]["mom_eye"]["P_K01"],
 "c_K01_R1_over_R0_salv09": RES["salv_0.9"]["media"]["mom_mf"]["P_K01"] / RES["salv_0.9"]["media"]["mom_eye"]["P_K01"],
}
# H153d：単位面積あたり（字/cm²）×P。fiche24 は 98 頁/155.4 cm²、手読み QR は 021 の画素 5.93 µm・1 bit・7 複製から
area = {"fiche24": 98 * PAGE / (10.5 * 14.8), "fiche48": 240 * PAGE / (4 * 6 * 6.4516),
        "piql_hand": (1e4 / (7 * 0.847)) ** 2 / 8 / 7, "mom_mf": 5e6 / 400, "mom_eye": 5e4 / 400, "rosetta": 13000 * PAGE / (np.pi * (3 * 2.54 / 2) ** 2)}
# 登録外（入力判明後）：同じ光学（021 の画素 7×0.847 µm）で 5×7 画素の字形＋1 画素の間隔の文字を直接焼く場合
area["microtext_same_optics"] = 1e8 / (6 * 8 * (7 * 0.847) ** 2)
main["microtext_same_optics"] = {"P_read_H": main["piql_hand"]["P_read_H"]}
dens = {k: {"chars_per_cm2": float(v), "x_P_read_H": float(v * main[k]["P_read_H"])} for k, v in area.items()}
H153["d_fiche24_over_hand"] = dens["fiche24"]["x_P_read_H"] / dens["piql_hand"]["x_P_read_H"]
H153["d_fiche48_over_hand"] = dens["fiche48"]["x_P_read_H"] / dens["piql_hand"]["x_P_read_H"]
judg = {
 "H153a": bool(H153["a_R0R1_long_min"] >= 0.85 and H153["a_R3_max"] < 0.25),
 "H153b": bool(H153["b_best_high"] in ("piql", "silica") and H153["b_ratio"] >= 1e3 and H153["b_argmax_all"] in ("piql", "silica")),
 "H153c_main": bool(H153["c_K01_R1_over_R0_main"] < 0.2),
 "H153c_salv05": bool(H153["c_K01_R1_over_R0_salv05"] < 0.2),
 "H153d": bool(H153["d_fiche24_over_hand"] > 1.0),
}
R = {"study": "141-medium-tier-readability", "seed": 20261141, "python": platform.python_version(),
     "check_038_reproduced": True, "page_chars_assumed": PAGE, "horizon": H,
     "media_table": [dict(zip(["key", "name", "tier", "life_lo", "life_hi", "unit", "chars_per_unit", "basis"], m)) for m in MEDIA],
     "tier_arrival_by_scenario": TQ, "variants": RES, "density": dens, "H153": H153, "judgement": judg,
     "runtime_s": round(time.time() - T0, 1)}
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
# ---- 表示 ----
print("tier arrival (heavy):", json.dumps(TQ, ensure_ascii=False))
print("GEN tier medians (main):", RES["main"]["GEN_tier_medians"])
print(f"{'media':10s} tier  P_read_H  E_frac   P_K01  P_K12  P_K2e  amount*P")
for m in MEDIA:
    o = main[m[0]]
    print(f"{m[0]:10s} {m[2]:4s} {o['P_read_H']:.3f}    {o['E_frac_H']:.3f}   {o['P_K01']:.3f}  {o['P_K12']:.3f}  {o['P_K2e']:.3f}  {o['amount_x_P']:.3g}")
for vn in VARS:
    if vn == "main": continue
    o = RES[vn]["media"]
    print(vn, {k: round(o[k]["P_read_H"], 3) for k in ("mom_mf", "fiche24", "rosetta", "piql", "silica", "mdisc", "ssd")},
          "K01 mom_mf", round(o["mom_mf"]["P_K01"], 3), "piql E_frac", round(o["piql"]["E_frac_H"], 3))
print("H153", json.dumps(H153, ensure_ascii=False)); print("judgement", judg)
print("density", json.dumps(dens))
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
keys = [m[0] for m in MEDIA]
ax[0].barh(keys, [main[k]["P_read_H"] for k in keys], color="tab:blue", label="P_read (1000 y)")
ax[0].barh(keys, [main[k]["P_K01"] for k in keys], color="tab:red", alpha=0.6, label="readable before lens stage (K01)")
ax[0].set_xlabel("probability within heavy collapse (model quantity)"); ax[0].legend(fontsize=8); ax[0].invert_yaxis()
ax[1].barh(keys, [max(main[k]["amount_x_P"], 1) for k in keys], color="tab:green"); ax[1].set_xscale("log")
ax[1].set_xlabel("chars per unit x expected readable fraction (log)"); ax[1].invert_yaxis()
plt.tight_layout(); plt.savefig(f"{HERE}/fig_medium_tier.png", dpi=110)
print("runtime", R["runtime_s"])
