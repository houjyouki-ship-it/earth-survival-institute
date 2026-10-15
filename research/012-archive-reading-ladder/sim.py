"""研究012：長期保存媒体の「読取技術の梯子」モデル
崩壊後の社会が各媒体の読取技術（階層）を取り戻すまでに、媒体が物理的に残っているか。
T1 媒体の一次資料（primer）が技術回復を早め、短寿命のデジタル媒体を「救う」相乗効果があるか。
seed 固定。numpy のみ。決定論的に再現する。
"""
import json, time, numpy as np
t0 = time.time()
SEED = 20261015
rng = np.random.default_rng(SEED)
N = 200_000
H = 1000.0  # 評価期間（年）

# --- 媒体表（主張寿命は出典の値。tier は読取に要する技術階層） ---
# tier: 0=肉眼, 1=簡単なレンズ(≤~25x), 2=光学顕微鏡(100-650x), 3=電子工学+計算機+専用読取装置, 4=DNA合成化学+シーケンサ
MEDIA = {
  "MDISC_BD":     dict(tier=3, claim=(100, 500),     site_factor="room",   src="Wikipedia M-DISC: NIST IR 8387 100+ yr; Verbatim 'several hundred years'"),
  "Piql_film_QR": dict(tier=3, claim=(500, 2000),    site_factor="vault",  src="Wikipedia Arctic World Archive: at least 500, possibly up to 2,000 years"),
  "Silica_glass": dict(tier=3, claim=(1e4, 1e5),     site_factor="room",   src="Microsoft Project Silica: tens to hundreds of thousands of years"),
  "DNA_silica":   dict(tier=4, claim=(2000, 1e4),    site_factor="room",   src="Grass et al. 2015: 70C 1 week = 2000 yr central Europe; 'millennia' [upper end 1e4 is a guess]"),
  "Microfilm_AgX":dict(tier=1, claim=(300, 500),     site_factor="room",   src="Wikipedia Microform: LE ~500 yr under proper storage [lower end 300 is a guess]"),
  "Rosetta_Ni":   dict(tier=2, claim=(1e3, 1e4),     site_factor="room",   src="Long Now Rosetta: prototype of 10,000-Year Library [range is a guess]"),
  "MOM_ceramic":  dict(tier=1, claim=(1e4, 1e6),     site_factor="vault",  src="Memory of Mankind: at least 1 million years (salt mine) [lower end is a guess]"),
}
WEIBULL_K = 2.0  # 【推測】摩耗型
# 崩壊後の保管条件による寿命割引 f（主張は理想条件）。【推測】
F_RANGE = {"room": (0.1, 1.0), "vault": (0.3, 1.0)}

def logu(lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

def lognorm(med, sig, n):
    return med * np.exp(sig * rng.standard_normal(n))

D23_GRID = [50, 100, 200, 300, 400, 600, 800, 1200, 1600]  # T2→T3 回復の中央値（年）【推測：主走査軸】

# 共通乱数：標準正規・一様を一度だけ引き、格子で使い回す（ペア比較の分散を下げる）
Z = {k: rng.standard_normal(N) for k in ["d01", "d12", "d23", "d34"]}
U_a = rng.uniform(np.log(0.2), np.log(0.7), N)       # primer による期間短縮係数 a（対数一様）【推測】
U_find_p = rng.uniform(0.3, 0.9, N)                  # 拠点が発見・アクセスされる確率【推測】
U_find = {k: rng.uniform(size=N) for k in ["primer", "corpus"]}
U_claim = {m: rng.uniform(size=N) for m in MEDIA}
U_f = {m: rng.uniform(size=N) for m in MEDIA}
U_surv = {m: rng.uniform(size=N) for m in MEDIA}

def media_median(m):
    lo, hi = MEDIA[m]["claim"]; flo, fhi = F_RANGE[MEDIA[m]["site_factor"]]
    claim = np.exp(np.log(lo) + U_claim[m] * (np.log(hi) - np.log(lo)))
    f = np.exp(np.log(flo) + U_f[m] * (np.log(fhi) - np.log(flo)))
    return claim * f, claim, f

def lifetime(m):
    med, _, _ = media_median(m)
    scale = med / np.log(2) ** (1 / WEIBULL_K)
    return scale * (-np.log(U_surv[m])) ** (1 / WEIBULL_K)   # 逆変換で Weibull 寿命

def tier_times(m23, primer_tier=None, primer_found=None, a=None, d23_sig=0.8):
    """崩壊時 t=0 に T0。各階層の回復時刻 T1..T4 を返す。
    primer（tier=primer_tier の媒体に置いた技術回復の手引き）が発見され、社会がその tier に達していれば、
    以降の各段の所要時間を a 倍にする。"""
    d01 = 20 * np.exp(0.7 * Z["d01"])
    d12 = 30 * np.exp(0.7 * Z["d12"])
    d23 = m23 * np.exp(d23_sig * Z["d23"])
    d34 = 50 * np.exp(0.8 * Z["d34"])
    mult = np.ones(N)
    T = np.zeros((5, N))
    steps = [d01, d12, d23, d34]
    for k, d in enumerate(steps):           # 段 k → k+1
        if primer_tier is not None:
            active = primer_found & (k >= primer_tier)
            mult = np.where(active, a, 1.0)
        T[k + 1] = T[k] + d * mult
    return T

def run_cell(m23, primer=None, n_corpus_sites=1, n_primer_sites=1, d23_sig=0.8):
    a = np.exp(U_a)
    p = U_find_p
    if primer is not None:
        pf = 1 - (1 - p) ** n_primer_sites
        primer_found = U_find["primer"] < pf
        # primer 媒体自体が、その tier に達するまで残っている必要がある
        T_tmp = tier_times(m23, None, None, None, d23_sig)
        primer_found &= lifetime(primer) > T_tmp[MEDIA[primer]["tier"]]
        T = tier_times(m23, MEDIA[primer]["tier"], primer_found, a, d23_sig)
    else:
        T = tier_times(m23, None, None, None, d23_sig)
    cf = 1 - (1 - p) ** n_corpus_sites
    corpus_found = U_find["corpus"] < cf
    out = {"E_T3_capped": float(np.mean(np.minimum(T[3], H))), "P_T3_le_H": float(np.mean(T[3] <= H)),
           "median_T3": float(np.median(T[3]))}
    read = {}
    read_nofind = {}
    surv = {}
    for m, spec in MEDIA.items():
        tk = T[spec["tier"]]
        alive = lifetime(m) > tk
        surv[m] = float(np.mean(alive))                    # 評価期間の上限なし：読取階層に戻るまで物理的に残るか
        ok_phys = alive & (tk <= H)
        read_nofind[m] = float(np.mean(ok_phys))
        read[m] = float(np.mean(ok_phys & corpus_found))
    out["P_read"] = read; out["P_read_if_found"] = read_nofind; out["P_survive_to_reader"] = surv
    return out, T

results = {"seed": SEED, "N": N, "H_years": H, "D23_grid": D23_GRID,
           "media": {m: {k: v for k, v in s.items()} for m, s in MEDIA.items()},
           "assumptions": {"weibull_k": WEIBULL_K, "F_RANGE": F_RANGE, "a_range": [0.2, 0.7], "p_find_range": [0.3, 0.9],
                           "D01": "LN(20, 0.7)", "D12": "LN(30, 0.7)", "D23": "LN(m23, 0.8)", "D34": "LN(50, 0.8)",
                           "label": "【推測】 all climb-time, derating, discovery and acceleration priors are assumptions"}}

# --- 1. 媒体ごとの「読めるまで残る」確率：primer なし／MOM primer あり ---
grid = {}
for m23 in D23_GRID:
    base, Tb = run_cell(m23, None)
    prim, Tp = run_cell(m23, "MOM_ceramic")
    grid[str(m23)] = {"no_primer": base, "MOM_primer": prim,
                      "synergy_P_read": {m: prim["P_read"][m] - base["P_read"][m] for m in MEDIA},
                      "synergy_P_read_if_found": {m: prim["P_read_if_found"][m] - base["P_read_if_found"][m] for m in MEDIA},
                      "E_T3_reduction": base["E_T3_capped"] - prim["E_T3_capped"]}
results["grid"] = grid

# --- 2. 臨界 D23*：発見を条件に P_read_if_found = 0.5 となる D23 中央値（primer なし／あり） ---
# v1 では P_read_if_found（≤H の打ち切りを含む）で臨界値を出し、長寿命媒体で「期間切れ」と「劣化」が混ざった。
# v2 では打ち切りなしの P_survive_to_reader を使う（定義の修正。報告書に記載）。
def crit(key, m, metric="P_survive_to_reader"):
    xs = np.array(D23_GRID, float); ys = np.array([grid[str(x)][key][metric][m] for x in D23_GRID])
    if ys[0] < 0.5: return "<%d" % D23_GRID[0]
    if ys[-1] >= 0.5: return ">%d" % D23_GRID[-1]
    i = int(np.where(ys < 0.5)[0][0])
    lx = np.log(xs[i-1]) + (0.5 - ys[i-1]) * (np.log(xs[i]) - np.log(xs[i-1])) / (ys[i] - ys[i-1])
    return float(np.exp(lx))
results["D23_star"] = {m: {"no_primer": crit("no_primer", m), "MOM_primer": crit("MOM_primer", m)} for m in MEDIA}
results["D23_star_v1_horizon_confounded"] = {m: {"no_primer": crit("no_primer", m, "P_read_if_found"), "MOM_primer": crit("MOM_primer", m, "P_read_if_found")} for m in MEDIA}

# --- 3. 拠点数の感度（m23=300） ---
sites = {}
for nc, npr in [(1, 1), (3, 1), (1, 3), (3, 3)]:
    b, _ = run_cell(300, None, nc, npr); p_, _ = run_cell(300, "MOM_ceramic", nc, npr)
    sites[f"corpus{nc}_primer{npr}"] = {"no_primer_P_read": b["P_read"], "MOM_primer_P_read": p_["P_read"]}
results["sites_m23_300"] = sites

# --- 4. D23 の幅（σ）の感度 ---
sig = {}
for s in [0.4, 0.8, 1.2]:
    b, _ = run_cell(300, None, d23_sig=s); p_, _ = run_cell(300, "MOM_ceramic", d23_sig=s)
    sig[str(s)] = {"P_read_if_found_no": b["P_read_if_found"], "P_read_if_found_MOM": p_["P_read_if_found"]}
results["sigma_d23_m23_300"] = sig

# --- 5. 大域感度：m23 を対数一様[50,1600]で引き、Spearman 順位相関（MDISC と Piql の可読、primer なし） ---
def rank(x): r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r
def spearman(x, y): return float(np.corrcoef(rank(x), rank(y))[0, 1])
m23_draw = np.exp(rng.uniform(np.log(50), np.log(1600), N))
d23 = m23_draw * np.exp(0.8 * Z["d23"])
T3 = 20*np.exp(0.7*Z["d01"]) + 30*np.exp(0.7*Z["d12"]) + d23
gs = {}
for m in ["MDISC_BD", "Piql_film_QR"]:
    med, claim, f = media_median(m)
    margin = np.log(lifetime(m)) - np.log(T3)          # 連続指標：寿命/読取回復時刻の対数比
    gs[m] = {"rho_m23": spearman(m23_draw, margin), "rho_f": spearman(f, margin), "rho_claim": spearman(claim, margin),
             "P_read_if_found": float(np.mean(margin > 0))}
results["global_sensitivity"] = gs

# --- 6. 費用・物量計算（出典値のみ。chars/word は【推測】） ---
WORDS = 5.29e9; GB_COMP = 26.80
cost = {}
for cpw in [5.0, 6.0, 7.0]:
    chars = WORDS * cpw; tabs = chars / 5e6
    cost[f"cpw{cpw}"] = {"chars": chars, "MOM_tablets": tabs, "MOM_EUR_list": tabs * 195.0}
cost["primer_1e8_chars"] = {"MOM_tablets": 1e8 / 5e6, "MOM_EUR_list": 1e8 / 5e6 * 195.0}
cost["primer_1e7_chars"] = {"MOM_tablets": 1e7 / 5e6, "MOM_EUR_list": 1e7 / 5e6 * 195.0}
cost["wiki_compressed_GB"] = GB_COMP
cost["MDISC_BDXL100_discs"] = int(np.ceil(GB_COMP / 100)); cost["MDISC_BD25_discs"] = int(np.ceil(GB_COMP / 25))
cost["Silica_7TB_platters_fraction"] = GB_COMP / 7000
cost["Piql_GB_per_reel_GitHub"] = 21000 / 186; cost["Piql_reels"] = GB_COMP / (21000 / 186)
cost["DNA_write_USD_2013rate"] = GB_COMP * 1000 * 12400; cost["DNA_write_USD_2017rate"] = GB_COMP * 1000 * 3500
results["cost_volume"] = cost

# --- 事前登録した判定 ---
ds = results["D23_star"]
def num(x): return x if isinstance(x, float) else (1e9 if str(x).startswith(">") else 0.0)
g300 = grid["300"]
H = {}
H["H24a"] = {"claim": "D23*(M-DISC, no primer) < 200 and 300 <= D23*(Piql, no primer) <= 1000 and D23*(Silica), D23*(MOM) > 1600",
             "values": {m: ds[m]["no_primer"] for m in ["MDISC_BD", "Piql_film_QR", "Silica_glass", "MOM_ceramic"]}}
H["H24a"]["supported"] = bool(num(ds["MDISC_BD"]["no_primer"]) < 200 and 300 <= num(ds["Piql_film_QR"]["no_primer"]) <= 1000
                              and num(ds["Silica_glass"]["no_primer"]) > 1600 and num(ds["MOM_ceramic"]["no_primer"]) > 1600)
H["H24b"] = {"claim": "at m23=300, synergy in P_read (with discovery) >= +0.10 for MDISC_BD and Piql_film_QR",
             "values": {m: g300["synergy_P_read"][m] for m in ["MDISC_BD", "Piql_film_QR"]}}
H["H24b"]["supported"] = bool(all(v >= 0.10 for v in H["H24b"]["values"].values()))
H["H24c"] = {"claim": "Wikipedia-text on MOM list price > EUR 1M for cpw in [5,7]; 1e8-char primer < EUR 10k",
             "values": {k: cost[k]["MOM_EUR_list"] for k in ["cpw5.0", "cpw7.0", "primer_1e8_chars"]}}
H["H24c"]["supported"] = bool(cost["cpw5.0"]["MOM_EUR_list"] > 1e6 and cost["primer_1e8_chars"]["MOM_EUR_list"] < 1e4)
H["H24d"] = {"claim": "synergy for Silica_glass < 0.02 at every m23 grid point (with discovery)",
             "values": {k: grid[k]["synergy_P_read"]["Silica_glass"] for k in grid}}
H["H24d"]["supported"] = bool(max(H["H24d"]["values"].values()) < 0.02)
H["H24e"] = {"claim": "|rho(m23)| > |rho(f)| for both MDISC and Piql (global sensitivity)", "values": gs}
H["H24e"]["supported"] = bool(all(abs(gs[m]["rho_m23"]) > abs(gs[m]["rho_f"]) for m in gs))
results["hypotheses"] = H

# --- 図 ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
cols = dict(zip(MEDIA, plt.cm.tab10.colors))
for m in MEDIA:
    y0 = [grid[str(x)]["no_primer"]["P_survive_to_reader"][m] for x in D23_GRID]
    y1 = [grid[str(x)]["MOM_primer"]["P_survive_to_reader"][m] for x in D23_GRID]
    ax[0].plot(D23_GRID, y0, "-o", color=cols[m], label=f"{m} (tier {MEDIA[m]['tier']})", ms=3)
    ax[0].plot(D23_GRID, y1, "--", color=cols[m], lw=1)
ax[0].axhline(0.5, color="k", lw=0.5); ax[0].set_xscale("log")
ax[0].set_xlabel("median years to regain electronics (T2->T3), m23"); ax[0].set_ylabel("P(medium physically survives until its reader tier is regained)")
ax[0].set_title("solid: no primer / dashed: + ceramic (tier-1) primer"); ax[0].legend(fontsize=7)
for m in ["MDISC_BD", "Piql_film_QR", "Silica_glass", "DNA_silica"]:
    ax[1].plot(D23_GRID, [grid[str(x)]["synergy_P_read"][m] for x in D23_GRID], "-o", color=cols[m], label=m, ms=3)
ax[1].plot(D23_GRID, [grid[str(x)]["E_T3_reduction"] / 1000 for x in D23_GRID], "k:", label="E[T3] reduction /1000 y")
ax[1].set_xscale("log"); ax[1].axhline(0.10, color="gray", lw=0.5)
ax[1].set_xlabel("m23 (years)"); ax[1].set_ylabel("synergy: P_read(with primer) - P_read(without)")
ax[1].set_title("primer rescues short-lived digital corpora (incl. site discovery)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_read_prob_and_synergy.png", dpi=130); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
keys = list(sites); xs = np.arange(len(keys))
for j, m in enumerate(["MDISC_BD", "Piql_film_QR", "Silica_glass"]):
    ax[0].bar(xs + j*0.27 - 0.27, [sites[k]["MOM_primer_P_read"][m] for k in keys], 0.27, color=cols[m], label=m + " (+primer)")
    ax[0].scatter(xs + j*0.27 - 0.27, [sites[k]["no_primer_P_read"][m] for k in keys], color="k", s=10, zorder=3)
ax[0].set_xticks(xs); ax[0].set_xticklabels(keys, fontsize=7); ax[0].set_ylabel("P_read (m23=300)"); ax[0].legend(fontsize=7)
ax[0].set_title("site redundancy (dots: no primer)")
labs = ["rho_m23", "rho_f", "rho_claim"]
for j, m in enumerate(gs):
    ax[1].bar(np.arange(3) + j*0.4 - 0.2, [gs[m][l] for l in labs], 0.4, color=cols[m], label=m)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["m23 (climb time)", "f (storage derating)", "claimed life"])
ax[1].set_ylabel("Spearman rho with log(life / T3)"); ax[1].legend(fontsize=8); ax[1].axhline(0, color="k", lw=0.5)
plt.tight_layout(); plt.savefig("fig_sites_and_sensitivity.png", dpi=130); plt.close()

results["runtime_seconds"] = time.time() - t0
json.dump(results, open("results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"D23_star": results["D23_star"], "hypotheses": {k: (v["supported"], v["values"]) for k, v in H.items()}}, ensure_ascii=False, indent=1, default=str))
print("m23=300:", json.dumps({k: g300[k] for k in ["synergy_P_read", "E_T3_reduction"]}, indent=1))
print("no_primer m23=300", json.dumps(g300["no_primer"], indent=1)); print("MOM m23=300", json.dumps(g300["MOM_primer"], indent=1))
print("sites", json.dumps(sites, indent=1)); print("sigma", json.dumps(sig, indent=1)); print("cost", json.dumps(cost, indent=1))
print("D23_star_v1", json.dumps(results["D23_star_v1_horizon_confounded"], default=str))
print("runtime_sec=%.1f" % results["runtime_seconds"])
