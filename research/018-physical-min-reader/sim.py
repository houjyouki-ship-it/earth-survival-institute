"""研究018：T3 読取の物理的最小規模（open_problems 54）
Piql 型 QR フィルムを、(M-U) 顕微鏡＋人手で非圧縮データを読む、(M-C) 人手で XZ/LZMA を展開する、
(E) Colossus 級の光電読取＋真空管論理機を自作する、の3経路で、人年・直列所要年数・最小人口を出す。
seed 固定。numpy と標準ライブラリのみ。事前登録：PREREGISTRATION.md
"""
import json, time, glob, lzma, hashlib, sys, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261016
rng = np.random.default_rng(SEED)
N = 200_000
SEC_PER_PY = 2000 * 3600          # 【推測】1人年の作業秒
SEC_SERIAL_YR = 3 * SEC_PER_PY    # 【推測】直列1チーム3交代の年あたり作業秒

def logu(lo, hi, n=N):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))

def q(x, ps=(5, 50, 95)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

R = {"seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform()}

# ---------- 事実入力（sources/excerpts.json） ----------
FACT = dict(
    piql_bytes_per_frame=2e6,            # piql.com writer: 2 MB data density per frame
    piql_px_per_frame=(8.8e6, 16e6),     # Fstoppers 8.8 million data points / CORDIS up to 16m pixels
    piql_reel_bytes=120e9, piql_frames=65000,
    github_archive_bytes=21e12,          # GitHub GUIDE: roughly 21 trillion bytes
    xz_decmem_MiB={"-0": 1, "-6": 9, "-9": 65},   # tukaani xz man preset table
    baby_mem_bits=32 * 32, baby_valves=550, baby_watts=3500,
    colossus_mk1_people=50, colossus_mk1_months=11, colossus_mk1_valves=1600, colossus_mk2_valves=2400,
    rca1930_employees=5500, rca1930_tubes_per_day=310000,
    leeuwenhoek_max_mag=275,
    england_pop_1620=4.689e6,            # 015 sources (OWID / BoE)
)
R["facts"] = FACT

# ---------- 圧縮率 ρ の実測（Python 標準ライブラリのソース、lzma preset 6） ----------
files = sorted(glob.glob(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/*.py"))
blobs = [open(f, "rb").read() for f in files]
raw = b"".join(blobs)
comp = lzma.compress(raw, preset=6)
rho_concat = len(raw) / len(comp)
per = np.array([len(b) / len(lzma.compress(b, preset=6)) for b in blobs if len(b) >= 2000])
sizes = np.array([len(b) for b in blobs])
R["rho"] = dict(n_files=len(files), raw_bytes=len(raw), comp_bytes=len(comp), rho_concat=rho_concat,
                rho_perfile_median=float(np.median(per)), rho_perfile_p10=float(np.percentile(per, 10)),
                rho_perfile_p90=float(np.percentile(per, 90)), sha256_raw=hashlib.sha256(raw).hexdigest(),
                file_size_median=float(np.median(sizes)), file_size_max=int(sizes.max()))
RHO = float(np.median(per))   # 1ファイル単位（GitHub は repo 単位で圧縮）に近い値を主に使う
O_MIN = 1 + 4 * 8 / RHO       # 出力1バイトあたり演算の下限：書き出し1 + 2値判定(≥8/ρ)×4演算

# ---------- 経路 M-U：顕微鏡＋人手で非圧縮 QR を読む ----------
px_per_byte = np.where(rng.uniform(size=N) < 0.5, FACT["piql_px_per_frame"][0], FACT["piql_px_per_frame"][1]) / FACT["piql_bytes_per_frame"]
r_px = logu(1, 20)
s_byte_U = px_per_byte * 2 / r_px                     # 二重読取
PY_MB_U = 1e6 * s_byte_U / SEC_PER_PY
MB_per_yr_100 = 100 * SEC_PER_PY / s_byte_U / 1e6
R["M_U"] = dict(px_per_byte=q(px_per_byte), s_per_byte=q(s_byte_U), PY_per_MB=q(PY_MB_U), MB_per_yr_100people=q(MB_per_yr_100),
                PY_full_reel=q(PY_MB_U * FACT["piql_reel_bytes"] / 1e6),
                PY_github_21TB=q(PY_MB_U * FACT["github_archive_bytes"] / 1e6))

# ---------- 経路 M-C：人手で LZMA を展開 ----------
o = np.exp(rng.uniform(np.log(O_MIN), np.log(10 * O_MIN), N))
t_op = logu(5, 60)
s_decode = o * t_op                                    # 1チームの直列秒/出力バイト
serial_yr_1MB = 1e6 * s_decode / SEC_SERIAL_YR
PY_MB_C = 1e6 * (s_byte_U / RHO + 2 * s_decode) / SEC_PER_PY   # 読取（圧縮後バイト）＋展開（検算2チーム）
ratio_CU = PY_MB_C / PY_MB_U
eps = logu(1e-4, 1e-2)
ops_1MB = 1e6 * o
P_clean_single = np.exp(-ops_1MB * eps)
P_clean_double = np.exp(-ops_1MB * eps ** 2)
R["M_C"] = dict(rho_used=RHO, o_min=O_MIN, ops_per_byte=q(o), t_op=q(t_op), serial_years_1MB=q(serial_yr_1MB),
                PY_per_MB=q(PY_MB_C), ratio_C_over_U=q(ratio_CU),
                P_no_error_1MB_single=q(P_clean_single), P_no_undetected_error_1MB_double=q(P_clean_double))

# ---------- 経路 E：Colossus 級機の自作に要る最小人口 ----------
L_colossus = FACT["colossus_mk1_people"] * FACT["colossus_mk1_months"] / 12
m_core = logu(1, 10)
V = logu(FACT["baby_valves"], FACT["colossus_mk2_valves"])
t_tube = logu(0.5, 5)
U = logu(3, 30)
W = 10.0
f_nf = logu(0.1, 0.4)
f_alloc = logu(0.005, 0.05)
L_core = L_colossus * m_core
L_tubes = V * 3 * t_tube / 250
L_total = U * (L_core + L_tubes)
N_min = L_total / (W * f_nf * f_alloc)
rca_pd_per_tube = FACT["rca1930_employees"] / FACT["rca1930_tubes_per_day"]
R["E"] = dict(L_colossus_py=L_colossus, rca1930_person_days_per_tube=rca_pd_per_tube,
              handcraft_vs_rca=[0.5 / rca_pd_per_tube, 5 / rca_pd_per_tube],
              L_core=q(L_core), L_tubes=q(L_tubes), L_total=q(L_total), N_min=q(N_min),
              share_tubes_in_direct=q(L_tubes / (L_core + L_tubes)),
              spearman_logNmin={k: spearman(np.log(N_min), np.log(v)) for k, v in
                                 dict(m_core=m_core, V=V, t_tube=t_tube, U=U, f_nf=f_nf, f_alloc=f_alloc).items()},
              P_Nmin_lt_1e4=float(np.mean(N_min < 1e4)), P_Nmin_lt_T2pop=float(np.mean(N_min < FACT["england_pop_1620"])))

# ---------- H30d：メモリ壁 ----------
mem_ratio = FACT["xz_decmem_MiB"]["-6"] * 2**20 * 8 / FACT["baby_mem_bits"]
mem_ratio_min = FACT["xz_decmem_MiB"]["-0"] * 2**20 * 8 / FACT["baby_mem_bits"]
R["memory_wall"] = dict(ratio_xz6_over_baby=mem_ratio, ratio_xz0_over_baby=mem_ratio_min,
                        note="辞書がファイルより大きい場合、必要メモリはおおむね非圧縮ファイルの大きさまで下がる（xz man の注記）")

# ---------- 梯子への接続（012 の d01, d12 と Piql 寿命） ----------
Z = rng.standard_normal((3, N))
T2 = 20 * np.exp(0.7 * Z[0]) + 30 * np.exp(0.7 * Z[1])
claim = logu(500, 2000); f = logu(0.3, 1.0)
med_life = claim * f
life = med_life / np.log(2) ** 0.5 * (-np.log(rng.uniform(size=N))) ** 0.5
L_alloc = logu(10, 1000)
S_p = 10.0  # MB
t_U = PY_MB_U * S_p / L_alloc
t_C = np.maximum(serial_yr_1MB, PY_MB_C * S_p / L_alloc)
P = {"manual_uncompressed": float(np.mean(T2 + t_U < life)),
     "manual_xz": float(np.mean(T2 + t_C < life)),
     "T2_only_upper_bound": float(np.mean(T2 < life))}
for m23 in (300, 943, 1600):
    P[f"T3_m23_{m23}"] = float(np.mean(T2 + m23 * np.exp(0.8 * Z[2]) < life))
R["ladder"] = dict(S_p_MB=S_p, P_read_before_life=P, t_manual_U_years=q(t_U), t_manual_C_years=q(t_C),
                   diff_U_minus_T3_943=P["manual_uncompressed"] - P["T3_m23_943"])

# ---------- 判定 ----------
H = {}
H["H30a"] = dict(PY_MB_median=R["M_U"]["PY_per_MB"]["p50"], MB_yr_median=R["M_U"]["MB_per_yr_100people"]["p50"],
                 MB_yr_p5=R["M_U"]["MB_per_yr_100people"]["p5"])
H["H30a"]["supported"] = bool(H["H30a"]["PY_MB_median"] <= 2 and H["H30a"]["MB_yr_median"] >= 10 and H["H30a"]["MB_yr_p5"] >= 1)
H["H30b"] = dict(serial_median=R["M_C"]["serial_years_1MB"]["p50"], ratio_median=R["M_C"]["ratio_C_over_U"]["p50"])
H["H30b"]["supported"] = bool(H["H30b"]["serial_median"] >= 5 and H["H30b"]["ratio_median"] >= 10)
H["H30c"] = dict(N_min_median=R["E"]["N_min"]["p50"], supported=bool(R["E"]["N_min"]["p50"] < 1e4))
H["H30d"] = dict(ratio=mem_ratio, supported=bool(mem_ratio >= 1e4))
H["H30e"] = dict(diff=R["ladder"]["diff_U_minus_T3_943"], supported=bool(R["ladder"]["diff_U_minus_T3_943"] >= 0.3))
H["H30f"] = dict(N_min_p95=R["E"]["N_min"]["p95"], supported=bool(R["E"]["N_min"]["p95"] < FACT["england_pop_1620"]))
R["hypotheses"] = H

# ---------- 探索的（事前登録外） ----------
EX = {}
L_total_bigU = logu(30, 300) * (L_core + L_tubes)
N_bigU = L_total_bigU / (W * f_nf * f_alloc)
EX["N_min_U30_300"] = q(N_bigU)
EX["P_N_bigU_lt_T2pop"] = float(np.mean(N_bigU < FACT["england_pop_1620"]))
# 必要な分業の深さ：N_min を T2 人口に等しくする上流倍率
EX["U_needed_to_reach_T2pop"] = q(U * FACT["england_pop_1620"] / N_min)
# 優先データの量を変えたときの M-U の読める確率
EX["P_manual_U_vs_size"] = {str(s): float(np.mean(T2 + PY_MB_U * s / L_alloc < life)) for s in (1, 10, 100, 1000, 10000)}
EX["P_manual_C_vs_filesize_MB"] = {str(s): float(np.mean(T2 + np.maximum(serial_yr_1MB * s, PY_MB_C * 10 / L_alloc) < life))
                                   for s in (0.01, 0.1, 1, 10)}
R["exploratory"] = EX
R["runtime_seconds"] = time.time() - t0

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.logspace(-3, 4, 80)
ax[0].hist(PY_MB_U, bins=bins, alpha=.6, label="M-U: uncompressed QR, by hand")
ax[0].hist(PY_MB_C, bins=bins, alpha=.6, label="M-C: XZ/LZMA, by hand")
ax[0].set_xscale("log"); ax[0].set_xlabel("person-years per MB of output"); ax[0].set_ylabel("MC draws"); ax[0].legend(fontsize=8)
ax[1].hist(serial_yr_1MB, bins=np.logspace(-1, 3.5, 70), color="C3", alpha=.7)
ax[1].axvline(5, color="k", ls="--", lw=.8); ax[1].set_xscale("log")
ax[1].set_xlabel("serial wall-clock years to hand-decompress 1 MB (one team, 3 shifts)"); ax[1].set_ylabel("MC draws")
fig.tight_layout(); fig.savefig("fig_manual_costs.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(N_min, bins=np.logspace(2, 8, 80), alpha=.7, label="U in [3,30] (registered)")
ax[0].hist(N_bigU, bins=np.logspace(2, 8, 80), alpha=.4, label="U in [30,300] (exploratory)")
ax[0].axvline(1e4, color="k", ls="--", lw=.8); ax[0].axvline(FACT["england_pop_1620"], color="r", ls="--", lw=.8)
ax[0].text(1.1e4, ax[0].get_ylim()[1] * .9, "10k", fontsize=8); ax[0].text(5e6, ax[0].get_ylim()[1] * .9, "England 1620", fontsize=8, color="r")
ax[0].set_xscale("log"); ax[0].set_xlabel("minimum population to build a Colossus-class reader in 10 yr"); ax[0].legend(fontsize=8)
keys = ["manual_uncompressed", "manual_xz", "T3_m23_300", "T3_m23_943", "T3_m23_1600"]
ax[1].bar(range(len(keys)), [P[k] for k in keys], color=["C0", "C1", "C2", "C2", "C2"])
ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels(["hand\nuncompr.", "hand\nXZ", "T3\nm23=300", "T3\nm23=943", "T3\nm23=1600"], fontsize=8)
ax[1].set_ylabel("P(10 MB priority data read before Piql film fails)")
fig.tight_layout(); fig.savefig("fig_nmin_and_ladder.png", dpi=130); plt.close(fig)

json.dump(R, open("results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ["rho", "M_U", "M_C", "E", "memory_wall", "ladder", "hypotheses", "exploratory"]}, ensure_ascii=False, indent=1))
print("runtime", R["runtime_seconds"])
