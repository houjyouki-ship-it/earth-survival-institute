"""研究020：手読み層の残留誤りを 1e-6 まで下げる最安の書式（open_problems 66・67）
019 の channel・費用モデルを再利用・拡張。seed 固定。numpy＋標準ライブラリ。
事前登録：PREREGISTRATION.md（2026-10-16T08:39:25Z）— 実行後に仮説を書き換えない。
"""
import json, time, sys, platform, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261020
rng = np.random.default_rng(SEED)
N = 100_000
SEC_PER_PY = 2000 * 3600
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
R = {"seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform(),
     "preregistration": "2026-10-16T08:39:25Z"}

M_SINGLE = math.sqrt(1e-4 * 3e-3)
M_DOUBLE_RES = M_SINGLE * 0.1
BURST = (1e-3, 100)
r019 = json.load(open(f"{HERE}/../019-hand-readable-format/results.json"))
MU018 = r019["facts"]["m018_MU_median_py_per_MB"]
A2_019 = None
for d in r019["binary_formats"]["grid"]:
    if d["p_film"] == 1e-3 and d["burst"]:
        A2_019 = d["A2_rep3_adaptive"]
R["inherited"] = dict(M_SINGLE=M_SINGLE, M_DOUBLE_RES=M_DOUBLE_RES, BURST=list(BURST),
                      MU018=MU018, A2_019_cost_p50=r019["cost_py_per_MB"]["A2_rep3_adaptive"]["p50"],
                      A2_019_residual_p1e3_burst=A2_019)

PX_PER_DATA_BYTE_PIQL = (4096 * 2160) / 2e6
PIQL_REEL = 120e9
SIZE_MB = 10.0
LINE = 256
NB = 2_000_000  # 2 MB

def channel(nbits, p, burst=None, rng=rng):
    flip = rng.random(nbits) < p
    in_burst = np.zeros(nbits, dtype=bool)
    if burst:
        f, L = burst
        starts = np.nonzero(rng.random(nbits) < f / L)[0]
        lens = rng.geometric(1.0 / L, len(starts))
        for s, l in zip(starts, lens):
            e = min(nbits, s + int(l))
            in_burst[s:e] = True
            flip[s:e] ^= (rng.random(e - s) < 0.5)
    return flip.astype(np.uint8), in_burst

def apply_erasure(flip, in_burst, v, rng):
    eras = np.zeros(len(flip), dtype=bool)
    if v > 0 and in_burst.any():
        cand = np.nonzero(in_burst)[0]
        eras[cand[rng.random(len(cand)) < v]] = True
        flip = flip.copy(); flip[eras] = 0
    return flip, eras

data_bits = rng.integers(0, 2, NB * 8, dtype=np.uint8)
nb = data_bits.size

def byte_err(dec):
    return float(np.mean((dec.reshape(-1, 8) != data_bits.reshape(-1, 8)).any(axis=1)))

def make_copies(n_copies, p_film, burst, v_erase, m_read, rng):
    copies, erasures = [], []
    for _ in range(n_copies):
        flip, in_b = channel(nb, p_film, burst, rng=rng)
        flip, eras = apply_erasure(flip, in_b, v_erase, rng)
        read_err = (rng.random(nb) < m_read) & (~eras)
        copies.append(data_bits ^ flip ^ read_err.astype(np.uint8))
        erasures.append(eras)
    return copies, erasures

def decode_adaptive(copies, erasures=None, n_initial=2):
    n = len(copies)
    votes = np.zeros(nb, dtype=np.int16)
    n_read = np.zeros(nb, dtype=np.int16)
    for i in range(n_initial):
        ok = np.ones(nb, dtype=bool) if erasures is None else (~erasures[i])
        votes += copies[i].astype(np.int16) * ok
        n_read += ok.astype(np.int16)
    if erasures is None:
        need = copies[0] != copies[1]
    else:
        both = (~erasures[0]) & (~erasures[1])
        need = ~(both & (copies[0] == copies[1]))
    for i in range(n_initial, n):
        ok = need if erasures is None else (need & (~erasures[i]))
        votes += copies[i].astype(np.int16) * ok
        n_read += ok.astype(np.int16)
    dec = (votes * 2 > n_read).astype(np.uint8)
    return dec, float(need.mean()), float(n_read.mean())

def add_parity(bits):
    n_lines = bits.size // LINE
    d = bits[:n_lines * LINE].reshape(n_lines, LINE)
    p = (d.sum(axis=1) & 1).astype(np.uint8)
    return np.concatenate([d, p[:, None]], axis=1).reshape(-1)

def run_A3p(p_film, burst, m_read, rng):
    n_lines = nb // LINE
    data_trim = data_bits[:n_lines * LINE]
    enc = add_parity(data_trim)
    ne = enc.size
    copies = []
    for _ in range(3):
        flip, _ = channel(ne, p_film, burst, rng=rng)
        copies.append(enc ^ flip ^ (rng.random(ne) < m_read).astype(np.uint8))
    c0 = copies[0].reshape(n_lines, LINE + 1)
    c1 = copies[1].reshape(n_lines, LINE + 1)
    c2 = copies[2].reshape(n_lines, LINE + 1)
    agree = np.all(c0 == c1, axis=1)
    def parity_ok(c):
        return (c[:, :LINE].sum(axis=1) & 1) == c[:, LINE]
    a_ok, b_ok = parity_ok(c0), parity_ok(c1)
    # 偶奇は検出専用：偶数誤りを「正しい側」と誤認しないよう、不一致なら必ず3本目
    agree_ok = agree & a_ok
    need3 = ~agree_ok
    out = c0[:, :LINE].copy()
    if need3.any():
        stack = np.stack([c0[need3, :LINE], c1[need3, :LINE], c2[need3, :LINE]], axis=0)
        out[need3] = (stack.sum(axis=0) >= 2).astype(np.uint8)
    residual = float(np.mean((out.reshape(-1, 8) != data_trim.reshape(-1, 8)).any(axis=1)))
    return dict(residual=residual, agree_line_frac=float(agree.mean()),
                agree_ok_frac=float(agree_ok.mean()),
                third_line_frac=float(need3.mean()), overhead_bits=(LINE + 1) / LINE)


def run_A3c(p_film, burst, m_read, rng):
    """3複製＋行XOR検査和（32データバイトに1検査バイト）。一致でも検査失敗なら3本目。"""
    B = 32
    n_lines = NB // B
    data_b = np.packbits(data_bits)[:n_lines * B].reshape(n_lines, B)
    chk = np.bitwise_xor.reduce(data_b, axis=1).astype(np.uint8)
    enc = np.concatenate([data_b, chk[:, None]], axis=1)
    flat = enc.reshape(-1)
    copies = []
    for _ in range(3):
        bits = np.unpackbits(flat)
        flip, _ = channel(bits.size, p_film, burst, rng=rng)
        bits2 = bits ^ flip ^ (rng.random(bits.size) < m_read).astype(np.uint8)
        copies.append(np.packbits(bits2).reshape(n_lines, B + 1))
    c0, c1, c2 = copies
    agree = np.all(c0 == c1, axis=1)
    def cok(c):
        return np.bitwise_xor.reduce(c[:, :B], axis=1) == c[:, B]
    a_ok, b_ok = cok(c0), cok(c1)
    agree_ok = agree & a_ok
    need3 = ~agree_ok
    out = c0[:, :B].copy()
    if need3.any():
        for j in range(B):
            maj = c0[need3, j].copy()
            eq12 = c1[need3, j] == c2[need3, j]
            eq01 = c0[need3, j] == c1[need3, j]
            maj[eq12 & ~eq01] = c1[need3, j][eq12 & ~eq01]
            out[need3, j] = maj
    residual = float(np.mean(out.reshape(-1) != data_b.reshape(-1)))
    return dict(residual=residual, agree_line_frac=float(agree.mean()),
                agree_ok_frac=float(agree_ok.mean()),
                third_line_frac=float(need3.mean()), overhead_bits=(B + 1) / B)

def run_glyph(p_film, burst, v_erase, m_char, cell, n_rep, rng):
    """文字書式。フィルムの独立誤りは文字あたり約 critical_px 画素相当【推測】。
    バーストは文字単位に縮約。目視できる損傷は消失。読み誤り m_char は置換。"""
    S = cell
    n_chars = NB
    data_c = np.packbits(data_bits)
    # 臨界画素数4【推測】：文字を誤認させるのに足りる損傷画素
    critical_px = 4
    q = 1.0 - (1.0 - p_film) ** critical_px
    L_char = max(1, BURST[1] // max(1, S // 2))
    burst_c = (BURST[0], L_char) if burst else None
    copies, erasures = [], []
    for _ in range(n_rep):
        flip, in_b = channel(n_chars, q, burst_c, rng=rng)
        flip, eras = apply_erasure(flip.astype(np.uint8), in_b, v_erase, rng)
        rx = data_c.copy()
        bad = flip.astype(bool) | ((rng.random(n_chars) < m_char) & (~eras))
        idx = np.nonzero(bad)[0]
        if len(idx):
            rx[idx] = rng.integers(0, 256, len(idx), dtype=np.uint8)
            same = rx[idx] == data_c[idx]
            while same.any():
                rx[idx[same]] = rng.integers(0, 256, int(same.sum()), dtype=np.uint8)
                same = rx[idx] == data_c[idx]
        copies.append(rx); erasures.append(eras)
    c0, c1 = copies[0], copies[1]
    e0, e1 = erasures[0], erasures[1]
    both = (~e0) & (~e1)
    agree = both & (c0 == c1)
    need = ~agree
    dec = c0.copy()
    dec[e0 & ~e1] = c1[e0 & ~e1]
    if need.any() and n_rep >= 3:
        for i in np.nonzero(need)[0]:
            ballots = [int(copies[r][i]) for r in range(n_rep) if not erasures[r][i]]
            if ballots:
                vals, cts = np.unique(ballots, return_counts=True)
                dec[i] = vals[np.argmax(cts)]
            else:
                dec[i] = 0
    return dict(residual=float(np.mean(dec != data_c)), need_more_frac=float(need.mean()),
                q_char=q, cell=S, n_rep=n_rep, px_per_char=S * S,
                critical_px=critical_px, v=v_erase, m_char=m_char)

print("residuals...", flush=True)
grid = []
for cond in (dict(label="iid_p1e-3", p=1e-3, burst=False),
             dict(label="burst_p1e-3", p=1e-3, burst=True),
             dict(label="burst_p1e-4", p=1e-4, burst=True),
             dict(label="burst_p1e-5", p=1e-5, burst=True)):
    p, burst = cond["p"], (BURST if cond["burst"] else None)
    row = dict(label=cond["label"], p_film=p, burst=cond["burst"])
    cps, _ = make_copies(3, p, burst, 0.0, M_SINGLE, rng)
    dec, frac, mr = decode_adaptive(cps, None, 2)
    row["A2"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mr)
    cps, _ = make_copies(5, p, burst, 0.0, M_SINGLE, rng)
    dec, frac, mr = decode_adaptive(cps, None, 2)
    row["A5"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mr)
    # 探索的：常時5本多数決（適応は先頭2本誤一致で3複製床に落ちる）
    votes = sum(c.astype(np.int16) for c in cps)
    decf = (votes >= 3).astype(np.uint8)
    row["A5f"] = dict(residual=byte_err(decf), extra_frac=1.0, mean_reads=5.0)
    # 探索的：5本を二重読み（読み誤り m=M_DOUBLE_RES）して多数決
    cps_d, _ = make_copies(5, p, burst, 0.0, M_DOUBLE_RES, rng)
    votes_d = sum(c.astype(np.int16) for c in cps_d)
    row["A5fd"] = dict(residual=byte_err((votes_d >= 3).astype(np.uint8)), extra_frac=1.0, mean_reads=10.0)
    # 探索的：7本多数決（1回読み）
    cps7, _ = make_copies(7, p, burst, 0.0, M_SINGLE, rng)
    votes7 = sum(c.astype(np.int16) for c in cps7)
    row["A7f"] = dict(residual=byte_err((votes7 >= 4).astype(np.uint8)), extra_frac=1.0, mean_reads=7.0)
    for v in (0.3, 0.5, 0.7, 0.9):
        cps, ers = make_copies(3, p, burst, v, M_SINGLE, rng)
        dec, frac, mr = decode_adaptive(cps, ers, 2)
        row[f"A3e_v{v}"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mr, v=v)
    row["A3p"] = run_A3p(p, burst, M_SINGLE, rng)
    row["A3c"] = run_A3c(p, burst, M_SINGLE, rng)
    grid.append(row)
    print(cond["label"], {k: row[k]["residual"] for k in row if isinstance(row.get(k), dict) and "residual" in row[k]}, flush=True)
R["residual_grid"] = grid
g_burst = next(r for r in grid if r["label"] == "burst_p1e-3")

print("glyph...", flush=True)
glyph_grid = []
for cell in (8, 16):
    for v in (0.5, 0.9):
        for m_char in (1e-4, 1e-3):
            for n_rep in (3,):
                glyph_grid.append(run_glyph(1e-3, True, v, m_char, cell, n_rep, rng))
R["glyph_grid"] = glyph_grid
g_ref = next(g for g in glyph_grid if g["cell"] == 8 and g["v"] == 0.9 and g["m_char"] == 1e-4 and g["n_rep"] == 3)

print("seed replicates...", flush=True)
reps = []
for sd in range(1, 6):
    rr = np.random.default_rng(SEED + 200 + sd)
    row = {"seed": int(SEED + 200 + sd)}
    for name, n_rep, v in (("A2", 3, 0.0), ("A5", 5, 0.0), ("A3e_v0.5", 3, 0.5)):
        cps, ers = make_copies(n_rep, 1e-3, BURST, v, M_SINGLE, rr)
        dec, _, _ = decode_adaptive(cps, ers if v > 0 else None, 2)
        row[name] = byte_err(dec)
        if name == "A5":
            votes = sum(c.astype(np.int16) for c in cps)
            row["A5f"] = byte_err((votes >= 3).astype(np.uint8))
            cps_d, _ = make_copies(5, 1e-3, BURST, 0.0, M_DOUBLE_RES, rr)
            row["A5fd"] = byte_err((sum(c.astype(np.int16) for c in cps_d) >= 3).astype(np.uint8))
            cps7, _ = make_copies(7, 1e-3, BURST, 0.0, M_SINGLE, rr)
            row["A7f"] = byte_err((sum(c.astype(np.int16) for c in cps7) >= 4).astype(np.uint8))
    row["A3p"] = run_A3p(1e-3, BURST, M_SINGLE, rr)["residual"]
    row["A3c"] = run_A3c(1e-3, BURST, M_SINGLE, rr)["residual"]
    reps.append(row); print(row, flush=True)
R["seed_replicates"] = reps

print("costs...", flush=True)
def logu(lo, hi):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), N))
t_px2 = logu(0.05, 1.0)
t_bit = logu(0.2, 2.0)
mu_microscope = logu(2.0, 20.0)
t_char_paper = 60.0 / (200.0 * 5.0)  # 0.06 s/char @ 200 wpm, 5 chars/word
t_char = t_char_paper * mu_microscope

def py(sec_per_byte):
    return sec_per_byte * 1e6 / SEC_PER_PY

cost = {}
cost["A2"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25 + g_burst["A2"]["extra_frac"] * 8 * t_px2)
cost["A5"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25 + g_burst["A5"]["extra_frac"] * 3 * 8 * t_px2)
cost["A5f"] = py(5 * 8 * t_px2 + 8 * t_bit * 0.5)
cost["A5fd"] = py(10 * 8 * t_px2 + 8 * t_bit * 0.5)  # 5本×二重読み
cost["A7f"] = py(7 * 8 * t_px2 + 8 * t_bit * 0.5)
for v in (0.3, 0.5, 0.7, 0.9):
    k = f"A3e_v{v}"
    cost[k] = py(g_burst[k]["mean_reads"] * 8 * t_px2 + 8 * t_bit * 0.3)
oh = (LINE + 1) / LINE
frac3 = g_burst["A3p"]["third_line_frac"]
cost["A3p"] = py(oh * (2 * 8 * t_px2 + 8 * t_bit * 0.25 + 2 * (8 * t_bit / 32) + frac3 * 8 * t_px2))
ohc = g_burst["A3c"]["overhead_bits"]; frac3c = g_burst["A3c"]["third_line_frac"]
cost["A3c"] = py(ohc * (2 * 8 * t_px2 + 8 * t_bit * 0.25 + 2 * (t_bit) + frac3c * 8 * t_px2))  # 行XORはバイト走査
mr_g = 2 + g_ref["need_more_frac"]
cost["G_cell8_v0.9"] = py(mr_g * t_char)

def qs(x):
    return {k: float(np.percentile(x, v)) for k, v in (("p5", 5), ("p50", 50), ("p95", 95))}
R["cost_py_per_MB"] = {k: qs(v) for k, v in cost.items()}
R["cost_extras"] = dict(
    A2_extra=float(g_burst["A2"]["extra_frac"]), A5_extra=float(g_burst["A5"]["extra_frac"]),
    A3e_v0_5_extra=float(g_burst["A3e_v0.5"]["extra_frac"]),
    A3e_v0_5_mean_reads=float(g_burst["A3e_v0.5"]["mean_reads"]),
    A3p_third_line=float(g_burst["A3p"]["third_line_frac"]),
    A3p_agree_ok=float(g_burst["A3p"].get("agree_ok_frac", float("nan"))),
    glyph_need_more=float(g_ref["need_more_frac"]), t_char_paper_sec=t_char_paper,
)

area = {}
px = {"A2": 24, "A5": 40, "A5f": 40, "A5fd": 40, "A7f": 56, "A3p": 24 * oh, "A3e": 24, "A3c": 24 * 33 / 32,
      "G_cell8": 3 * 64, "G_cell16": 3 * 256}
for k, v in px.items():
    for s in (1, 2, 5):
        eq = SIZE_MB * 1e6 * v * s * s / PX_PER_DATA_BYTE_PIQL
        area[f"{k}_s{s}"] = dict(px_per_data_byte=v, linear_scale=s,
                                 frac_of_reel=eq / PIQL_REEL, piql_equiv_MB=eq / 1e6)
R["area"] = area

def meets(res): return res <= 1e-6
H = {}
A5_res = g_burst["A5"]["residual"]
H["H32a"] = dict(A5_residual=A5_res, threshold=1e-6, supported=bool(meets(A5_res)),
                 seed_pass_frac=float(np.mean([r["A5"] <= 1e-6 for r in reps])))
A3e_res = g_burst["A3e_v0.5"]["residual"]
A3e_cost = R["cost_py_per_MB"]["A3e_v0.5"]["p50"]
A5_cost = R["cost_py_per_MB"]["A5"]["p50"]
H["H32b"] = dict(residual=A3e_res, cost=A3e_cost, A5_cost=A5_cost,
                 residual_ok=bool(meets(A3e_res)), cost_ok=bool(A3e_cost <= A5_cost),
                 supported=bool(meets(A3e_res) and A3e_cost <= A5_cost),
                 seed_pass_frac=float(np.mean([r["A3e_v0.5"] <= 1e-6 for r in reps])))
A3p_res = g_burst["A3p"]["residual"]
A3p_cost = R["cost_py_per_MB"]["A3p"]["p50"]
H["H32c"] = dict(residual=A3p_res, cost=A3p_cost, A5_cost=A5_cost,
                 residual_ok=bool(meets(A3p_res)), cost_ok=bool(A3p_cost <= A5_cost),
                 supported=bool(meets(A3p_res) and A3p_cost <= A5_cost),
                 seed_pass_frac=float(np.mean([r["A3p"] <= 1e-6 for r in reps])))
candidates = []
for name, res in (("A5", A5_res), ("A3p", A3p_res), ("A3e_v0.5", A3e_res)):
    candidates.append(dict(name=name, residual=res, cost=R["cost_py_per_MB"][name]["p50"], meets=bool(meets(res)), registered=True))
G_res, G_cost = g_ref["residual"], R["cost_py_per_MB"]["G_cell8_v0.9"]["p50"]
candidates.append(dict(name="G_cell8_v0.9", residual=G_res, cost=G_cost, meets=bool(meets(G_res)), registered=True))
# 探索的（事前登録外）：常時5本多数決、行XOR検査和
A5f_res = g_burst["A5f"]["residual"]; A3c_res = g_burst["A3c"]["residual"]
candidates.append(dict(name="A5f", residual=A5f_res, cost=R["cost_py_per_MB"]["A5f"]["p50"], meets=bool(meets(A5f_res)), registered=False))
candidates.append(dict(name="A3c", residual=A3c_res, cost=R["cost_py_per_MB"]["A3c"]["p50"], meets=bool(meets(A3c_res)), registered=False))
for ename in ("A5fd", "A7f"):
    candidates.append(dict(name=ename, residual=g_burst[ename]["residual"],
                           cost=R["cost_py_per_MB"][ename]["p50"], meets=bool(meets(g_burst[ename]["residual"])), registered=False))
ok_reg = [c for c in candidates if c["meets"] and c["registered"]]
ok_all = [c for c in candidates if c["meets"]]
cheapest_reg = min(ok_reg, key=lambda c: c["cost"]) if ok_reg else None
cheapest = min(ok_all, key=lambda c: c["cost"]) if ok_all else None
H["H32d"] = dict(candidates=candidates, cheapest_registered=(cheapest_reg["name"] if cheapest_reg else None),
                 cheapest_including_exploratory=(cheapest["name"] if cheapest else None),
                 supported=bool(cheapest_reg is not None and cheapest_reg["name"] == "A3e_v0.5"),
                 note="適応A5は先頭2本の誤一致で3複製床に落ちる。A5f/A3cは探索的。")
best_dot = min((c for c in candidates if c["name"] != "G_cell8_v0.9" and c["meets"]),
               key=lambda c: c["cost"], default=None)
area_G = area["G_cell8_s2"]["frac_of_reel"]
area_dot = None
if best_dot:
    ak = best_dot["name"]
    if ak.startswith("A3e"): ak = "A3e"
    if f"{ak}_s2" not in area:
        ak = "A5" if ak.startswith("A5") else ("A7f" if ak.startswith("A7") else "A2")
    area_dot = area[f"{ak}_s2"]["frac_of_reel"]
H["H32e"] = dict(G_cost=G_cost, best_dot_cost=(best_dot["cost"] if best_dot else None),
                 best_dot_name=(best_dot["name"] if best_dot else None),
                 G_cheaper_in_PY=(bool(G_cost < best_dot["cost"]) if best_dot else None),
                 area_G_s2=area_G, area_dot_s2=area_dot,
                 G_more_area=(bool(area_G > area_dot) if area_dot is not None else None),
                 supported=bool(best_dot and G_cost < best_dot["cost"] and area_G > area_dot),
                 note="読取速度は紙の校正200語/分(Ziefle 1998 via Bailey)を顕微鏡減速μ∈[2,20]で割った【推測】。Rosetta実測速度は未確認。")
R["hypotheses"] = H
R["recommendation"] = dict(
    meeting_1e6=[c["name"] for c in candidates if c["meets"]],
    cheapest_meeting=(cheapest["name"] if cheapest else None),
    cheapest_cost_p50=(cheapest["cost"] if cheapest else None),
    adaptive_A5_note="adaptive A5 residual equals ~A2/maj3 floor; use A5f or A3c",
    suggested_line_width_scale=2,
    suggested_checksum=("line_xor_32B" if cheapest and cheapest["name"]=="A3c" else
                        ("line_parity_256b" if cheapest and cheapest["name"]=="A3p" else "none")),
    suggested_replicas=(7 if cheapest and cheapest["name"]=="A7f" else
                        (5 if cheapest and str(cheapest["name"]).startswith("A5") else
                         (3 if cheapest and "A3" in str(cheapest["name"]) else None))),
    double_read=bool(cheapest and cheapest["name"]=="A5fd"),
    erasure_v_assumed=0.5,
)

# figures
fig, ax = plt.subplots(figsize=(8.5, 4.8))
labels = ["A2", "A5", "A5f", "A5fd", "A7f", "A3p", "A3c"] + [f"A3e_v{v}" for v in (0.3, 0.5, 0.7, 0.9)]
res_vals = [max(g_burst[lab]["residual"], 1e-8) for lab in labels]
ax.bar(range(len(labels)), res_vals, color=["#888" if v > 1e-6 else "#2a7" for v in res_vals])
ax.axhline(1e-6, color="r", ls="--", label="target 1e-6")
ax.axhline(1e-4, color="orange", ls=":", label="019 ~1e-4")
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("residual byte error")
ax.set_title("Residual @ burst p=1e-3 (2 MB, seed=20261020)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_residual_formats.png", dpi=120)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ckeys = ["A2", "A5", "A5f", "A5fd", "A7f", "A3p", "A3c", "A3e_v0.5", "G_cell8_v0.9"]
ax.boxplot([np.log10(cost[k]) for k in ckeys], whis=(5, 95), showfliers=False)
ax.set_xticks(range(1, len(ckeys) + 1), [k.replace("_", "\n") for k in ckeys], fontsize=7)
ax.axhline(np.log10(MU018), color="r", ls="--", label=f"018 M-U {MU018:.2f}")
for i, k in enumerate(ckeys):
    res = G_res if k.startswith("G") else g_burst[k]["residual"]
    ax.scatter([i + 1], [np.log10(R["cost_py_per_MB"][k]["p50"])],
               c=("green" if res <= 1e-6 else "red"), zorder=5, s=40)
ax.set_ylabel("log10 person-years / MB"); ax.legend(fontsize=8)
ax.set_title("Cost (green=residual≤1e-6)"); fig.tight_layout()
fig.savefig(f"{HERE}/fig_cost_vs_residual.png", dpi=120)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
vs = [0.0, 0.3, 0.5, 0.7, 0.9]
ress = [g_burst["A2"]["residual"]] + [g_burst[f"A3e_v{v}"]["residual"] for v in (0.3, 0.5, 0.7, 0.9)]
ax.semilogy(vs, [max(r, 1e-8) for r in ress], "o-", label="A3e")
ax.axhline(max(g_burst["A5"]["residual"], 1e-8), color="C1", ls="--",
           label=f"A5={g_burst['A5']['residual']:.2e}")
ax.axhline(1e-6, color="r", ls=":")
ax.set_xlabel("visible burst fraction v"); ax.set_ylabel("residual")
ax.set_title("Erasure sensitivity"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_erasure_sensitivity.png", dpi=120)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({"hypotheses": H, "cost": R["cost_py_per_MB"], "recommendation": R["recommendation"],
                  "glyph_ref": g_ref, "runtime": R["runtime_seconds"]}, ensure_ascii=False, indent=1, default=float))
print("DONE", R["runtime_seconds"])
