"""研究020：手読み層の残留誤りを 1e-6 まで下げる最安の書式（open_problems 66・67）
019 の channel・費用モデルを再利用・拡張。seed 固定。numpy と標準ライブラリのみ。
事前登録：PREREGISTRATION.md（2026-10-16T08:39:25Z）
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
R = {"seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform(),
     "preregistration": "2026-10-16T08:39:25Z"}
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# ---- 019 から継承した定数 ----
M_SINGLE = math.sqrt(1e-4 * 3e-3)
M_DOUBLE_RES = M_SINGLE * 0.1
BURST = (1e-3, 100)  # (f, L)
r019 = json.load(open(f"{HERE}/../019-hand-readable-format/results.json"))
MU018 = r019["facts"]["m018_MU_median_py_per_MB"]
A2_019_cost = r019["cost_py_per_MB"]["A2_rep3_adaptive"]
R["inherited"] = dict(M_SINGLE=M_SINGLE, M_DOUBLE_RES=M_DOUBLE_RES, BURST=BURST,
                      MU018=MU018, A2_019_cost_p50=A2_019_cost["p50"],
                      A2_019_residual_p1e3_burst=None)
for d in r019["binary_formats"]["grid"]:
    if d["p_film"] == 1e-3 and d["burst"]:
        R["inherited"]["A2_019_residual_p1e3_burst"] = d["A2_rep3_adaptive"]

# 面積用（019）
PX_PER_DATA_BYTE_PIQL = (4096 * 2160) / 2e6
PIQL_REEL = 120e9
SIZE_MB = 10.0

# ---- チャネル（019 と同じ＋消失マスク） ----
def channel(nbits, p, burst=None, rng=rng):
    flip = rng.random(nbits) < p
    in_burst = np.zeros(nbits, dtype=bool)
    if burst:
        f, L = burst
        starts = np.nonzero(rng.random(nbits) < f / L)[0]
        lens = rng.geometric(1 / L, len(starts))
        for s, l in zip(starts, lens):
            e = min(nbits, s + int(l))
            seg = slice(s, e)
            in_burst[seg] = True
            flip[seg] ^= (rng.random(e - s) < 0.5)
    return flip.astype(np.uint8), in_burst

def apply_erasure(flip, in_burst, v, rng):
    """バースト区間の割合 v を目視で分かる消失にする。消失ビットは誤りにせずマスクする。"""
    eras = np.zeros(flip.shape, dtype=bool)
    if v <= 0 or not in_burst.any():
        return flip, eras
    cand = np.nonzero(in_burst)[0]
    mark = rng.random(len(cand)) < v
    eras[cand[mark]] = True
    flip = flip.copy()
    flip[eras] = 0  # 消失位置の反転は無効（読まない）
    return flip, eras

# ---- データ ----
NB = 2_000_000  # 2 MB（1e-6 を数える分解能用。019 は 1 MB）
data_bits = rng.integers(0, 2, NB * 8, dtype=np.uint8)
nb = data_bits.size

def byte_err(dec_bits):
    return float(np.mean((dec_bits.reshape(-1, 8) != data_bits.reshape(-1, 8)).any(axis=1)))

def make_copies(n_copies, p_film, burst, v_erase, m_read, rng):
    """各複製：(受信ビット, 消失マスク)。消失はバースト可視分。読み誤りは消失以外に載る。"""
    copies, erasures = [], []
    for _ in range(n_copies):
        flip, in_b = channel(nb, p_film, burst, rng=rng)
        flip, eras = apply_erasure(flip, in_b, v_erase, rng)
        read_err = (rng.random(nb) < m_read) & (~eras)
        rx = data_bits ^ flip ^ read_err.astype(np.uint8)
        copies.append(rx); erasures.append(eras)
    return copies, erasures

def decode_adaptive_majority(copies, erasures=None, n_initial=2):
    """最初 n_initial 本を読み、食い違い／消失があれば残りを読んで多数決。"""
    n = len(copies)
    known = np.zeros(nb, dtype=bool)
    votes = np.zeros(nb, dtype=np.int16)
    n_read = np.zeros(nb, dtype=np.int16)
    # 最初の n_initial
    for i in range(n_initial):
        if erasures is None:
            ok = np.ones(nb, dtype=bool)
        else:
            ok = ~erasures[i]
        votes += copies[i].astype(np.int16) * ok
        n_read += ok.astype(np.int16)
    # 最初2本が両方既知かつ一致 → 確定
    if erasures is None:
        agree = copies[0] == copies[1]
        need_more = ~agree
    else:
        both = (~erasures[0]) & (~erasures[1])
        agree = both & (copies[0] == copies[1])
        need_more = ~agree
    # 追加読み
    for i in range(n_initial, n):
        if erasures is None:
            ok = need_more
            votes += copies[i].astype(np.int16) * ok
            n_read += ok.astype(np.int16)
        else:
            ok = need_more & (~erasures[i])
            votes += copies[i].astype(np.int16) * ok
            n_read += ok.astype(np.int16)
    # まだ票が足りないビットは全複製を使う（上で need_more に追加済み）
    # 多数決：votes > n_read/2。n_read==0 は稀（全消失）→ 0 とする
    dec = (votes * 2 > n_read).astype(np.uint8)
    third_frac = float(need_more.mean())  # 「追加読みが必要だった割合」
    mean_reads = float(n_read.mean())
    return dec, third_frac, mean_reads

# ---- 偶奇検査和（32バイト＝256ビットごと） ----
LINE = 256  # データビット/行
def add_parity(bits):
    """bits → (data+parity) を行ごとに連結した1次元配列。"""
    n_lines = bits.size // LINE
    d = bits[:n_lines * LINE].reshape(n_lines, LINE)
    p = d.sum(axis=1) & 1
    return np.concatenate([d, p[:, None]], axis=1).reshape(-1)


def run_A3p(p_film, burst, m_read, rng):
    """3複製＋行偶奇。符号化：各256ビットに偶奇1。適応：2本読み→不一致行は偶奇判定→だめなら3本目。"""
    # 符号化
    n_lines = nb // LINE
    data_trim = data_bits[:n_lines * LINE]
    enc = add_parity(data_trim)  # length n_lines * 257
    ne = enc.size
    copies = []
    for _ in range(3):
        flip, _ = channel(ne, p_film, burst, rng=rng)
        read_err = rng.random(ne) < m_read
        copies.append(enc ^ flip ^ read_err.astype(np.uint8))
    c0, c1, c2 = [c.reshape(n_lines, LINE + 1) for c in copies]
    # 行ごとに判定
    out = np.empty((n_lines, LINE), dtype=np.uint8)
    used3 = 0
    parity_rescue = 0
    agree_lines = 0
    for i in range(n_lines):
        a, b = c0[i], c1[i]
        if np.array_equal(a, b):
            out[i] = a[:LINE]
            agree_lines += 1
            continue
        # 偶奇検査
        def ok(row):
            return (row[:LINE].sum() & 1) == row[LINE]
        a_ok, b_ok = ok(a), ok(b)
        if a_ok and not b_ok:
            out[i] = a[:LINE]; parity_rescue += 1
        elif b_ok and not a_ok:
            out[i] = b[:LINE]; parity_rescue += 1
        else:
            # 3本目を含めビット多数決
            used3 += 1
            stack = np.stack([a[:LINE], b[:LINE], c2[i, :LINE]])
            out[i] = (stack.sum(axis=0) >= 2).astype(np.uint8)
    dec = out.reshape(-1)
    # 末尾の端数ビット（LINE で割り切れない分）は生の3複製多数決
    rem = nb - n_lines * LINE
    if rem:
        # 端数は本シミュレーションでは 0（NB*8 は 256 で割れる）
        pass
    residual = float(np.mean((dec.reshape(-1, 8) != data_trim.reshape(-1, 8)).any(axis=1)))
    return dict(residual=residual, agree_line_frac=agree_lines / n_lines,
                parity_rescue_frac=parity_rescue / n_lines, third_line_frac=used3 / n_lines,
                overhead_bits=(LINE + 1) / LINE)

# ---- 文字（グリフ）書式 ----
# 1バイト＝1文字（ASCII 的）。セルは S×S 画素。セル内のバースト可視 → 文字消失。
def run_glyph(p_film, burst, v_erase, m_char, cell, n_rep, rng):
    """文字を S×S 画素で書く。フィルム誤りは画素。セル内に可視バーストがあれば消失。
    読み誤り m_char は文字置換。n_rep 複製の適応多数決（消失対応）。"""
    S = cell
    n_chars = NB  # 1バイト/文字
    # 各文字の「フィルム破損」：セル内のいずれかの画素がバーストに入り、かつ可視なら消失
    # 近似：セルを代表する画素列で channel し、セル単位に集約
    n_px = n_chars * S * S
    # メモリ節約：セルあたり「代表ビット」ではなく、バースト被覆をセル単位で近似
    # セルがバーストに覆われる確率 ≈ 1-(1-f/L)^{S*S} は重いので、
    # 画素チャネルを文字あたり1サンプルに縮約：有効 p_char ≈ 1-(1-p)^{S*S}
    # バーストは文字列上で (f_c, L_c) に縮約。L_c = max(1, L/(S*S)), f_c = f
    # より率直に：文字単位の独立誤り q = 1-(1-p_film)^{S*S} ＋ 文字バースト
    q = 1 - (1 - p_film) ** (S * S)
    # 文字バースト：元バーストがセルをまたぐ。平均長 L_char = max(1, L/(S))（行方向想定）
    L_char = max(1, BURST[1] // S)
    f_char = BURST[0]
    burst_c = (f_char, L_char) if burst else None

    # data_bits は NB*8。バイト化（ビット順は誤り率には影響しない）
    data_c = np.packbits(data_bits)  # length NB

    copies, erasures = [], []
    for _ in range(n_rep):
        flip, in_b = channel(n_chars, q, burst_c, rng=rng)
        flip, eras = apply_erasure(flip.astype(np.uint8), in_b, v_erase, rng)
        # 読み誤り：消失以外で文字を別の値に
        rx = data_c.copy()
        bad = (flip.astype(bool) | ((rng.random(n_chars) < m_char) & (~eras)))
        # 反転相当：ランダムな別文字へ
        idx = np.nonzero(bad)[0]
        if len(idx):
            rx[idx] = rng.integers(0, 256, len(idx), dtype=np.uint8)
            # 元と同じ偶然を避ける
            same = rx[idx] == data_c[idx]
            while same.any():
                rx[idx[same]] = rng.integers(0, 256, int(same.sum()), dtype=np.uint8)
                same = rx[idx] == data_c[idx]
        copies.append(rx); erasures.append(eras)

    # 適応多数決（文字単位）
    n_initial = 2
    votes = [{} for _ in range(n_chars)]  # 遅すぎる
    # ベクトル化：まず2本
    c0, c1 = copies[0], copies[1]
    e0, e1 = erasures[0], erasures[1]
    both = (~e0) & (~e1)
    agree = both & (c0 == c1)
    need = ~agree
    # 出力
    dec = c0.copy()
    dec[e0 & ~e1] = c1[e0 & ~e1]
    # 不一致・双方消失は追加複製で多数決
    if need.any() and n_rep >= 3:
        idxs = np.nonzero(need)[0]
        for i in idxs:
            ballots = []
            for r in range(n_rep):
                if not erasures[r][i]:
                    ballots.append(int(copies[r][i]))
            if not ballots:
                dec[i] = 0
            else:
                # 多数決
                vals, cts = np.unique(ballots, return_counts=True)
                dec[i] = vals[np.argmax(cts)]
    residual = float(np.mean(dec != data_c))
    return dict(residual=residual, need_more_frac=float(need.mean()), q_char=q,
                cell=S, n_rep=n_rep, px_per_char=S * S)

# ---- メイン格子：残留誤り ----
print("simulating residuals...", flush=True)
grid = []
conditions = [
    dict(label="iid_p1e-3", p=1e-3, burst=False),
    dict(label="burst_p1e-3", p=1e-3, burst=True),
    dict(label="burst_p1e-4", p=1e-4, burst=True),
    dict(label="burst_p1e-5", p=1e-5, burst=True),
]
for cond in conditions:
    p = cond["p"]
    burst = BURST if cond["burst"] else None
    row = dict(label=cond["label"], p_film=p, burst=cond["burst"])
    # A2: 3複製（019 基線）
    cps, ers = make_copies(3, p, burst, 0.0, M_SINGLE, rng)
    dec, frac, mreads = decode_adaptive_majority(cps, None, 2)
    row["A2"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mreads)
    # A5: 5複製
    cps, _ = make_copies(5, p, burst, 0.0, M_SINGLE, rng)
    dec, frac, mreads = decode_adaptive_majority(cps, None, 2)
    row["A5"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mreads)
    # A3e: 各 v
    for v in (0.3, 0.5, 0.7, 0.9):
        cps, ers = make_copies(3, p, burst, v, M_SINGLE, rng)
        dec, frac, mreads = decode_adaptive_majority(cps, ers, 2)
        row[f"A3e_v{v}"] = dict(residual=byte_err(dec), extra_frac=frac, mean_reads=mreads, v=v)
    # A3p
    row["A3p"] = run_A3p(p, burst, M_SINGLE, rng)
    grid.append(row)
    print(cond["label"], {k: (row[k]["residual"] if isinstance(row[k], dict) and "residual" in row[k] else row[k])
                          for k in row if k not in ("label", "p_film", "burst")}, flush=True)

R["residual_grid"] = grid

# グリフ感度（バースト p=1e-3）
print("glyph sensitivity...", flush=True)
glyph_grid = []
for cell in (8, 12, 16):
    for v in (0.5, 0.9):
        for m_char in (1e-4, 1e-3, 5e-3):
            for n_rep in (2, 3):
                g = run_glyph(1e-3, True, v, m_char, cell, n_rep, rng)
                g.update(v=v, m_char=m_char)
                glyph_grid.append(g)
R["glyph_grid"] = glyph_grid

# ---- 種依存性（H32 境界） ----
print("seed replicates...", flush=True)
reps = []
for sd in range(1, 6):
    rr = np.random.default_rng(SEED + 200 + sd)
    # 短い 1 MB で速度優先
    # 一時的に data を共有したまま channel だけ変える
    row = {"seed": SEED + 200 + sd}
    for name, n_rep, v in (("A2", 3, 0.0), ("A5", 5, 0.0), ("A3e_v0.5", 3, 0.5)):
        cps, ers = make_copies(n_rep, 1e-3, BURST, v, M_SINGLE, rr)
        dec, frac, mreads = decode_adaptive_majority(cps, ers if v > 0 else None, 2)
        row[name] = byte_err(dec)
    row["A3p"] = run_A3p(1e-3, BURST, M_SINGLE, rr)["residual"]
    reps.append(row)
    print(row, flush=True)
R["seed_replicates"] = reps

# ---- 費用モンテカルロ ----
print("costs...", flush=True)
def logu(lo, hi):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), N))
t_px2 = logu(0.05, 1.0)
t_bit = logu(0.2, 2.0)
# グリフ：校正読み 200 wpm × 5文字/語 = 1000文字/分 = 16.667/秒（紙）【事実：Bailey 要約の Ziefle】
# 顕微鏡減速 μ 【推測】対数一様 [2,20]
mu_microscope = logu(2.0, 20.0)
t_char_paper = 60.0 / (200.0 * 5.0)  # 秒/文字 = 0.06
t_char = t_char_paper * mu_microscope

def py(sec_per_byte):
    return sec_per_byte * 1e6 / SEC_PER_PY

# 追加読み率は burst_p1e-3 の実測を使う
g_burst = next(r for r in grid if r["label"] == "burst_p1e-3")
extra = {k: g_burst[k]["extra_frac"] if "extra_frac" in g_burst[k] else g_burst[k].get("third_line_frac", 0)
         for k in g_burst if k.startswith("A")}
# A3p の third_line_frac
extra["A3p"] = g_burst["A3p"]["third_line_frac"]
parity_ops = g_burst["A3p"]  # rescue 率など

cost = {}
# A2: 2本読み＋食い違い時3本目。019 と同じ式
p_dis_A2 = g_burst["A2"]["extra_frac"]
cost["A2"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25 + p_dis_A2 * 8 * t_px2)
# A5: 2本＋追加で最大3本（extra_frac のビットで3本追加読み）
p_dis_A5 = g_burst["A5"]["extra_frac"]
cost["A5"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25 + p_dis_A5 * 3 * 8 * t_px2)
# A3e: 2本＋消失/不一致で追加。消失ビットも「読まない」ので読み画素は減るが、
# 別複製へ飛ぶコストを extra_frac で近似。読みは平均 mean_reads
for v in (0.3, 0.5, 0.7, 0.9):
    k = f"A3e_v{v}"
    mr = g_burst[k]["mean_reads"]  # ビットあたり平均読み本数
    cost[k] = py(mr * 8 * t_px2 + 8 * t_bit * 0.3)  # 消失判定の目視を bit*0.3 で近似【推測】
# A3p: 2本読み＋行偶奇（行あたり数ビットの目視）＋一部3本目
# 偶奇：256ビットの1の数を目で数える ≈ 32バイトを走査して偶奇更新 → 256 * t_bit 相当は過大。
# 目視検査和は「行末の偶奇ビットと、行の偶奇をざっと確認」：行あたり 8 * t_bit 【推測】
# 2本とも確認する行は不一致行のみ、一致行は省略可。保守的に全行1本分＋不一致行でもう1本
n_lines_per_MB = (1e6 * 8) / LINE
sec_parity_per_byte = (1.0 / LINE) * 8 * t_bit   # データ1ビットあたり偶奇コストを行で割る
# より分かりやすく：1バイトあたり
frac3 = g_burst["A3p"]["third_line_frac"]
# 符号化オーバーヘッド 257/256
oh = (LINE + 1) / LINE
cost["A3p"] = py(oh * (2 * 8 * t_px2 + 8 * t_bit * 0.25  # 2本＋照合
                       + 2 * (8 * t_bit / 32)             # 両複製の行偶奇ざっと（32バイトに1回、8操作）【推測】
                       + frac3 * 8 * t_px2))              # 3本目

# グリフ費用：3複製適応、文字/バイト
# 最良想定：cell=8, v=0.9, m_char=1e-4, n_rep=3
g_ref = next(g for g in glyph_grid if g["cell"] == 8 and g["v"] == 0.9 and g["m_char"] == 1e-4 and g["n_rep"] == 3)
mr_g = 2 + g_ref["need_more_frac"]  # 平均読み本数の近似
cost["G_cell8_v0.9"] = py(mr_g * t_char)  # 1文字=1バイト
# 感度：μ の代わりに既に t_char に入っている

def qs(x):
    return {k: float(np.percentile(x, v)) for k, v in (("p5", 5), ("p50", 50), ("p95", 95))}
R["cost_py_per_MB"] = {k: qs(v) for k, v in cost.items()}
R["cost_extras"] = {k: float(extra.get(k, g_burst.get(k, {}).get("extra_frac", float("nan"))
                                       if isinstance(g_burst.get(k), dict) else float("nan")))
                    for k in list(cost.keys()) if k in g_burst or k in extra}
R["cost_extras"] = {
    "A2_extra": float(g_burst["A2"]["extra_frac"]),
    "A5_extra": float(g_burst["A5"]["extra_frac"]),
    "A3e_v0.5_extra": float(g_burst["A3e_v0.5"]["extra_frac"]),
    "A3e_v0.5_mean_reads": float(g_burst["A3e_v0.5"]["mean_reads"]),
    "A3p_third_line": float(g_burst["A3p"]["third_line_frac"]),
    "A3p_parity_rescue": float(g_burst["A3p"]["parity_rescue_frac"]),
    "glyph_need_more": float(g_ref["need_more_frac"]),
    "t_char_paper_sec": t_char_paper,
}

# ---- 面積 ----
area = {}
px = {"A2": 24, "A5": 40, "A3p": 24 * (LINE + 1) / LINE, "A3e": 24,
      "G_cell8": 3 * 64, "G_cell16": 3 * 256}  # 3複製のグリフ
for k, v in px.items():
    for s in (1, 2, 5):
        eq = SIZE_MB * 1e6 * v * s * s / PX_PER_DATA_BYTE_PIQL
        area[f"{k}_s{s}"] = dict(px_per_data_byte=v, linear_scale=s,
                                 frac_of_reel=eq / PIQL_REEL, piql_equiv_MB=eq / 1e6)
R["area"] = area

# ---- 仮説判定 ----
def meets(res): return res <= 1e-6
H = {}
A5_res = g_burst["A5"]["residual"]
H["H32a"] = dict(A5_residual=A5_res, threshold=1e-6, supported=meets(A5_res),
                 seed_pass_frac=float(np.mean([r["A5"] <= 1e-6 for r in reps])))
A3e_res = g_burst["A3e_v0.5"]["residual"]
A3e_cost = R["cost_py_per_MB"]["A3e_v0.5"]["p50"]
A5_cost = R["cost_py_per_MB"]["A5"]["p50"]
H["H32b"] = dict(**{"A3e_v0.5_residual": A3e_res}, A3e_cost=A3e_cost, A5_cost=A5_cost,
                 residual_ok=meets(A3e_res), cost_ok=A3e_cost <= A5_cost,
                 supported=meets(A3e_res) and (A3e_cost <= A5_cost),
                 seed_pass_frac=float(np.mean([r["A3e_v0.5"] <= 1e-6 for r in reps])))
A3p_res = g_burst["A3p"]["residual"]
A3p_cost = R["cost_py_per_MB"]["A3p"]["p50"]
H["H32c"] = dict(residual=A3p_res, cost=A3p_cost, A5_cost=A5_cost,
                 residual_ok=meets(A3p_res), cost_ok=A3p_cost <= A5_cost,
                 supported=meets(A3p_res) and (A3p_cost <= A5_cost),
                 seed_pass_frac=float(np.mean([r["A3p"] <= 1e-6 for r in reps])))
# H32d: 残留を満たすものの最安が A3e_v0.5
candidates = []
for name, res_key in (("A5", A5_res), ("A3p", A3p_res), ("A3e_v0.5", A3e_res)):
    c = R["cost_py_per_MB"][name]["p50"]
    candidates.append(dict(name=name, residual=res_key, cost=c, meets=meets(res_key)))
# グリフ：参照条件の残留
G_res = g_ref["residual"]
G_cost = R["cost_py_per_MB"]["G_cell8_v0.9"]["p50"]
candidates.append(dict(name="G_cell8_v0.9", residual=G_res, cost=G_cost, meets=meets(G_res)))
ok_c = [c for c in candidates if c["meets"]]
cheapest = min(ok_c, key=lambda c: c["cost"]) if ok_c else None
H["H32d"] = dict(candidates=candidates, cheapest=(cheapest["name"] if cheapest else None),
                 supported=(cheapest is not None and cheapest["name"] == "A3e_v0.5"))
# H32e: グリフは人年では安いが面積では高い
best_dot = min((c for c in candidates if c["name"] != "G_cell8_v0.9" and c["meets"]),
               key=lambda c: c["cost"], default=None)
area_G = area["G_cell8_s2"]["frac_of_reel"]
area_dot = area[f"{best_dot['name'] if best_dot and best_dot['name']!='A3e_v0.5' else 'A3e'}_s2"]["frac_of_reel"] if best_dot else None
# A3e key in area is "A3e"
if best_dot:
    ak = "A3e" if best_dot["name"].startswith("A3e") else best_dot["name"]
    if ak not in ("A2", "A5", "A3p", "A3e"):
        ak = "A3e"
    area_dot = area[f"{ak}_s2"]["frac_of_reel"]
H["H32e"] = dict(G_cost=G_cost, best_dot_cost=(best_dot["cost"] if best_dot else None),
                 best_dot_name=(best_dot["name"] if best_dot else None),
                 G_cheaper_in_PY=(G_cost < best_dot["cost"]) if best_dot else None,
                 area_G_s2=area_G, area_dot_s2=area_dot,
                 G_more_area=(area_G > area_dot) if area_dot is not None else None,
                 supported=((G_cost < best_dot["cost"]) if best_dot else False)
                           and ((area_G > area_dot) if area_dot is not None else False),
                 note="読取速度は紙の校正200語/分を顕微鏡減速μ∈[2,20]で割った【推測】。Rosettaの実測速度は未確認。")
R["hypotheses"] = H

# 推奨一行用
R["recommendation"] = dict(
    meeting_1e6=[c["name"] for c in candidates if c["meets"]],
    cheapest_meeting=(cheapest["name"] if cheapest else None),
    cheapest_cost_p50=(cheapest["cost"] if cheapest else None),
    suggested_line_width_scale=2,
    suggested_checksum="none" if (cheapest and cheapest["name"].startswith("A3e")) else
                       ("line_parity_256b" if (cheapest and cheapest["name"] == "A3p") else "n/a"),
    suggested_replicas=3 if (cheapest and "A3" in cheapest["name"]) else (5 if cheapest and cheapest["name"] == "A5" else None),
    erasure_v_assumed=0.5,
)

# ---- 図 ----
fig, ax = plt.subplots(figsize=(8.5, 4.8))
labels = ["A2", "A5", "A3p"] + [f"A3e_v{v}" for v in (0.3, 0.5, 0.7, 0.9)]
res_vals = []
for lab in labels:
    if lab == "A3p":
        res_vals.append(max(g_burst["A3p"]["residual"], 1e-8))
    else:
        res_vals.append(max(g_burst[lab]["residual"], 1e-8))
ax.bar(range(len(labels)), res_vals, color=["#888" if v > 1e-6 else "#2a7" for v in res_vals])
ax.axhline(1e-6, color="r", ls="--", label="target 1e-6")
ax.axhline(1e-4, color="orange", ls=":", label="019 boundary 1e-4")
ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
ax.set_yscale("log"); ax.set_ylabel("residual byte error (burst p=1e-3)")
ax.set_title("Hand-readable residual error (2 MB sim, seed=20261020)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{HERE}/fig_residual_formats.png", dpi=120)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ckeys = ["A2", "A5", "A3p", "A3e_v0.5", "A3e_v0.9", "G_cell8_v0.9"]
ax.boxplot([np.log10(cost[k]) for k in ckeys], whis=(5, 95), showfliers=False)
ax.set_xticks(range(1, len(ckeys) + 1), [k.replace("_", "\n") for k in ckeys], fontsize=7)
ax.axhline(np.log10(MU018), color="r", ls="--", label=f"018 M-U {MU018:.2f}")
# 残留達成の印
for i, k in enumerate(ckeys):
    res = (G_res if k.startswith("G") else
           g_burst["A3p"]["residual"] if k == "A3p" else
           g_burst[k]["residual"])
    ax.scatter([i + 1], [np.log10(R["cost_py_per_MB"][k]["p50"])],
               c=("green" if res <= 1e-6 else "red"), zorder=5, s=40)
ax.set_ylabel("log10 person-years / MB"); ax.legend(fontsize=8)
ax.set_title("Cost by format (green marker: residual≤1e-6 at burst p=1e-3)")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_cost_vs_residual.png", dpi=120)

# 感度：v と残留
fig, ax = plt.subplots(figsize=(6.5, 4.2))
vs = [0.0, 0.3, 0.5, 0.7, 0.9]
# A2 を v=0 として
ress = [g_burst["A2"]["residual"]] + [g_burst[f"A3e_v{v}"]["residual"] for v in (0.3, 0.5, 0.7, 0.9)]
ax.semilogy(vs, [max(r, 1e-8) for r in ress], "o-", label="A3e (3-rep + erasure)")
ax.axhline(g_burst["A5"]["residual"] or 1e-8, color="C1", ls="--", label=f"A5 residual={g_burst['A5']['residual']:.2e}")
ax.axhline(1e-6, color="r", ls=":")
ax.set_xlabel("visible fraction v of burst pixels (erasure)"); ax.set_ylabel("residual byte error")
ax.set_title("Erasure model sensitivity (burst p=1e-3)")
ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(f"{HERE}/fig_erasure_sensitivity.png", dpi=120)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
summary = dict(hypotheses=H, cost_py_per_MB=R["cost_py_per_MB"], recommendation=R["recommendation"],
               residual_burst_p1e3={k: (g_burst[k]["residual"] if "residual" in g_burst[k] else g_burst[k])
                                   for k in g_burst if k not in ("label", "p_film", "burst")},
               seed_replicates=reps, runtime=R["runtime_seconds"])
print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
print("runtime", R["runtime_seconds"])
