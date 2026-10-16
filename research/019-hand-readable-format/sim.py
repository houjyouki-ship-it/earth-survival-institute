"""研究019：手で読めるアーカイブ書式の最適化（open_problems 62・63）
Piql の公開ソース（github.com/piql/unboxing：4kv6 設定、cipher.c、reedsolomon.c）にある実際の書式
（LFSR 攪拌＋ブロック・インターリーブ＋連接 RS(255,207)/RS(207,197)＋2ビット/画素）を、
顕微鏡＋人手で読む費用と、手読み向けに設計した書式（1ビット/画素の生データ、離れた3複製、ハミング(7,4)、
インターリーブ付きハミング）の費用・残留誤り・フィルム面積を比べる。
実際に符号化・誤り注入・復号して数える。seed 固定。numpy と標準ライブラリのみ。事前登録：PREREGISTRATION.md
"""
import json, time, sys, platform, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
SEED = 20261019
rng = np.random.default_rng(SEED)
N = 100_000
SEC_PER_PY = 2000 * 3600   # 018 と同じ【推測】
R = {"seed": SEED, "N": N, "python": sys.version.split()[0], "platform": platform.platform()}
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."

# ---------------- 事実入力（sources/） ----------------
FACT = dict(
    piql_4kv6_width=4096, piql_4kv6_height=2160,        # cfg_4kv6.h FrameFormat
    piql_maxLevelsPerSymbol=4,                           # cfg_4kv6.h（2ビット/画素）
    piql_4kv6_rs_inner=(255, 207), piql_4kv6_rs_outer=(207, 197),   # byteParityNumber 48/10, messageSize 207/197
    piql_4kv6_interleave_inner=248, piql_4kv6_interleave_outer=207,
    piql_4kv11_rs_inner=(255, 231), piql_4kv11_rs_outer=(231, 225),
    piql_lfsr_taps=0xD0000001, piql_cipher_block=64,    # cipher.c
    piql_rs_prim_poly=285,                               # reedsolomon.c RS_PRIM_POLY_285
    piql_oversampling=2.7,                               # boxing README
    piql_bytes_per_frame_claim=2e6,                      # piql.com writer（018）
    piql_reel_bytes=120e9,                               # piql.com film（018）
    piql_lp_per_mm=1180,                                 # piql.com film（018）
    abbe_d="lambda/(2NA)",                               # Wikipedia Diffraction-limited system
    m018_MU_median_py_per_MB=None,
)
r018 = json.load(open(f"{HERE}/../018-physical-min-reader/results.json"))
def find_key(d, key):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key: return v
            x = find_key(v, key)
            if x is not None: return x
    return None
mu = r018["M_U"]
R["m018_M_U_keys"] = list(mu.keys())
# 018 の M-U の 1MB あたり人年の中央値（report.md では 0.37）
MU018 = None
for k in ("PY_per_MB", "py_per_MB"):
    if k in mu: MU018 = mu[k]["p50"] if isinstance(mu[k], dict) else mu[k]
if MU018 is None:
    MU018 = 0.37
FACT["m018_MU_median_py_per_MB"] = MU018
R["facts"] = FACT

# ---------------- 1. Piql の LFSR 攪拌（cipher.c の移植） ----------------
TAPS = 0xD0000001
def piql_keystream(nbytes, key=1):
    lfsr = key; out = bytearray(nbytes); lsb_ones = 0
    for i in range(nbytes):
        cb = 0
        for _ in range(8):
            b = lfsr & 1; lsb_ones += b
            lfsr = ((lfsr >> 1) ^ (TAPS if b else 0)) & 0xFFFFFFFF
            cb = ((cb << 1) | ((lfsr >> 31) & 1)) & 0xFF
        out[i] = cb
    return bytes(out), lsb_ones / (8 * nbytes)

def piql_block_ok(enc):
    # cipher.c：64バイト塊ごとに1の数が 8 以上かつ (64-1)*8 以下でなければ鍵を変える
    for i in range(0, len(enc), 64):
        ones = sum(bin(x).count("1") for x in enc[i:i + 64])
        if ones < 8 or ones > 63 * 8: return False
    return True

text = open(f"{HERE}/../012-archive-reading-ladder/sources/github_archive_GUIDE.md", "rb").read()
ks, p_lsb = piql_keystream(len(text))
enc = bytes(a ^ b for a, b in zip(text, ks))
dec = bytes(a ^ b for a, b in zip(enc, ks))
def printable_frac(b):
    return sum(1 for x in b if 32 <= x < 127 or x in (9, 10, 13)) / len(b)
zeros = bytes(4096)
C_KS = 1 + 4 * p_lsb          # 鍵ストリーム1ビットの手作業：lsb を見る1＋lsb=1 なら4ビット反転
R["cipher"] = dict(n_bytes=len(text), roundtrip_ok=(dec == text), printable_plain=printable_frac(text),
                   printable_whitened=printable_frac(enc), lsb_one_fraction=p_lsb,
                   bit_steps_per_keystream_bit=C_KS, key1_ok_on_text=piql_block_ok(enc),
                   key1_ok_on_zeros=piql_block_ok(bytes(a ^ b for a, b in zip(zeros, ks[:4096]))),
                   first8_keystream_hex=ks[:8].hex())

# ---------------- 2. Piql フレーム容量の整合性 ----------------
raw_frame = FACT["piql_4kv6_width"] * FACT["piql_4kv6_height"] * 2 / 8
R["frame_capacity"] = dict(raw_bytes_2bpp=raw_frame,
                           data_bytes_4kv6=raw_frame * 197 / 255, data_bytes_4kv11=raw_frame * 225 / 255,
                           claim=FACT["piql_bytes_per_frame_claim"],
                           note="境界・同期点・メタデータを引く前の上限")
PX_PER_DATA_BYTE_PIQL = (FACT["piql_4kv6_width"] * FACT["piql_4kv6_height"]) / FACT["piql_bytes_per_frame_claim"]

# ---------------- 3. GF(256) Reed–Solomon（原始多項式 285） ----------------
EXP = [0] * 512; LOG = [0] * 256; x = 1
for i in range(255):
    EXP[i] = x; LOG[x] = i; x <<= 1
    if x & 0x100: x ^= 285
for i in range(255, 512): EXP[i] = EXP[i - 255]
OPS = [0]
def gmul(a, b):
    OPS[0] += 1
    return 0 if a == 0 or b == 0 else EXP[LOG[a] + LOG[b]]
def gpow(a, p): return EXP[(LOG[a] * p) % 255]
def ginv(a): return EXP[255 - LOG[a]]
def gdiv(a, b):
    OPS[0] += 1
    return 0 if a == 0 else EXP[(LOG[a] + 255 - LOG[b]) % 255]
def pscale(p, s): return [gmul(c, s) for c in p]
def padd(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)): r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)): r[i + len(r) - len(q)] ^= q[i]
    return r
def pmul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)): r[i + j] ^= gmul(p[i], q[j])
    return r
def peval(p, xv):
    y = p[0]
    for i in range(1, len(p)): y = gmul(y, xv) ^ p[i]
    return y
def pdiv(dividend, divisor):
    out = list(dividend)
    for i in range(len(dividend) - (len(divisor) - 1)):
        c = out[i]
        if c:
            for j in range(1, len(divisor)):
                if divisor[j]: out[i + j] ^= gmul(divisor[j], c)
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]
GEN = {}
def gen_poly(nsym):
    if nsym not in GEN:
        g = [1]
        for i in range(nsym): g = pmul(g, [1, gpow(2, i)])
        GEN[nsym] = g
    return GEN[nsym]
def rs_encode(msg, nsym):
    g = gen_poly(nsym); out = list(msg) + [0] * nsym
    for i in range(len(msg)):
        c = out[i]
        if c:
            for j in range(1, len(g)): out[i + j] ^= gmul(g[j], c)
    return list(msg) + out[len(msg):]
class RSFail(Exception): pass
def syndromes(msg, nsym): return [0] + [peval(msg, gpow(2, i)) for i in range(nsym)]
def err_locator(synd, nsym):
    loc = [1]; old = [1]; shift = len(synd) - nsym
    for i in range(nsym):
        K = i + shift; d = synd[K]
        for j in range(1, len(loc)): d ^= gmul(loc[-(j + 1)], synd[K - j])
        old = old + [0]
        if d:
            if len(old) > len(loc):
                new = pscale(old, d); old = pscale(loc, ginv(d)); loc = new
            loc = padd(loc, pscale(old, d))
    while len(loc) and loc[0] == 0: del loc[0]
    if (len(loc) - 1) * 2 > nsym: raise RSFail()
    return loc
def find_errors(loc_rev, n):
    errs = len(loc_rev) - 1; pos = []
    for i in range(n):
        if peval(loc_rev, gpow(2, i)) == 0: pos.append(n - 1 - i)
    if len(pos) != errs: raise RSFail()
    return pos
def correct_errata(msg, synd, err_pos):
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    e_loc = [1]
    for i in coef_pos: e_loc = pmul(e_loc, padd([1], [gpow(2, i), 0]))
    _, rem = pdiv(pmul(synd[::-1], e_loc), [1] + [0] * len(e_loc))
    err_eval = rem[::-1]
    X = [gpow(2, -(255 - cp)) for cp in coef_pos]
    E = [0] * len(msg)
    for i, Xi in enumerate(X):
        Xi_inv = ginv(Xi); prime = 1
        for j in range(len(X)):
            if j != i: prime = gmul(prime, 1 ^ gmul(Xi_inv, X[j]))
        y = gmul(Xi, peval(err_eval[::-1], Xi_inv))
        E[err_pos[i]] = gdiv(y, prime)
    return padd(msg, E)
def rs_decode(msg, nsym):
    """戻り値：(データ, 状態)。状態は 'clean'/'corrected'/'fail'（検出された失敗）"""
    synd = syndromes(msg, nsym)
    if max(synd) == 0: return list(msg[:-nsym]), "clean"
    try:
        loc = err_locator(synd, nsym)
        pos = find_errors(loc[::-1], len(msg))
        out = correct_errata(list(msg), synd, pos)
        if max(syndromes(out, nsym)) > 0: raise RSFail()
        return out[:-nsym], "corrected"
    except RSFail:
        return list(msg[:-nsym]), "fail"

# RS の検証：RS(255,207) に独立バイト誤り q を注入
def binom_tail(n, q, t):  # P(Binom(n,q) > t)
    return float(1 - sum(math.comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(t + 1)))
rs_val = []
NCW = 300
for p_px in (1e-3, 1e-2, 2e-2, 2.5e-2, 3e-2):
    q = 1 - (1 - p_px) ** 4          # 2ビット/画素：1バイト=4画素
    st = {"clean": 0, "corrected": 0, "fail": 0}; wrong = 0; ops_by_e = []; fail_e = []
    for _ in range(NCW):
        msg = list(rng.integers(0, 256, 207))
        cw = rs_encode(msg, 48)
        mask = rng.random(255) < q
        rx = list(cw)
        for i in np.nonzero(mask)[0]: rx[i] ^= int(rng.integers(1, 256))
        OPS[0] = 0
        out, s = rs_decode(rx, 48)
        ops_by_e.append((int(mask.sum()), OPS[0]))
        st[s] += 1
        if s == "fail": fail_e.append(int(mask.sum()))
        if s != "fail" and out != msg: wrong += 1   # 誤訂正（検出されない誤り）
    e = np.array(ops_by_e)
    rs_val.append(dict(p_px=p_px, q_byte=q, **st, miscorrected=wrong,
                       fail_rate_mc=(st["fail"] + wrong) / NCW, fail_rate_analytic=binom_tail(255, q, 24),
                       mean_errors=float(e[:, 0].mean()), mean_gf_ops=float(e[:, 1].mean()),
                       errors_in_failed=sorted(fail_e)[:20]))
R["rs_validation"] = rs_val
# 復号の GF 演算数を誤り数 e に回帰（ops = a + b e + c e^2）
allops = []
for p_px in (1e-3, 1e-2):
    pass
E_ = []; O_ = []
for _ in range(150):
    e = int(rng.integers(0, 25)); msg = list(rng.integers(0, 256, 207)); cw = rs_encode(msg, 48)
    rx = list(cw)
    for i in rng.choice(255, e, replace=False): rx[i] ^= int(rng.integers(1, 256))
    OPS[0] = 0; rs_decode(rx, 48); E_.append(e); O_.append(OPS[0])
E_ = np.array(E_); O_ = np.array(O_)
coef = np.polyfit(E_[E_ >= 1], O_[E_ >= 1], 2)
OPS0 = float(O_[E_ == 0].mean()) if (E_ == 0).any() else 48 * 254.0
R["rs_ops_fit_inner"] = dict(c2=float(coef[0]), c1=float(coef[1]), c0=float(coef[2]),
                             ops_e0_measured=OPS0, ops_e10=float(np.polyval(coef, 10)),
                             note="1符号語（255バイト、48パリティ）の自前実装での GF 乗除算の回数。検査（シンドローム）だけで 48×255")
def inner_ops(e): return np.polyval(coef, max(e, 1.0))   # 誤りのある符号語（e≥1）の復号演算数

# ---------------- 4. 2値チャネルでの手読み書式（実際に符号化・誤り注入・復号） ----------------
NB = 1_000_000   # データ 1 MB
data_bits = rng.integers(0, 2, NB * 8, dtype=np.uint8)
def channel(nbits, p, burst=None, rng=rng):
    """独立反転 p ＋（任意で）バースト：開始率 f/L、長さは平均 L の幾何分布、区間内は確率0.5で反転"""
    flip = rng.random(nbits) < p
    if burst:
        f, L = burst
        starts = np.nonzero(rng.random(nbits) < f / L)[0]
        lens = rng.geometric(1 / L, len(starts))
        for s, l in zip(starts, lens):
            seg = slice(s, min(nbits, s + l)); flip[seg] ^= (rng.random(min(nbits, s + l) - s) < 0.5)
    return flip.astype(np.uint8)
def byte_err(dec_bits):
    return float(np.mean((dec_bits.reshape(-1, 8) != data_bits.reshape(-1, 8)).any(axis=1)))
# ハミング(7,4)：符号語 [p1 p2 d1 p3 d2 d3 d4]
G_pos = dict(p1=(0, 1, 3), p2=(0, 2, 3), p3=(1, 2, 3))
def ham_encode(bits):
    d = bits.reshape(-1, 4)
    p1 = d[:, 0] ^ d[:, 1] ^ d[:, 3]; p2 = d[:, 0] ^ d[:, 2] ^ d[:, 3]; p3 = d[:, 1] ^ d[:, 2] ^ d[:, 3]
    return np.stack([p1, p2, d[:, 0], p3, d[:, 1], d[:, 2], d[:, 3]], axis=1)   # (n,7)
def ham_decode(cw):
    c = cw.copy()
    s1 = c[:, 0] ^ c[:, 2] ^ c[:, 4] ^ c[:, 6]; s2 = c[:, 1] ^ c[:, 2] ^ c[:, 5] ^ c[:, 6]; s3 = c[:, 3] ^ c[:, 4] ^ c[:, 5] ^ c[:, 6]
    syn = s1 + 2 * s2 + 4 * s3
    idx = np.nonzero(syn)[0]; c[idx, syn[idx] - 1] ^= 1
    return c[:, [2, 4, 5, 6]].reshape(-1)
def run_formats(p_film, m_single, m_double_res, burst):
    out = {}
    nb = data_bits.size
    # A1：生データ1ビット/画素、二重読み（読み誤りは残留 m_double_res）
    rx = data_bits ^ channel(nb, p_film, burst) ^ (rng.random(nb) < m_double_res).astype(np.uint8)
    out["A1_raw_double"] = byte_err(rx)
    # A2：離れた3複製。複製1・2を1回ずつ読み、食い違えば3本目を読んで多数決
    c = [data_bits ^ channel(nb, p_film, burst) ^ (rng.random(nb) < m_single).astype(np.uint8) for _ in range(3)]
    agree = c[0] == c[1]
    dec = np.where(agree, c[0], (c[0].astype(int) + c[1] + c[2] >= 2).astype(np.uint8))
    out["A2_rep3_adaptive"] = byte_err(dec); out["A2_third_read_frac"] = float(1 - agree.mean())
    # A3：ハミング(7,4)、ラスタ順にそのまま並べる、1回読み
    cw = ham_encode(data_bits); flat = cw.reshape(-1)
    rxf = flat ^ channel(flat.size, p_film, burst) ^ (rng.random(flat.size) < m_single).astype(np.uint8)
    out["A3_hamming_single"] = byte_err(ham_decode(rxf.reshape(-1, 7)))
    # A3i：ハミング、符号語の7ビットを7つの離れた帯に置く（帯は全長の1/7ずつ）
    flat_i = cw.T.reshape(-1)
    rxi = flat_i ^ channel(flat_i.size, p_film, burst) ^ (rng.random(flat_i.size) < m_single).astype(np.uint8)
    out["A3i_hamming_interleaved_single"] = byte_err(ham_decode(rxi.reshape(7, -1).T))
    return out
M_SINGLE = math.sqrt(1e-4 * 3e-3)     # 1回読みの読み誤り：[1e-4, 3e-3] の幾何中心【推測】
M_DOUBLE_RES = M_SINGLE * 0.1         # 二重読み後の残留（照合で9割を捕まえる）【推測】
BURST = (1e-3, 100)
grid = []
for p in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2):
    grid.append(dict(p_film=p, burst=False, **run_formats(p, M_SINGLE, M_DOUBLE_RES, None)))
grid.append(dict(p_film=1e-5, burst=True, **run_formats(1e-5, M_SINGLE, M_DOUBLE_RES, BURST)))
grid.append(dict(p_film=1e-3, burst=True, **run_formats(1e-3, M_SINGLE, M_DOUBLE_RES, BURST)))
R["binary_formats"] = dict(m_single=M_SINGLE, m_double_res=M_DOUBLE_RES, burst_f_L=BURST, grid=grid)

# Piql（2ビット/画素、内外 RS、インターリーブ）の残留誤り：独立誤りは解析式（上で MC と照合）、バーストは並べ替えを再現して数える
def piql_residual(p_px, m_res):
    q = 1 - (1 - p_px - m_res) ** 4
    P_in = binom_tail(255, q, 24)
    P_out = binom_tail(207, P_in, 5) if P_in > 0 else 0.0
    return dict(q_byte=q, skip_rs=q, inner_fail=P_in, full_rs_codeword_fail=P_out)
piql_grid = [dict(p_film=p, **piql_residual(p, M_DOUBLE_RES * 3)) for p in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2)]
# バースト：内側インターリーブ距離248 → 符号語 k の j 番目のバイトは位置 j*248+k
blk = 248 * 255; nblocks = 400; maxerr = 0; bad = 0
for _ in range(nblocks):
    fl = channel(blk * 4, 1e-5, BURST).reshape(-1, 4).any(axis=1)
    per_cw = fl.reshape(255, 248).sum(axis=0)
    maxerr = max(maxerr, int(per_cw.max())); bad += int((per_cw > 24).sum())
R["piql_residual"] = dict(iid=piql_grid, burst=dict(blocks=nblocks, max_errors_per_inner_codeword=maxerr,
                                                   inner_codewords_over_t=bad))

# ---------------- 5. 手作業費用のモンテカルロ（1 MB のデータあたり人年） ----------------
def logu(lo, hi): return np.exp(rng.uniform(np.log(lo), np.log(hi), N))
t_px2 = logu(0.05, 1.0)       # 2値画素1つを読む秒（018：1〜20画素/秒）【推測】
g4 = logu(1.5, 4.0)           # 4階調画素の読取の倍率【推測】
t_bit = logu(0.2, 2.0)        # 1ビットを見て偶奇・XOR を更新する秒【推測】
t_gf = logu(15, 180)          # GF(256) 乗算1回（対数表3引き＋加算）秒【推測：018 の手計算1演算 5〜60秒 ×3】
t_idx = logu(2, 30)           # インターリーブされたバイト1つの位置を求めて台帳に写す秒（1段あたり）【推測】
def py(sec_per_byte): return sec_per_byte * 1e6 / SEC_PER_PY
SIZE_MB = 10.0
amort = min(1.0, raw_frame / 1e6 / SIZE_MB)   # 鍵ストリームはフレームごとに同じ（key=1）なので1フレーム分で済む
cost = {}
cost["A1_raw_double"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25)
p_dis = 2 * (1e-3 + M_SINGLE)
cost["A2_rep3_adaptive"] = py(2 * 8 * t_px2 + 8 * t_bit * 0.25 + p_dis * 8 * t_px2)
cost["A3_hamming_single"] = py(14 * t_px2 + 2 * 12 * t_bit)
read_p = 2 * 4 * g4 * t_px2
dewhite = 8 * t_bit + 8 * C_KS * t_bit * amort
cost["P_skip"] = py(read_p + dewhite + 2 * t_idx)
syn_full = (48 * 255 + 10 * 207) / 197
def piql_full(p_px):
    q = 1 - (1 - p_px - M_DOUBLE_RES * 3) ** 4
    flag = 1 - (1 - q) ** 255
    e_cond = 255 * q / flag
    corr_in = max(0.0, float(inner_ops(e_cond)) - OPS0) * flag / 197
    return py((255 / 197) * (read_p + dewhite + 2 * t_idx) + (syn_full + corr_in) * t_gf)
def piql_lite(p_px):
    q = 1 - (1 - p_px - M_DOUBLE_RES * 3) ** 4
    flag_in = 1 - (1 - q) ** 255
    e_mean = 255 * q / max(flag_in, 1e-300)
    det = (2 * 255 + 2 * 207) / 197
    full_in = float(inner_ops(e_mean)) * flag_in / 197
    return py((255 / 197) * (read_p + dewhite + 2 * t_idx) + (det + full_in) * t_gf), flag_in
for p in (1e-5, 1e-4, 1e-3):
    cost[f"P_full_p{p:g}"] = piql_full(p)
    cost[f"P_lite_p{p:g}"], _ = piql_lite(p)
def qs(x): return {k: float(np.percentile(x, v)) for k, v in (("p5", 5), ("p50", 50), ("p95", 95))}
R["cost_py_per_MB"] = {k: qs(v) for k, v in cost.items()}
R["cost_components_median_sec_per_byte"] = dict(
    read_piql_double=float(np.median(read_p)), dewhiten=float(np.median(dewhite)), index_2levels=float(np.median(2 * t_idx)),
    rs_syndromes_full=float(np.median(syn_full * t_gf)), syn_full_gf_per_data_byte=syn_full)
rs_ratio = {}
for p in (1e-5, 1e-3):
    q = 1 - (1 - p - M_DOUBLE_RES * 3) ** 4
    flag = 1 - (1 - q) ** 255
    e_mean = 255 * q / flag
    rs_full_ops = syn_full + max(0.0, float(inner_ops(e_mean)) - OPS0) * flag / 197
    rs_lite_ops = (2 * 255 + 2 * 207) / 197 + float(inner_ops(e_mean)) * flag / 197
    rs_ratio[f"p{p:g}"] = dict(flag_inner=flag, full_gf_per_byte=rs_full_ops, lite_gf_per_byte=rs_lite_ops,
                               lite_over_full=rs_lite_ops / rs_full_ops)
R["rs_detect_then_correct"] = rs_ratio

# ---------------- 6. フィルム面積 ----------------
px_per_byte = dict(A1_raw_double=8, A2_rep3_adaptive=24, A3_hamming_single=14, A3i_hamming_interleaved_single=14)
area = {}
for k, v in px_per_byte.items():
    for s in (1, 2, 5):
        eq_bytes = SIZE_MB * 1e6 * v * s * s / PX_PER_DATA_BYTE_PIQL
        area[f"{k}_s{s}"] = dict(px_per_byte=v, linear_scale=s, piql_equiv_MB=eq_bytes / 1e6,
                                 frac_of_reel=eq_bytes / FACT["piql_reel_bytes"])
feat = 1 / (2 * FACT["piql_lp_per_mm"]) * 1000   # µm
R["area"] = dict(piql_px_per_data_byte=PX_PER_DATA_BYTE_PIQL, size_MB=SIZE_MB, table=area,
                 min_feature_um=feat, NA_needed={f"s{s}": 0.55 / (2 * feat * s) for s in (1, 2, 5)})

# ---------------- 7. 仮説の判定 ----------------
g = {(d["p_film"], d["burst"]): d for d in grid}
H = {}
full_med = R["cost_py_per_MB"]["P_full_p0.0001"]["p50"]
H["H31a"] = dict(P_full_median=full_med, threshold=10 * MU018, supported=full_med >= 10 * MU018)
skip_med = R["cost_py_per_MB"]["P_skip"]["p50"]
H["H31b"] = dict(P_skip_median=skip_med, threshold=3 * MU018, supported=skip_med >= 3 * MU018)
h_err = g[(1e-3, False)]["A3_hamming_single"]; h_cost = R["cost_py_per_MB"]["A3_hamming_single"]["p50"]
H["H31c"] = dict(residual=h_err, cost_median=h_cost, err_ok=h_err <= 1e-4, cost_ok=h_cost <= MU018,
                 supported=(h_err <= 1e-4) and (h_cost <= MU018))
hb = g[(1e-3, True)]
H["H31d"] = dict(hamming_noninterleaved=hb["A3_hamming_single"], rep3=hb["A2_rep3_adaptive"],
                 supported=(hb["A3_hamming_single"] > 1e-4) and (hb["A2_rep3_adaptive"] <= 1e-4))
# 最良の手読み書式：p=1e-3 独立とバーストの両方で残留 ≤1e-4 を満たすものの中で費用中央値が最小
cands = []
for k in ("A1_raw_double", "A2_rep3_adaptive", "A3_hamming_single"):
    ok = g[(1e-3, False)][k] <= 1e-4 and g[(1e-3, True)][k] <= 1e-4
    cands.append((R["cost_py_per_MB"][k]["p50"], k, ok))
ok_c = sorted([c for c in cands if c[2]])
best = ok_c[0][1] if ok_c else None
H["H31e"] = dict(best_format=best, candidates=[dict(cost=c[0], fmt=c[1], meets_residual=c[2]) for c in cands],
                 frac_of_reel_s5=(area[f"{best}_s5"]["frac_of_reel"] if best else None),
                 supported=(best is not None and area[f"{best}_s5"]["frac_of_reel"] < 0.01))
H["H31f"] = dict(ratio_p1e5=rs_ratio["p1e-05"]["lite_over_full"], ratio_p1e3=rs_ratio["p0.001"]["lite_over_full"],
                 supported=(rs_ratio["p1e-05"]["lite_over_full"] <= 0.2) and (rs_ratio["p0.001"]["lite_over_full"] >= 0.5))
R["hypotheses"] = H

# 探索的（事前登録外）：H31d の境界判定の種依存性。(p=1e-3, バースト) を別の種5つで再実行
rep = []
for sd in range(1, 6):
    rr = np.random.default_rng(SEED + 100 + sd)
    def ch(nb, p, burst, rng=rr): return channel(nb, p, burst, rng=rr)
    nb = data_bits.size
    c = [data_bits ^ ch(nb, 1e-3, BURST) ^ (rr.random(nb) < M_SINGLE).astype(np.uint8) for _ in range(3)]
    agree = c[0] == c[1]
    dec = np.where(agree, c[0], (c[0].astype(int) + c[1] + c[2] >= 2).astype(np.uint8))
    cw = ham_encode(data_bits); fl = cw.reshape(-1)
    rxf = fl ^ ch(fl.size, 1e-3, BURST) ^ (rr.random(fl.size) < M_SINGLE).astype(np.uint8)
    rep.append(dict(seed=SEED + 100 + sd, rep3=byte_err(dec), hamming=byte_err(ham_decode(rxf.reshape(-1, 7)))))
R["exploratory_H31d_seed_replicates"] = dict(runs=rep, rep3_pass_frac=float(np.mean([r["rep3"] <= 1e-4 for r in rep])),
                                             hamming_fail_frac=float(np.mean([r["hamming"] > 1e-4 for r in rep])))
# 探索的：1000人年（100人×10年）で正しく読める MB（残留 ≤1e-4 を満たす書式のみ）
R["exploratory_MB_per_1000py"] = {k: float(1000 / R["cost_py_per_MB"][k]["p50"]) for k in cost}
# 探索的：Hamming の費用が 018 M-U 以下になるのに必要な t_bit（他は中央値）
need = (MU018 * SEC_PER_PY / 1e6 - 14 * float(np.median(t_px2))) / 24
R["exploratory_hamming_tbit_needed_sec"] = need

# ---------------- 8. 図 ----------------
fig, ax = plt.subplots(figsize=(9, 4.8))
keys = ["A1_raw_double", "A2_rep3_adaptive", "A3_hamming_single", "P_skip", "P_lite_p1e-05", "P_lite_p0.001", "P_full_p0.0001"]
ax.boxplot([np.log10(cost[k]) for k in keys], whis=(5, 95), showfliers=False)
ax.set_xticks(range(1, len(keys) + 1), [k.replace("_", "\n", 1) for k in keys], fontsize=7)
ax.axhline(np.log10(MU018), color="r", ls="--", label=f"018 M-U median {MU018:.2f}")
ax.set_ylabel("log10 person-years per MB (5-95%)"); ax.legend(); ax.set_title("Hand-reading cost by format (019)")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_cost_by_format.png", dpi=120)
fig, ax = plt.subplots(figsize=(7, 4.6))
ps = [d["p_film"] for d in grid if not d["burst"]]
for k, mk in (("A1_raw_double", "o"), ("A2_rep3_adaptive", "s"), ("A3_hamming_single", "^"), ("A3i_hamming_interleaved_single", "v")):
    ax.loglog(ps, [max(d[k], 1e-7) for d in grid if not d["burst"]], marker=mk, label=k)
ax.loglog(ps, [d["skip_rs"] for d in piql_grid], "k--", label="Piql, RS skipped")
ax.loglog(ps, [max(d["full_rs_codeword_fail"], 1e-7) for d in piql_grid], "k:", label="Piql, full RS (cw fail)")
ax.axhline(1e-4, color="gray", lw=0.8)
for k, mk in (("A1_raw_double", "o"), ("A2_rep3_adaptive", "s"), ("A3_hamming_single", "^"), ("A3i_hamming_interleaved_single", "v")):
    ax.scatter([1.3e-3], [max(hb[k], 1e-7)], marker=mk, color="r")
ax.set_xlabel("film pixel error p (red: +bursts f=1e-3, L=100 at p=1e-3)"); ax.set_ylabel("residual byte error (floor 1e-7)")
ax.legend(fontsize=7); ax.set_title("Residual error after hand decoding (1 MB simulated)")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_residual_error.png", dpi=120)
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy([d["p_px"] for d in rs_val], [max(d["fail_rate_mc"], 1 / (2 * NCW)) for d in rs_val], "o", label="MC (own RS decoder)")
ax.semilogy([d["p_px"] for d in rs_val], [max(d["fail_rate_analytic"], 1e-9) for d in rs_val], "-", label="Binomial tail t=24")
ax.set_xlabel("pixel error p (4 px/byte)"); ax.set_ylabel("RS(255,207) codeword failure"); ax.legend()
fig.tight_layout(); fig.savefig(f"{HERE}/fig_rs_validation.png", dpi=120)

R["runtime_seconds"] = time.time() - t0
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("cipher", "frame_capacity", "rs_validation", "rs_ops_fit_inner", "piql_residual",
                                     "cost_py_per_MB", "cost_components_median_sec_per_byte", "rs_detect_then_correct",
                                     "hypotheses", "exploratory_MB_per_1000py", "exploratory_hamming_tbit_needed_sec")},
                 ensure_ascii=False, indent=1, default=float))
print("binary grid:"); [print(d) for d in grid]
print("runtime", R["runtime_seconds"])
