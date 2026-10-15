"""010 補足（事後解析。H19 の登録後、sim.py 実行中に追加）: 3区間の境界（1914, 1940）はデータを見て選ばれているため、
固定ビンの逸脱度検定は「見る場所の選択」効果で一定率を過剰に棄却しうる。Clauset 2018 Table 1 と同じ型の走査統計で確認する。
再現: python3 supp_scan.py > supp_stdout.txt (seed=20261014) -> supp_results.json
定義（年単位に離散化した本研究所の定義。Clauset の正確な定式化とは一致しない【推測】）:
  V: 連続27年の窓に大規模戦争の開始が10件以上, P: 連続64年の窓に5件以下, VP: V の窓の直後64年が P。
  期間 1823–2003（181年）。一定率は 24/181 件/年。Cox は sim.py と同じ対数正規 OU（λ_w は sim.py のプロファイル値）。
【事実（Clauset 2018, https://pmc.ncbi.nlm.nih.gov/articles/PMC5834001/ Table 1）】定常モデル1–3で Pr(V)=0.107/0.159/0.121,
  Pr(P)=0.622/0.569/0.681, Pr(V の後に P)=0.0030/0.0029/0.0055。
"""
import json, math
import numpy as np
SEED = 20261014; YEARS = 181; N = 200_000
res = {"seed": SEED, "N": N, "definition": "yearly bins; V: >=10 in 27y; P: <=5 in 64y; VP: V window then next 64y <=5"}
def stats(counts):
    cs = np.concatenate([np.zeros((counts.shape[0], 1), dtype=np.int64), np.cumsum(counts, axis=1)], axis=1)
    w27 = cs[:, 27:] - cs[:, :-27]; w64 = cs[:, 64:] - cs[:, :-64]
    V = (w27 >= 10).any(axis=1); P = (w64 <= 5).any(axis=1)
    # VP: 窓 [k,k+27) が V かつ [k+27,k+91) が P
    nk = YEARS - 91 + 1
    VP = ((w27[:, :nk] >= 10) & (w64[:, 27:27 + nk] <= 5)).any(axis=1)
    obs3 = None
    return {"P_V": float(V.mean()), "P_P": float(P.mean()), "P_VP": float(VP.mean()),
            "SE_VP": float(VP.std() / math.sqrt(len(VP)))}
emp = np.zeros((1, YEARS), dtype=np.int64)  # 経験データは年次が手元にないので3区間集計の確認のみ
rng = np.random.default_rng([SEED, 10, 99])
lam = 24 / YEARS
res["const_poisson"] = stats(rng.poisson(lam, size=(N, YEARS)))
lik = json.load(open("results.json"))["clauset_cox_grid"]
for sg, th in [(0.5, 10.0), (1.0, 3.0), (1.0, 10.0), (1.0, 30.0), (1.5, 3.0), (2.0, 3.0)]:
    lw = lik[f"s{sg}_t{th:g}"]["lam_w_hat"]
    r = np.random.default_rng([SEED, 10, 98, int(sg * 100), int(th)])
    x = r.standard_normal(N) * sg; a = math.exp(-1 / th); c = sg * math.sqrt(1 - a * a)
    L = np.empty((N, YEARS))
    for k in range(YEARS):
        L[:, k] = lw * np.exp(x - sg ** 2 / 2); x = a * x + c * r.standard_normal(N)
    res[f"cox_s{sg}_t{th:g}"] = dict(stats(r.poisson(L)), lam_w=lw)
for k, v in res.items():
    if isinstance(v, dict): print(k, v)
json.dump(res, open("supp_results.json", "w"), indent=1)
