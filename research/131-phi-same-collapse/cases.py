"""研究131 段を保った地域と失った地域の停滞の比 phi（事前登録 PREREGISTRATION.md の規則）。
注：F_heavy・V は生存確率や復興確率ではない。数値は sources/ の Wikipedia 頁（2026-10-26 取得）から。"""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
R125 = json.load(open(os.path.join(HERE, "..", "125-pop-recovery-history", "results.json")))
brit = [c for c in R125["cases"] if c["key"] == "Britain_postRoman"][0]
R_MED = math.log(brit["P0"] / brit["Pmin"]) / brit["pred038_median_years"]  # 038 の r の中央値を 125 から逆算

def interp_year(y0, p0, y1, p1, target):  # 指数補間で target に達する年
    return y0 + (y1 - y0) * math.log(target / p0) / math.log(p1 / p0)

def excess(P0, Pmin, t_start, t_rec):
    L = 1 - Pmin / P0
    if L < 0.2: return dict(L=L, T_rec=None, pred=None, excess=0.0, note="L<0.2 で停滞なし（規則）")
    pred = math.log(P0 / Pmin) / R_MED
    return dict(L=L, T_rec=t_rec - t_start, pred=pred, excess=(t_rec - t_start) - pred)

C = {}
# Britain（lost, 125 の値のまま）
C["Britain"] = dict(cls="lost", src="125", **excess(brit["P0"], brit["Pmin"], brit["t_start"], brit["t_rec"]))
# Italy（kept, 129）：165 CE 8–9M（Demography_of_the_Roman_Empire）、Charlemagne の頃 4M（Medieval_demography）、
# 表（Medieval_demography）1200 8M・1250 9M → 指数補間で回復年
for lab, P0 in (("lo", 8.0e6), ("main", 8.5e6), ("hi", 9.0e6)):
    tr = interp_year(1200, 8e6, 1250, 9e6, P0) if P0 > 8e6 else 1200.0
    C[f"Italy|{lab}"] = dict(cls="kept", t_rec=tr, **excess(P0, 4.0e6, 165, tr))
# Gaul/France（kept：Merovingian の貨幣・Latin の文書、Francia・Merovingian 頁）：Demographics_of_France 表 120 7.2M、400 5.5M、850 7.0M、1226 16M
tr_fr = interp_year(850, 7.0e6, 1226, 16e6, 7.2e6)
C["France|Pmin5.5(400)"] = dict(cls="kept", t_rec=tr_fr, **excess(7.2e6, 5.5e6, 120, tr_fr))
C["France|Pmin5.0(Carolingian)"] = dict(cls="kept", t_rec=tr_fr, **excess(7.2e6, 5.0e6, 120, tr_fr))
# Iberia（kept：Visigothic の tremissis・Reccopolis の basilica・法典）：Frier 164 CE 7.5M／Harper 9M、Charlemagne の頃 4M、表 1000 9M（Spain and Portugal）
for lab, P0 in (("Frier7.5", 7.5e6), ("Harper9", 9.0e6)):
    tr = interp_year(800, 4e6, 1000, 9e6, P0)
    C[f"Iberia|{lab}"] = dict(cls="kept", t_rec=tr, immig="あり（711 以後のアラブ・ベルベル人の移入、量の数値なし）", **excess(P0, 4.0e6, 164, tr))
# 共通の始点 400 の版：France は 400 5.5M、その後の谷は Carolingian 5M（別の範囲の推定）→ L=0.09
C["France|start400"] = dict(cls="kept", **excess(5.5e6, 5.0e6, 400, None))
NOT = {"Italy|start400": "400 年のイタリアの人口の数値なし", "Iberia|start400": "400 年のイベリアの人口の数値なし",
       "Egypt_LBA": "Ancient_Egypt：前3千年紀 1–1.5M、前1千年紀 2–3M のみ。前1200 前後の減少と回復の数値なし",
       "Hittite_Anatolia": "Hittites：王国の人口 14–15万（Simon）／20万超（Bryce）のみ、崩壊後の数値なし",
       "Assyria": "Middle/Neo-Assyrian 頁に地域の人口の時系列なし"}
eb = C["Britain"]["excess"]
PHI = {k: max(v["excess"], 0) / eb for k, v in C.items() if v["cls"] == "kept"}
pairs_peak = {"Italy": PHI["Italy|main"], "France": PHI["France|Pmin5.5(400)"], "Iberia": PHI["Iberia|Frier7.5"]}
new_peak = sorted([pairs_peak["France"], pairs_peak["Iberia"]])
OUT = dict(R_MED=R_MED, cases=C, not_obtained=NOT, phi=PHI, pairs_peak=pairs_peak,
           phi_new_median_peak=sum(new_peak) / 2, phi_all_median_peak=sorted(pairs_peak.values())[1],
           phi_common_start=PHI["France|start400"],
           r_eff_descriptive={"Britain": math.log(3) / (1184 - 600), "Italy": math.log(8.5 / 4) / (C["Italy|main"]["t_rec"] - 800),
                              "France": math.log(7.2 / 5.0) / (tr_fr - 800), "Iberia": math.log(7.5 / 4) / (C["Iberia|Frier7.5"]["t_rec"] - 800)})
if __name__ == "__main__":
    for k, v in C.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "immig"})
    print("phi", {k: round(v, 3) for k, v in PHI.items()})
    print("median new(peak)", round(OUT["phi_new_median_peak"], 3), "all(peak)", round(OUT["phi_all_median_peak"], 3), "common start", OUT["phi_common_start"])
    print("r_eff", {k: round(v, 5) for k, v in OUT["r_eff_descriptive"].items()})
    json.dump(OUT, open(os.path.join(HERE, "cases.json"), "w"), indent=1, ensure_ascii=False, default=float)
