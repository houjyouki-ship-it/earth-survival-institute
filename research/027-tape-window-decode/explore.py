"""研究027 探索（事前登録外）：最良構成が格子の端（K=8, L=64 KiB）だったので、K と L をさらに広げる。
sim.py の図・保存より前の部分をそのまま実行して関数と母集団を使い、explore_results.json に書く。"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(f"{HERE}/sim.py").read().split("# ---- 図 ----")[0]
g = {"__file__": f"{HERE}/sim.py", "__name__": "sim_core"}; exec(src, g)
E = {"note": "exploratory, not preregistered", "configs": {}}
for K in (8, 12, 16, 32):
    for L in (32 * 1024, 64 * 1024):
        for lab, v in (("main", g["V_RW"]), ("v500", g["V_REW"])):
            res = g["run"](dict(K=K, L=L), 65536, v_sp=v)
            T = g["repo_times"](res, 65536)
            E["configs"][f"K{K}_L{L // 1024}K|{lab}"] = {"median_ratio": g["med_ratio"](res), "q_Y10": g["q_budget"](T, 10), "q_Y1": g["q_budget"](T, 1),
                                                         "total_machine_years_all": float((g["w_cnt"] * T).sum() / g["YEAR"])}
cpu_T = g["repo_times"]({r["idx"]: (0.0, {}) for r in g["reps"]}, 65536)
E["cpu_only_q_Y10"] = g["q_budget"](cpu_T, 10); E["cpu_only_total_machine_years"] = float((g["w_cnt"] * cpu_T).sum() / g["YEAR"])
json.dump(E, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(E, ensure_ascii=False, indent=1))
