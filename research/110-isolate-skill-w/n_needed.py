"""研究110 付録：遅い型が s 件のとき、w の事後の中央値が 0.022 を下回るのに要る速い型の件数 f（Jeffreys 事前）。"""
from scipy.stats import beta
import json
out = {}
for s in (0, 1, 2):
    f = 0
    while beta.median(0.5 + s, 0.5 + f) >= 0.022: f += 1
    f95 = 0
    while beta.ppf(0.95, 0.5 + s, 0.5 + f95) >= 0.022: f95 += 1
    out[s] = {"fast_needed_median": f, "fast_needed_95pct": f95}
print(json.dumps(out)); json.dump(out, open("n_needed.json", "w"))
