"""世界銀行 API（UIS の年齢別識字率）を取得して wb_literacy.json に保存する。"""
import json, os, time, urllib.request
H = os.path.dirname(os.path.abspath(__file__))
IND = ["SE.ADT.1524.LT.ZS", "UIS.LR.AG25T64", "UIS.LR.AG65", "SE.ADT.LITR.ZS", "SE.ADT.1524.LT.FE.ZS", "UIS.LR.AG25T64.F", "UIS.LR.AG65.F"]
CC = ["KHM", "SOM", "AFG", "IRQ", "LBR", "SLE", "SSD", "YEM", "SYR", "MOZ", "AGO", "RWA", "BDI", "CAF", "COD", "ERI", "ETH", "TJK", "TCD", "MLI", "NER", "SDN", "LKA", "NPL", "GNB", "UGA", "ZWE", "BIH", "PSE", "HTI"]
out = {}
for i in IND:
    url = f"https://api.worldbank.org/v2/country/{';'.join(CC)}/indicator/{i}?format=json&per_page=20000"
    for k in range(4):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "EarthSurvivalInstitute/0.1"}), timeout=60)); break
        except Exception as e: print("retry", i, e); time.sleep(5)
    out[i] = {"url": url, "rows": [[x["countryiso3code"], x["date"], x["value"]] for x in (d[1] or []) if x["value"] is not None]}
    print(i, len(out[i]["rows"])); time.sleep(1)
json.dump(out, open(os.path.join(H, "wb_literacy.json"), "w"), indent=0)
