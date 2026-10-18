"""本回開く Wikipedia の本文を MediaWiki API（plain text extract）で取得する。"""
import json, os, sys, time, urllib.request, urllib.parse
H = os.path.dirname(os.path.abspath(__file__))
def get(title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "EarthSurvivalInstitute/0.1 (research; box)"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    p = next(iter(d["query"]["pages"].values()))
    if "extract" not in p: print("MISSING", title); return
    t = "URL: https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_") + "\n" + p["extract"]
    fn = "wp_" + p["title"].replace(" ", "_").replace("/", "_").replace("'", "") + ".txt"
    open(os.path.join(H, fn), "w", encoding="utf8").write(t); print(fn, len(t))
for t in sys.argv[1:]:
    for k in range(4):
        try:
            get(t); break
        except Exception as e:
            print('retry', t, e); time.sleep(5)
    time.sleep(2)
