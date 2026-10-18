import json, sys, urllib.request, urllib.parse, os, time
H = os.path.dirname(os.path.abspath(__file__))
def get(title):
    q = urllib.parse.urlencode({"action":"query","prop":"extracts","explaintext":1,"redirects":1,"format":"json","titles":title})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?"+q, headers={"User-Agent":"ESI-research/1.0 (contact: none)"})
    for k in range(5):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=30)); break
        except Exception as e:
            print("retry", title, e); time.sleep(3 + 3 * k)
    else:
        print("FAILED", title); return
    p = list(d["query"]["pages"].values())[0]
    if "extract" not in p: print("MISSING", title); return
    t = "URL: https://en.wikipedia.org/wiki/" + p["title"].replace(" ", "_") + "\n" + p["extract"]
    fn = "wp_" + p["title"].replace(" ", "_").replace("/", "_").replace("'", "") + ".txt"
    open(os.path.join(H, fn), "w", encoding="utf8").write(t); print(fn, len(t))
for t in sys.argv[1:]:
    get(t); time.sleep(1.5)
