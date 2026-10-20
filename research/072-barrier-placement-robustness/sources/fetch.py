"""英語版 Wikipedia の本文（平文）を MediaWiki API で取得して保存する（公開情報の閲覧のみ）。"""
import json, urllib.request, urllib.parse, sys
PAGES = ["Kula ring", "Moka exchange", "Hxaro", "Hopewell tradition", "Lapita culture", "Polynesian navigation",
         "Trade in Aboriginal Australia", "Inuit", "San people", "Aboriginal Australians", "Japanese missions to Tang China",
         "History of Indigenous Australians", "Pauline Wiessner"]  # 後の2つは欠けた頁の代わり（報告の逸脱）
def get(title):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1, "inprop": "url",
                                "titles": title, "format": "json", "formatversion": 2})
    req = urllib.request.Request("https://en.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "EarthSurvivalInstitute/0.1 (research; contact via github houjyouki-ship-it)"})
    p = json.load(urllib.request.urlopen(req, timeout=60))["query"]["pages"][0]
    return {"title": p.get("title"), "url": p.get("fullurl"), "missing": p.get("missing", False), "lastrevid": p.get("lastrevid"), "text": p.get("extract", "")}
out = json.load(open("wiki_texts.json"))
for t in (sys.argv[1:] or PAGES):
    if t not in out: out[t] = get(t)
json.dump(out, open("wiki_texts.json", "w"), ensure_ascii=False, indent=1)
for t, v in out.items(): print(t, "->", v["title"], v["missing"], len(v["text"]), v["url"])
