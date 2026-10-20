"""本回開く公開の本文を取得して sources/ に保存する（計算の前に1回だけ実行。sim.py はネット接続なし）。
pm:<PMID>   PubMed の抄録（NCBI E-utilities efetch、rettype=abstract）
pmc:<PMCID> Europe PMC の全文 XML から本文の文字だけ（公開のもの）
pmchtml:<PMCID> PMC の記事の HTML から本文の文字（Europe PMC の XML が出ない著者稿）
wp:<title> / ja:<title> Wikipedia の本文（MediaWiki API の plain text extract）
"""
import json, os, re, sys, time, urllib.request, urllib.parse, html
H = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "EarthSurvivalInstitute/0.1 (research; box)"}
def http(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf8", "replace")
def pm(pmid):
    t = http("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&rettype=abstract&retmode=text&id=" + pmid)
    fn = f"pm_{pmid}.txt"; open(os.path.join(H, fn), "w", encoding="utf8").write(f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n" + t); print(fn, len(t))
def pmc(pmcid):
    x = http(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    x = re.sub(r"<(table-wrap|xref)[^>]*>", " ", x)
    x = re.sub(r"</(p|title|sec|caption|td|tr|th|label)>", "\n", x)
    t = html.unescape(re.sub(r"<[^>]+>", "", x)); t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
    fn = f"pmc_{pmcid}.txt"; open(os.path.join(H, fn), "w", encoding="utf8").write(f"URL: https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/\n" + t); print(fn, len(t))
def pmchtml(pmcid):
    """Europe PMC の XML が出ない著者稿は PMC の記事の HTML から本文の文字だけを取る。"""
    x = urllib.request.urlopen(urllib.request.Request(f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/", headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) EarthSurvivalInstitute/0.1"}), timeout=60).read().decode("utf8", "replace")
    x = re.sub(r"(?s)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", x)
    x = re.sub(r"</(p|h[1-6]|li|td|tr|th|caption|figcaption|div)>", "\n", x)
    t = html.unescape(re.sub(r"<[^>]+>", "", x)); t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
    fn = f"pmc_{pmcid}.txt"; open(os.path.join(H, fn), "w", encoding="utf8").write(f"URL: https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/\n" + t); print(fn, len(t))
def wp(title, lang="en"):
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json", "titles": title})
    d = json.loads(http(f"https://{lang}.wikipedia.org/w/api.php?" + q)); p = next(iter(d["query"]["pages"].values()))
    if "extract" not in p: print("MISSING", title); return
    t = f"URL: https://{lang}.wikipedia.org/wiki/" + p["title"].replace(" ", "_") + "\n" + p["extract"]
    fn = ("wp_" if lang == "en" else f"wp{lang}_") + p["title"].replace(" ", "_").replace("/", "_").replace("'", "") + ".txt"
    open(os.path.join(H, fn), "w", encoding="utf8").write(t); print(fn, len(t))
for a in sys.argv[1:]:
    kind, _, v = a.partition(":")
    for k in range(4):
        try:
            {"pm": pm, "pmc": pmc, "pmchtml": pmchtml, "wp": wp, "ja": lambda s: wp(s, "ja")}[kind](v); break
        except Exception as e:
            print("retry", a, e); time.sleep(5)
    time.sleep(1)
