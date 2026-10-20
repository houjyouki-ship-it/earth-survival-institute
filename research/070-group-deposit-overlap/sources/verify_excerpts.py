"""cases.json の引用文が保存した本文にあるかを確かめる（空白を詰めて照合）。"""
import json, re
sq = lambda s: re.sub(r"\s+", " ", s)
C = json.load(open("cases.json", encoding="utf8")); ok = []
txt = {"36464495": open("pm_36464495.txt", encoding="utf8").read(), "33886124": open("pm_33886124.txt", encoding="utf8").read()}
ok.append(sq(C["refuges"]["asrs_2023"]["quote"]) in sq(txt["36464495"]))
ok.append(sq(C["refuges"]["bio_2021"]["quote"]) in sq(txt["33886124"]))
sh = json.load(open("wiki_Southern_Hemisphere.json", encoding="utf8"))["parse"]["wikitext"].replace("&nbsp;", " ")
ok.append(sq(C["southern_hemisphere"]["quote"]) in sq(sh))
x = open("xia_moesm1.txt", encoding="utf8").read()
ok.append("Total                         6700.1   254.7   925.6   1426.1 2080.9 2512.2 5340.8   5205.1   5081.2" in x)
print(len(ok), "excerpts checked; all found:", all(ok), ok)
