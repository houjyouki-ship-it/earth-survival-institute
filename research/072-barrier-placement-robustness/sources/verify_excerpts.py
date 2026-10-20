import json
T = json.load(open("wiki_texts.json", encoding="utf8")); C = json.load(open("cases.json", encoding="utf8"))["cases"]
ok = 0
for c in C:
    s = " ".join(T[c["page"]]["text"].split()); q = " ".join(c["quote"].split())
    good = q in s; ok += good; print("OK " if good else "NG ", c["id"])
print(f"{ok}/{len(C)} 一致"); assert ok == len(C)
