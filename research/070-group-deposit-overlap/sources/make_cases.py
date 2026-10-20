"""資料を符号化して cases.json を作る（計算のみ、ネット接続なし）。
地域は国連 M49 の小地域（intermediate-region があればそれ、なければ sub-region）。対応表は lukes/ISO-3166-Countries-with-Regional-Codes の all.csv。"""
import csv, json, re
from collections import Counter, defaultdict
ISO = {r["name"]: r for r in csv.DictReader(open("iso_regions.csv", encoding="utf8"))}
ALIAS = {"Bolivia": "Bolivia, Plurinational State of", "Brunei": "Brunei Darussalam", "Cape Verde": "Cabo Verde", "Republic of Congo": "Congo",
         "Czech Republic": "Czechia", "Iran": "Iran, Islamic Republic of", "Laos": "Lao People's Democratic Republic", "Moldova": "Moldova, Republic of",
         "Netherlands": "Netherlands, Kingdom of the", "North Korea": "Korea, Democratic People's Republic of", "Russia": "Russian Federation",
         "South Korea": "Korea, Republic of", "Tanzania": "Tanzania, United Republic of", "Turkey": "Türkiye", "United Kingdom": "United Kingdom of Great Britain and Northern Ireland",
         "United States": "United States of America", "Venezuela": "Venezuela, Bolivarian Republic of", "Vietnam": "Viet Nam", "Taiwan": "Taiwan, Province of China",
         "Scotland": "United Kingdom of Great Britain and Northern Ireland", "Hong Kong": "Hong Kong", "Syria": "Syrian Arab Republic",
         "Democratic Republic of the Congo": "Congo, Democratic Republic of the", "DR Congo": "Congo, Democratic Republic of the", "Ivory Coast": "Côte d'Ivoire",
         "Cote d'Ivoire": "Côte d'Ivoire", "Swaziland": "Eswatini", "Macedonia": "North Macedonia", "Burma": "Myanmar", "East Timor": "Timor-Leste",
         "Palestine": "Palestine, State of", "Micronesia": "Micronesia (Federated States of)", "Federated States of Micronesia": "Micronesia (Federated States of)",
         "The Gambia": "Gambia", "Gambia": "Gambia", "The Bahamas": "Bahamas", "Vatican City": "Holy See", "São Tomé and Príncipe": "Sao Tome and Principe",
         "Sao Tome and Principe": "Sao Tome and Principe", "Kosovo": None, "Czechia": "Czechia", "Republic of the Congo": "Congo", "Guinea Bissau": "Guinea-Bissau",
         "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Trinidad & Tobago": "Trinidad and Tobago", "Saint Vincent and Grenadines": "Saint Vincent and the Grenadines",
         "Türkiye": "Türkiye", "Netherlands": "Netherlands, Kingdom of the", "Macao": "Macao", "Korea, North": "Korea, Democratic People's Republic of", "Republic of Ireland": "Ireland", "Macau": "Macao", "East_Timor": "Timor-Leste", "Åland": "Åland Islands", "Wales": "United Kingdom of Great Britain and Northern Ireland", "Quebec": "Canada", "Chechnya": "Russian Federation", "Tatarstan": "Russian Federation", "Catalonia": "Spain", "British Virgin Islands": "Virgin Islands (British)", "Korea, South": "Korea, Republic of"}
def region(name):
    n = ALIAS.get(name, name)
    if n is None or n not in ISO: return None
    r = ISO[n]; return r["intermediate-region"] or r["sub-region"]
def continent(name):
    n = ALIAS.get(name, name); r = ISO.get(n) if n else None
    if r is None: return None
    if r["region"] == "Americas": return "Americas_N" if r["sub-region"] == "Northern America" or r["intermediate-region"] in ("Central America", "Caribbean") else "Americas_S"
    return r["region"]
# 1) Xia 2022 補足 Table S5（Partial Livestock、本文 Table 1 の数）・S6（No Livestock）、貿易なし
X = json.load(open("xia_tables.json", encoding="utf8"))
cols = X["columns"]; i150 = cols.index("150Tg"); i47 = cols.index("47Tg")
surv = {}; unmapped = []
for tab in ("S5_partial_livestock", "S6_no_livestock"):
    agg = defaultdict(lambda: {"pop2010": 0.0, "surv150": 0.0, "surv47": 0.0})
    for c, v in X[tab].items():
        rg = region(c)
        if rg is None: unmapped.append(c); continue
        agg[rg]["pop2010"] += v[0]; agg[rg]["surv150"] += max(v[0] - v[i150], 0.0); agg[rg]["surv47"] += max(v[0] - v[i47], 0.0)
    surv[tab] = {k: dict(v) for k, v in agg.items()}
# 2) 1,000万点以上の蔵書の図書館（Wikipedia "List of largest libraries"）
t = json.load(open("wiki_List_of_largest_libraries.json", encoding="utf8"))["parse"]["wikitext"]
LIB = []
for r in t.split("\n|-")[1:]:
    m = re.search(r"\{\{[Ff]lag(?:icon|country)?\|([^}|]+)", r); n = re.search(r"\|\s*\[\[([^\]|]+)", r)
    if m: LIB.append({"library": n.group(1) if n else None, "country": m.group(1).strip(), "region": region(m.group(1).strip())})
# 3) 国立図書館（Wikipedia "List of national and state libraries" の National libraries の節、"see under" の行を除く）
t = json.load(open("wiki_List_of_national_and_state_libraries.json", encoding="utf8"))["parse"]["wikitext"].split("==State libraries==")[0]
NAT = []
for r in t.split("\n|-"):
    m = re.search(r"\{\{flagicon\|([^}|]+)", r)
    if not m or "see under" in r: continue
    if not re.search(r"\[\[[^\]]*(Library|Biblio|library|Archive|Bibli)", r): continue
    c = m.group(1).strip(); NAT.append({"country": c, "region": region(c)})
# 4) Boyd & Wilson（PubMed 抄録）の島の避難所
REF = {"asrs_2023": {"pmid": 36464495, "countries": ["Australia", "New Zealand", "Iceland", "Solomon Islands", "Vanuatu"],
                     "quote": "The island nations of Australia, New Zealand, Iceland, the Solomon Islands, and Vanuatu appear most resilient to ASRS."},
       "bio_2021": {"pmid": 33886124, "scores": {"Australia": 0.71, "New Zealand": 0.64, "Iceland": 0.58},
                    "quote": "Australia (0.71), New Zealand (0.64), and Iceland (0.58) as the leading candidate island nation refuges"}}
for k in REF:
    cs = REF[k].get("countries") or list(REF[k]["scores"]); REF[k]["regions"] = {c: region(c) for c in cs}
# 5) 南半球の人口の割合（Wikipedia "Southern Hemisphere"）と北の金庫（Arctic World Archive、Svalbard）
SH = {"quote": "More than 850 million people live in the Southern Hemisphere, representing around 10–12% of the total global human population", "share": [0.10, 0.12]}
regions_all = sorted(set(r["intermediate-region"] or r["sub-region"] for r in ISO.values() if r["region"]))
cont_of_region = {}
for r in ISO.values():
    if r["region"]: cont_of_region[r["intermediate-region"] or r["sub-region"]] = continent(r["name"])
out = {"regions": regions_all, "continent_of_region": cont_of_region, "xia_survivors_by_region": surv, "xia_unmapped": sorted(set(unmapped)),
       "largest_libraries": LIB, "national_libraries": NAT, "refuges": REF, "southern_hemisphere": SH,
       "switch_rule_count": {"rule": "publicly opened numeric sources on survivor geography or deposit geography >= 5",
                             "sources": ["Xia 2022 SI Table S5/S6 (per-country survivors, survivor geography)",
                                         "Wikipedia List of largest libraries (34 libraries >=10M items by country, deposit geography)",
                                         "Wikipedia List of national and state libraries (national libraries by country, deposit geography)",
                                         "Boyd & Wilson 2023 Risk Anal (PMID 36464495, ASRS island refuges, survivor geography)",
                                         "Boyd & Wilson 2021 Risk Anal (PMID 33886124, bio refuge scores, survivor geography)",
                                         "Wikipedia Southern Hemisphere (10-12% of population, survivor geography check)"]}}
json.dump(out, open("cases.json", "w"), indent=1, ensure_ascii=False)
print("regions", len(regions_all), "xia unmapped", out["xia_unmapped"])
print("largest libs", len(LIB), Counter(l["region"] for l in LIB).most_common())
print("national libs", len(NAT), "unmapped", [n["country"] for n in NAT if n["region"] is None])
for tab in surv:
    tot = sum(v["surv150"] for v in surv[tab].values()); tp = sum(v["pop2010"] for v in surv[tab].values())
    print(tab, round(tp, 1), round(tot, 1), sorted(((k, round(v["surv150"] / tot, 3)) for k, v in surv[tab].items()), key=lambda x: -x[1])[:10])
