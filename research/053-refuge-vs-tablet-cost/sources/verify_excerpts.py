import json, re, html, os
H = os.path.dirname(os.path.abspath(__file__))
def wiki(f): return list(json.load(open(os.path.join(H, f)))["query"]["pages"].values())[0]["extract"]
def page(f):
    t = open(os.path.join(H, f), errors="ignore").read(); t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
E = [
 ("https://en.wikipedia.org/wiki/Amundsen–Scott_South_Pole_Station", "Amundsen–Scott_South_Pole_Station.json", "In 1992, the design of a new station began for an 80,000-square-foot (7,400 m2) building with two floor levels that cost US$150 million."),
 ("https://en.wikipedia.org/wiki/Amundsen–Scott_South_Pole_Station", "Amundsen–Scott_South_Pole_Station.json", "with a peak population of around 150 in the summer operational season"),
 ("https://en.wikipedia.org/wiki/Amundsen–Scott_South_Pole_Station", "Amundsen–Scott_South_Pole_Station.json", "The facility was officially dedicated on January 12, 2008"),
 ("https://en.wikipedia.org/wiki/Halley_Research_Station", "Halley_Research_Station.json", "Cost: Approximately £26 million"),
 ("https://en.wikipedia.org/wiki/Halley_Research_Station", "Halley_Research_Station.json", "staff numbers count about 52"),
 ("https://en.wikipedia.org/wiki/Biosphere_2", "Biosphere_2.json", "with Bass providing US$150 million in funding until 1991"),
 ("https://en.wikipedia.org/wiki/Biosphere_2", "Biosphere_2.json", "Its mission was a two-year closure experiment with a crew of eight humans."),
 ("https://en.wikipedia.org/wiki/Biosphere_2", "Biosphere_2.json", "the eight inhabitants reported continual hunger"),
 ("https://en.wikipedia.org/wiki/Svalbard_Global_Seed_Vault", "Svalbard_Global_Seed_Vault.json", "approximately 45 million kr (US$8.8 million in 2008) construction cost"),
 ("https://en.wikipedia.org/wiki/Svalbard_Global_Seed_Vault", "Svalbard_Global_Seed_Vault.json", "In 2019, the Seed Vault cost about 2.4 million kr (US$282,000) to maintain."),
 ("https://en.wikipedia.org/wiki/Fallout_shelter", "Fallout_shelter.json", "Germany has protected shelters for 3% of its population, Austria for 30%, Finland for 70%, Sweden for 81%, and Switzerland for 114%."),
 ("https://en.wikipedia.org/wiki/Fallout_shelter", "Fallout_shelter.json", "most residential shelters are no longer stocked with the food and water required for prolonged habitation"),
 ("https://en.wikipedia.org/wiki/Memory_of_Mankind", "Memory_of_Mankind.json", "carries up to 5 million characters"),
 ("https://en.wikipedia.org/wiki/Arctic_World_Archive", "Arctic_World_Archive.json", "The data is stored on 186 film reels"),
 ("https://en.wikipedia.org/wiki/Vivos_(underground_shelter)", "Vivos_(underground_shelter).json", "With accommodations for 80 people"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-token-by-mail/", "mom_1_1.html", "1/1 Tablet in the MOM Archiv plus Token by mail € 195.00"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-token-by-mail/", "mom_1_1.html", "If you write solely text: capacity up to 50,000 characters"),
 ("https://www.thenationalnews.com/uae/super-rich-prepare-for-doomsday-in-dh180m-american-bunkers-1.804263", "national.html", "a cool $50 million (Dh183 million) can get you the entire 15-storey silo, which can accommodate 12"),
 ("https://www.thenationalnews.com/uae/super-rich-prepare-for-doomsday-in-dh180m-american-bunkers-1.804263", "national.html", "People can try to ride out the worst there for at least five years."),
 ("https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1913-", "mpls_cpi.html", "1962 30.3"),
]
out = []; ok = 0
for url, f, q in E:
    t = wiki(f) if f.endswith(".json") else page(f); hit = q in t; ok += hit
    out.append({"url": url, "file": f, "quote": q, "found": hit})
# WebFetch でだけ確認（ファイルなし）：Ullmann 1962
out.append({"url": "https://prephole.com/wp-content/preparedness-ebooks/National-Shelter-Program-FeasibilityStudy.pdf", "file": None,
            "quote": "The shelters are estimated to cost about $50,500 for 20 persons and $1,063,400 for 500 persons, or from $2,125 to $2,525 per place.", "found": "checked via WebFetch 2026-10-19, not saved"})
out.append({"url": "https://prephole.com/wp-content/preparedness-ebooks/National-Shelter-Program-FeasibilityStudy.pdf", "file": None,
            "quote": "food and water stores have been set at six months", "found": "checked via WebFetch 2026-10-19, not saved"})
out.append({"url": "https://prephole.com/wp-content/preparedness-ebooks/National-Shelter-Program-FeasibilityStudy.pdf", "file": None,
            "quote": "Storage space (2 cubic feet per person per week @ $1.50 per cubic feet) 26 weeks ... $2,835 (for 20 persons, food + storage)", "found": "checked via WebFetch 2026-10-19 (OCR table, paraphrased layout)"})
json.dump(out, open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(ok, "/", len(E), "found")
