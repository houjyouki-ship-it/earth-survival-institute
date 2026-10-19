# 研究054：引用を保存した本文と照合する（ネット接続なし）。WebFetch でだけ確認したものは found に注記。
import json, re, html, os
H = os.path.dirname(os.path.abspath(__file__))
def wiki(f): return list(json.load(open(os.path.join(H, f)))["query"]["pages"].values())[0]["extract"]
def page(f):
    t = open(os.path.join(H, f), errors="ignore").read(); t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\u00ad", "")
    return re.sub(r"\s+", " ", t)
W = "https://en.wikipedia.org/wiki/"
E = [
 (W + "Svalbard_Global_Seed_Vault", "Svalbard_Global_Seed_Vault.json", "approximately 45 million kr (US$8.8 million in 2008) construction cost"),
 (W + "Svalbard_Global_Seed_Vault", "Svalbard_Global_Seed_Vault.json", "In 2019, the Seed Vault cost about 2.4 million kr (US$282,000) to maintain."),
 (W + "Svalbard_Global_Seed_Vault", "Svalbard_Global_Seed_Vault.json", "The Crop Trust provides most of the annual operating costs for the facility and has set aside an endowment fund to do so, while the Norwegian government finances upkeep of the structure itself."),
 (W + "Arctic_World_Archive", "Arctic_World_Archive.json", "The data is stored on 186 film reels measuring 1 kilometre"),
 (W + "Clock_of_the_Long_Now", "Clock_of_the_Long_Now.json", "is being funded by Jeff Bezos's investment firm Bezos Expeditions, with $42 million"),
 (W + "Southern_Hemisphere", "Southern_Hemisphere.json", "More than 850 million people live in the Southern Hemisphere, representing around 10–12% of the total global human population."),
 (W + "Memory_of_Mankind", "Memory_of_Mankind.json", "carries up to 5 million characters"),
 ("https://arcticworldarchive.org/deposit/", "awa_deposit.html", "from € 139 one-time deposit, no subscription"),
 ("https://arcticworldarchive.org/deposit/", "awa_deposit.html", "For larger organisations wishing to own their dedicated film(s) and preserve large volumes of data."),
 ("https://arcticworldarchive.org/deposit/", "awa_deposit.html", "from € 8900 one-time deposit, no subscription"),
 ("https://www.nordgen.org/our-work/svalbard-global-seed-vault/frequently-asked-questions/", "nordgen_faq.html", "Norway spent approximately EUR 8.3 million in constructing the Seed Vault and later invested approximately EUR 20 million in the technical upgrade of the facility."),
 ("https://www.nordgen.org/our-work/svalbard-global-seed-vault/frequently-asked-questions/", "nordgen_faq.html", "The annual running cost of the Seed Vault is approximately EUR 1 million."),
 ("https://www.nordgen.org/our-work/svalbard-global-seed-vault/frequently-asked-questions/", "nordgen_faq.html", "There are no permanent staff on site at the Svalbard Global Seed Vault."),
 ("https://www.stortinget.no/no/Saker-og-publikasjoner/Publikasjoner/Innstillinger/Stortinget/2017-2018/inns-201718-220s", "stortinget_220s.html", "Prosjektet har en styringsramme (P50) på 155,8 mill. kroner og en kostnadsramme (P85) på 202,95 mill. kroner i 2018-priser."),
 ("https://www.nrk.no/klima/matte-reparere-frohvelvet-for-193-millioner.-skyldte-pa-klimaendringer.-1.15412828", "nrk_193.html", "Hun ledet komiteen som i 2018 bevilget 193 millioner skattekroner til utbedringen av Svalbard globale frøhvelv."),
 ("https://www.nrk.no/klima/matte-reparere-frohvelvet-for-193-millioner.-skyldte-pa-klimaendringer.-1.15412828", "nrk_193.html", "Det var fire ganger så mye som det hadde kostet å bygge anlegget."),
 ("https://www.botanicgardens.org.au/our-science/science-facilities/australian-plantbank/about-australian-plantbank/plantbank-purpose", "plantbank_purpose.html", "The Australian PlantBank was built in 2013 at a cost of $19.8 million; the NSW state government contributed $15.5 million"),
 ("https://www.theguardian.com/science/2013/oct/11/australian-plantbank-opens-with-mission-to-protect-and-preserve", "guardian_plantbank.html", "The $20m Sydney facility includes the largest native plant seed bank in Australia"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-token-by-mail/", "mom_1_1.html", "1/1 Tablet in the MOM Archiv plus Token by mail € 195.00"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-token-by-mail/", "mom_1_1.html", "If you write solely text: capacity up to 50,000 characters"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-1-duplicate-and-the-token-by-mail/", "mom_1_1_dup.html", "1/1 Tablet in the MOM Archiv plus 1 Duplicate and the Token by mail € 455.00"),
 ("https://www.memory-of-mankind.com/product/1-1-tablet-in-the-mom-archiv-plus-1-duplicate-and-the-token-by-mail/", "mom_1_1_dup.html", "In addition: a duplicate tablet in a gift-box together with the certificate and the token."),
]
for y, v in (("2008", "215.3"), ("2011", "224.9"), ("2013", "233.0"), ("2018", "251.1"), ("2019", "255.7"), ("2025", "321.9")):
    E.append(("https://www.minneapolisfed.org/about-us/monetary-policy/inflation-calculator/consumer-price-index-1913-", "mpls_cpi.html", f"{y} {v}"))
out = []; ok = 0
for url, f, qq in E:
    t = wiki(f) if f.endswith(".json") else page(f); hit = qq in t; ok += hit
    out.append({"url": url, "file": f, "quote": qq, "found": hit})
IT = "https://www.itpro.com/infrastructure/server-storage/358137/deep-freeze-data-that-will-last-1000-years"
for qq in ("Each reel of film is 950m long and holds 120GB of data.",
           "In July 2020, GitHub sent 180 reels of piqlFilm to Svalbard, which equates to more than 20TB of data.",
           "Dohmke tells us the cost of archiving with Piql is “not significant in the business model of GitHub and Microsoft”."):
    out.append({"url": IT, "file": None, "quote": qq, "found": "checked via WebFetch 2026-10-19, not saved (curl returned a page without the article)"})
for qq in ("Earth’s Black Box will be built on 500 million y.o. granite in remote west coast Tasmania.",
           "The device is yet to be built, but it was launched during COP26 and is already recording."):
    out.append({"url": "https://www.rouserlab.com/earths-black-box/", "file": None, "quote": qq, "found": "checked via WebFetch 2026-10-19, not saved"})
json.dump(out, open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(ok, "/", len(E), "found in saved text")
for o in out:
    if o["found"] is False: print("NOT FOUND:", o["quote"])
