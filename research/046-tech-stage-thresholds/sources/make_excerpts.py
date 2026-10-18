"""本回開いた Wikipedia の本文（MediaWiki API の plain text、2026-10-18 取得）から引用を照合して excerpts.json を作る。"""
import json, os, re, html
H = os.path.dirname(os.path.abspath(__file__))
Q = [
 ("wp_Norse_settlements_in_Greenland.txt", "The already not very productive smelting of iron ore quickly reached its limits due to the lack of suitable fuel (charcoal), so that the settlements were almost entirely dependent on imports."),
 ("wp_Norse_settlements_in_Greenland.txt", "There were no known ore deposits in Greenland at the time of the Vikings."),
 ("wp_LAnse_aux_Meadows.txt", "There is no way to know the site's population at any given time, though the dwellings could accommodate 30 to 160 people. The entire population of Greenland at the time was about 2,500"),
 ("wp_LAnse_aux_Meadows.txt", "Workshops were identified as an iron smithy (building J) containing a forge and iron slag"),
 ("wp_Bog_iron.txt", "Large scale production of bog iron was also established in Iceland at sites known as \"iron farms\". Smaller scale production sites in Iceland consisted of large farmsteads and some original Icelandic settlements, but these seemed to only produce enough iron to be self-sufficient."),
 ("wp_Demographics_of_Iceland.txt", "The population of Iceland probably wavered between about 30,000 and 80,000 for most of the time since settlement. Official statistics begin in 1703, since when the population has grown from 50,358"),
 ("wp_Saugus_Iron_Works_National_Historic_Site.txt", "It is the site of the first integrated ironworks in North America, founded by John Winthrop the Younger and in operation between 1646 and approximately 1670."),
 ("wp_Saugus_Iron_Works_National_Historic_Site.txt", "Once functioning, the Iron Works ran for thirty weeks of the year and produced one ton of cast iron a day."),
 ("wp_Saugus_Iron_Works_National_Historic_Site.txt", "it seldom operated at a profit. The Iron Works closed around 1670."),
 ("wp_Saugus_Iron_Works_National_Historic_Site.txt", "35 or 37 of the 60 Scots intended for the Iron Works were employed there"),
 ("wp_Falling_Creek_Ironworks.txt", "It was short-lived due to an attack by Native Americans in 1622."),
 ("wp_Tatara_(furnace).txt", "Four or five people need to constantly work on this process."),
 ("wp_Tatara_(furnace).txt", "the tatara will have consumed about 9.1 t (9.0 long tons; 10.0 short tons) of satetsu and 11 t (11 long tons; 12 short tons) of charcoal, leaving about 2.3 t (2.3 long tons; 2.5 short tons) of tamahagane."),
 ("wp_Fitzroy_Iron_Works.txt", "between 600 and 700 people (employees and their families) lived off wages paid by the works."),
 ("wp_Fitzroy_Iron_Works.txt", "in 1865 the landed cost of imported pig-iron was £5 per ton, while that made at the works cost £5 17s 6d."),
 ("wp_List_of_19th-century_iron_smelting_operations_in_Australia.txt", "After 1884, there was no commercial iron smelting in Australia, until William Sandford built a modern blast furnace at Lithgow in 1907."),
 ("wp_Jamestown_Glasshouse.txt", "However, their operations did not likely survive the winter of 1609, when the colonist population dropped from around 500 to 60."),
 ("wp_Jamestown_Glasshouse.txt", "A second attempt was made around 1621, when four Italian glassblowers and two assistants attempted to revive operations. However, this attempt was not productive"),
 ("wp_Early_glassmaking_in_the_United_States.txt", "Production failure was attributed to inadequate security, food supply, quality of sand, and disagreement among supervisors and glass workers."),
 ("wp_Early_glassmaking_in_the_United_States.txt", "Some historians believe the works operated sporadically until as late as 1661, while others believe it shut down earlier in 1642 or 1643."),
 ("wp_Early_glassmaking_in_the_United_States.txt", "In the Province of New Jersey, Wistar's glass works was the first to achieve large-scale, long-term success."),
 ("wp_Wistarburg_Glass_Works.txt", "Caspar Wistar founded the glass works company in 1739."),
 ("wp_Wistarburg_Glass_Works.txt", "Wistar's factory produced about 15,000 glass bottles per year made in the Waldglas style"),
 ("wp_Colonial_American_astronomy.txt", "It is believed that he built the first telescope and observatory in America. It was in this observatory on his farm that Rittenhouse observed the 1769 Venus Transit."),
 ("wp_Iwahashi_Zenbei.txt", "Zenbei researched some imported optical instruments, then in 1793, made his first telescope."),
 ("wp_Iwahashi_Zenbei.txt", "His descendants became telescope makers for four generations."),
 ("wp_Edo_period.txt", "At around 1721, the population of Japan was close to 30 million and the figure was only around 32 million around the Meiji Restoration around 150 years later."),
 ("wp_Edo_kiriko.txt", "Edo kiriko was first produced in 1834 by Kagaya Kyūbei, a glassware merchant in Edo."),
]
out = []
for f, q in Q:
    t = open(os.path.join(H, f), encoding="utf8").read()
    url = t.splitlines()[0].replace("URL: ", "")
    out.append({"url": url, "file": "sources/" + f, "quote": q, "found_in_fetched_text": q in t})
# 植民地の人口表（HTML を解析）
h = open(os.path.join(H, "wp_colonial_pop.html"), encoding="utf8").read()
pop = {}
for tb in re.findall(r"<table.*?</table>", h, re.S):
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S)
    cell = lambda r: [html.unescape(re.sub("<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)]
    hdr = cell(rows[0]); years = [c for c in hdr if c.isdigit() and len(c) == 4 and c not in ("1606",)]
    for r in rows[1:]:
        c = cell(r)
        if c and c[0].split("[")[0] in ("Total", "Massachusetts", "Virginia"):
            vals = c[-len(years):]
            for y, v in zip(years, vals):
                pop.setdefault(c[0].split("[")[0], {})[y] = int(v.replace(",", "")) if v.replace(",", "").isdigit() else None
out.append({"url": "https://en.wikipedia.org/wiki/List_of_colonial_and_pre-Federal_U.S._historical_population", "file": "sources/wp_colonial_pop.html",
            "table_values": pop, "found_in_fetched_text": True})
json.dump({"retrieved": "2026-10-18T14:40-15:05Z (MediaWiki API, en.wikipedia.org)", "excerpts": out}, open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(sum(not o["found_in_fetched_text"] for o in out), "not found;", json.dumps(pop))
