"""本回開いた Wikipedia の本文（MediaWiki API の plain text、2026-10-18 22:3xZ 取得、fetch.py）から引用を照合して excerpts.json を作る。
Easter Island・Inughuit・Chatham Islands の本文は 049 の取得（20:4xZ）の写し（舟の喪失 B の根拠）。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
Q = [
 ("wp_Peopling_of_the_Americas.txt", "Around 25,000 years ago, they became isolated, forming a new genetic group linked to today's Indigenous populations"),
 ("wp_Peopling_of_the_Americas.txt", "With the rise of sea level after the Last Glacial Maximum (LGM), the Beringian land bridge was again submerged."),
 ("wp_Pre-Columbian_era.txt", "Whether or not early Polynesian–American exchanges occurred, no compelling human-genetic, archaeological, cultural, or linguistic legacy of such contact has turned up."),
 ("wp_Pre-Columbian_era.txt", "Estimates vary, but 30–50 million are often given, and 100 million by some estimates."),
 ("wp_Pre-Columbian_era.txt", "At their peak, the Valley of Mexico where the Aztec Empire presided, saw a population growth that included nearly one million people during the late Aztec period (1350–1519)."),
 ("wp_Pre-Columbian_era.txt", "In addition, they used native copper, silver, and gold for metalworking."),
 ("wp_Pre-Columbian_era.txt", "Some Woodland people continued to use spears and atlatls until the end of the period when they were replaced by bows and arrows."),
 ("wp_Inca_Empire.txt", "Most population estimates are in the range of 6 to 14 million."),
 ("wp_Aboriginal_Australians.txt", "This indicates that populations in Australia were isolated for a long time from the rest of Southeast Asia."),
 ("wp_Aboriginal_Australians.txt", "It is estimated that the Indigenous Australian population at the time of first European settlement numbered at least 314,500 people."),
 ("wp_Prehistory_of_Australia.txt", "Estimates of the Aboriginal population at the time of the first European settlement in 1788, range from 300,000 to 3 million."),
 ("wp_Prehistory_of_Australia.txt", "However, trade and intercourse between the separated lands continued across the newly formed Torres Strait"),
 ("wp_Prehistory_of_Australia.txt", "Macassan traders from Sulawesi regularly visited the coast of northern Australia to fish for trepang, an edible sea cucumber to trade with the Chinese since at least the early 18th century."),
 ("wp_Prehistory_of_Australia.txt", "Aboriginal people adopted dugout canoes and metal harpoon heads from the Indonesians"),
 ("wp_Prehistory_of_Australia.txt", "By 3,700 BP they had ceased to eat fish and use bone tools."),
 ("wp_History_of_Papua_New_Guinea.txt", "resulting populations remained relatively isolated from the rest of the world throughout prehistory."),
 ("wp_History_of_Papua_New_Guinea.txt", "Agriculture was independently developed in the New Guinea highlands around 7000 BC"),
 ("wp_History_of_Papua_New_Guinea.txt", "The highland valleys were explored by prospectors in the 1930s and were found to be inhabited by over a million people."),
 ("wp_History_of_Madagascar.txt", "During this early period, Madagascar served as an important transoceanic trading port for the East African coast"),
 ("wp_History_of_Madagascar.txt", "Beginning in the 10th or 11th century, Arab and traders worked their way down the Swahili coast in their dhows and established settlements on the west coast of Madagascar."),
 ("wp_History_of_Iceland.txt", "Icelanders began to trade for grain from continental Europe, which was an expensive proposition."),
 ("wp_History_of_Iceland.txt", "Church fast days increased demand for dried codfish, which was easily caught and prepared for export, and the cod trade became an important part of the economy."),
 ("wp_Jōmon_people.txt", "who became isolated from other mainland Asian groups about 22,000 to 25,000 years ago"),
 ("wp_Jōmon_period.txt", "steadily increasing contact with the Korean Peninsula eventually led to the establishment of Korean-type settlements in western Kyushu, beginning around 900 BC."),
 ("wp_Easter_Island.txt", "Loss of large trees meant that residents were no longer able to build seaworthy vessels, significantly diminishing their fishing abilities."),
 ("wp_Inughuit.txt", "Around this period, the Inughuit also lost the knowledge to build kayaks or umiaks, which further restricted travel and contact with other communities."),
 ("wp_Inughuit.txt", "The Baffin Inuit reintroduced some technologies lost to the Inughuit such as boats, leisters, and bows and arrows."),
 ("wp_Chatham_Islands.txt", "While the islands lacked suitable trees for building ocean-going craft for long voyages, the Moriori invented the waka kōrari"),
]
out = []
for fn, q in Q:
    t = open(os.path.join(H, fn), encoding="utf8").read()
    url = t.splitlines()[0].replace("URL: ", "")
    out.append({"url": url, "file": "sources/" + fn, "quote": q, "found_in_fetched_text": q in t})
json.dump({"retrieved": "2026-10-18T22:3xZ (MediaWiki API, en.wikipedia.org; Easter Island, Inughuit, Chatham Islands copied from 049, 20:4xZ)", "excerpts": out},
          open(os.path.join(H, "excerpts.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(sum(e["found_in_fetched_text"] for e in out), "/", len(out))
for i, e in enumerate(out):
    if not e["found_in_fetched_text"]: print("MISSING", i, e["quote"][:80])
