"""本回開いた Wikipedia の本文（MediaWiki API の plain text、2026-10-18 20:4xZ 取得、fetch.py）から引用を照合して excerpts.json を作る。
Easter Island と Norse Greenland の本文は 048 で取得したものの写し。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
Q = [
 ("wp_Aboriginal_Tasmanians.txt", "Aboriginal Tasmanians remained isolated from mainland Aboriginal populations for approximately 8,000 years, until European exploration in the late 18th and early 19th centuries."),
 ("wp_Aboriginal_Tasmanians.txt", "Estimates of the combined population of Aboriginal people in Tasmania before European arrival range from 3,000 to 15,000."),
 ("wp_Aboriginal_Tasmanians.txt", "Approximately 4,000 years ago, Aboriginal Tasmanians largely dropped scaled fish from their diet"),
 ("wp_Aboriginal_Tasmanians.txt", "The significance of the disappearance of bone tools (believed to have been primarily used for fishing) and fish in the diet is heavily debated."),
 ("wp_Aboriginal_Tasmanians.txt", "small Aboriginal Tasmanian populations living on the highlands of what are now King Island and the Furneaux Group became stranded by rising waters after the formation of Bass Strait and later died out."),
 ("wp_Aboriginal_Tasmanians.txt", "despite claims that the native Tasmanians had \"lost\" the ability to make fire"),
 ("wp_Moriori.txt", "They are Polynesians who came from the New Zealand mainland by at least about 1500 AD, and possibly around the mid-15th century."),
 ("wp_Moriori.txt", "The Chathams proved unsuitable for the cultivation of most crops known to Polynesians, and the Moriori adopted a hunter-gatherer lifestyle."),
 ("wp_Moriori.txt", "The Moriori lived in isolation from the outside world until 1791, when the first Europeans arrived."),
 ("wp_Moriori.txt", "The local population was estimated at 1,600 in the mid-1830s"),
 ("wp_Chatham_Islands.txt", "While the islands lacked suitable trees for building ocean-going craft for long voyages, the Moriori invented the waka kōrari"),
 ("wp_Inughuit.txt", "Around the 17th century, climate change cooled the northwest areas of Greenland, which cut off the Inughuit from other Inuit and regions."),
 ("wp_Inughuit.txt", "Around this period, the Inughuit also lost the knowledge to build kayaks or umiaks, which further restricted travel and contact with other communities."),
 ("wp_Inughuit.txt", "The Baffin Inuit reintroduced some technologies lost to the Inughuit such as boats, leisters, and bows and arrows."),
 ("wp_Inughuit.txt", "Before 1880, their population was estimated to be between 100 and 200 people."),
 ("wp_Inughuit.txt", "From the 11th to 16th century, the peoples of the region had extensive trading contacts with other Inuit from western and southern Greenland along with Ellesmere Island."),
 ("wp_Dorset_culture.txt", "Possibly, due to a shift from terrestrial to aquatic hunting, or the deaths of those who possessed the knowledge, the bow and arrow became lost to the Dorset."),
 ("wp_Dorset_culture.txt", "Both the Pre-Dorset and Thule (Inuit) had drills."),
 ("wp_Pitcairn_Islands.txt", "the earliest settlers of the Pitcairn Islands were Polynesians who occupied Pitcairn and Henderson for several centuries until the islands were abandoned"),
 ("wp_Henderson_Island_(Pitcairn_Islands).txt", "The reasons for the inhabitants' disappearance remain uncertain, but they may relate to an exhaustion of resources and to the disappearance around the same time of the Polynesians on Pitc"),
 ("wp_Mangareva.txt", "However, excessive logging by the islanders between the 10th and 15th centuries resulted in deforestation of the island, with disastrous results for both its environment and its economy"),
 ("wp_Mangareva.txt", "The island was once heavily forested and supported a large population that traded with other islands via canoes."),
 ("wp_Kangaroo_Island.txt", "The people disappeared from the archaeological record sometime after the land became an island"),
 ("wp_Flinders_Island.txt", "A population remained until about 4,500 years ago, succumbing to thirst and hunger following an acute El Niño climate shift."),
 ("wp_Norfolk_Island.txt", "Ultimately, the relative isolation of the island, and its poor horticultural environment, were not favourable to long-term settlement."),
 ("wp_Raoul_Island.txt", "although the occupation of the islands was relatively brief"),
 ("wp_Nīhoa.txt", "The abandonment of Nīhoa may be tied to deforestation although this has not been proven"),
 ("wp_Hawaii.txt", "The islands' history is marked by a slow, steady growth in population and the size of the chiefdoms, which grew to encompass whole islands."),
 ("wp_Hawaii.txt", "declining from an estimated 300,000 in the 1770s"),
 ("wp_Hawaii.txt", "the impact of these foreign visitors on the hitherto isolated Hawaiian Islands"),
 ("wp_Māori_people.txt", "Over several centuries in isolation, these settlers developed a distinct culture"),
 ("wp_Māori_people.txt", "The Māori population around the late 18th century was estimated by James Cook at 100,000. Historian Michael King suggests a slightly higher figure of 110,000 is more likely."),
 ("wp_Māori_people.txt", "The Classic period is characterised by finely made pounamu (greenstone) weapons and ornaments, elaborately carved war canoes and wharenui (meeting houses)."),
 ("wp_Andamanese_peoples.txt", "appear to have lived in substantial isolation for thousands of years"),
 ("wp_Andamanese_peoples.txt", "an estimated 7,000 Andamanese remained"),
 ("wp_Andamanese_peoples.txt", "The Andamanese knew of no method for making fire in the nineteenth century."),
 ("wp_Sentinelese.txt", "the 1971 census of India estimated the population at around 82, and the 1981 census at 100. A 1986 expedition recorded a count of 98."),
 ("wp_Sentinelese.txt", "They use spears with bows and arrows to hunt terrestrial wildlife"),
 ("wp_Guanches.txt", "Their population seems to have lived in relative obscurity and isolation up until the time of Castilian conquest"),
 ("wp_Guanches.txt", "this account suggests that the Guanche had sporadic contacts with populations from the mainland"),
 ("wp_Tikopia.txt", "Before European contact, there was frequent canoe voyaging between the islands"),
 ("wp_Tikopia.txt", "The population of Tikopia is about 1,200"),
 ("wp_Easter_Island.txt", "the current best estimate for colonization is in the 12th century CE"),
 ("wp_Easter_Island.txt", "By the time of European arrival in 1722, the island's population was estimated to be 2,000 to 3,000."),
 ("wp_Norse_settlements_in_Greenland.txt", "Estimates put the combined population of the settlements at their height between 2,000 and 10,000, with recent estimates trending toward the lower figure."),
]
out = []
for f, q in Q:
    t = open(os.path.join(H, f), encoding="utf8").read()
    out.append({"url": t.splitlines()[0].replace("URL: ", ""), "file": "sources/" + f, "quote": q, "found_in_fetched_text": q in t})
json.dump({"retrieved": "2026-10-18T20:4xZ (MediaWiki API, en.wikipedia.org; Easter Island and Norse Greenland copied from 048, 18:4xZ)", "excerpts": out},
          open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(sum(not o["found_in_fetched_text"] for o in out), "not found of", len(out))
for i, o in enumerate(out):
    if not o["found_in_fetched_text"]: print(i, o["file"], o["quote"][:90])
for i, o in enumerate(out): print(i, o["file"][8:40])
