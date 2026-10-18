"""本回開いた Wikipedia の本文（MediaWiki API の plain text、2026-10-18 18:4xZ 取得、fetch.py）から引用を照合して excerpts.json を作る。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
Q = [
 ("wp_Societal_collapse.txt", "Most never recovered, such as the Western and Eastern Roman Empires, the Maya civilization, and the Easter Island civilization."),
 ("wp_Societal_collapse.txt", "It coincided with and may have caused the decline and the fall of the Akkadian Empire in Mesopotamia and the Indus Valley Civilization."),
 ("wp_Societal_collapse.txt", "The Mississippian culture collapsed during this period. The Ancestral Puebloans left their established homes in the 12th and 13th centuries."),
 ("wp_Greek_Dark_Ages.txt", "On the Greek mainland, the Linear B script, used by Mycenaean bureaucrats to write the Greek language, ceased to be used. The later Greek alphabet did not develop until hundreds of years later, in the beginning of the Protohistoric Iron Age, c. 800 BC."),
 ("wp_Greek_Dark_Ages.txt", "With the collapse of the palatial centers, no more monumental stone buildings were built, and the practice of wall painting may have ceased. Writing in the Linear B script also ceased, and vital trade links were lost as towns and villages were abandoned."),
 ("wp_Greek_Dark_Ages.txt", "the Greeks relearned how to write, but using the alphabet of the Phoenicians instead of the Linear B script used by the Mycenaeans"),
 ("wp_Linear_B.txt", "The succeeding period, known as the Greek Dark Ages, provides no conclusive evidence of the use of writing."),
 ("wp_Linear_B.txt", "a relatively small number of scribes have been detected: 45 in Pylos (west coast of the Peloponnese, in Southern Greece) and 66 in Knossos (Crete)."),
 ("wp_Late_Bronze_Age_collapse.txt", "The deterioration of these governments interrupted trade routes and led to severely reduced literacy rates in much of this area."),
 ("wp_Late_Bronze_Age_collapse.txt", "Archaeometallurgical studies performed by various teams have also shown that trade in tin–a non-local and essential metal for bronze production–neither stopped nor decreased after 1200 BC"),
 ("wp_Late_Bronze_Age_collapse.txt", "Millek also demonstrated that trade with Egypt continued following 1200 BC."),
 ("wp_Late_Bronze_Age_collapse.txt", "Other cities, such as Athens, continued to be occupied, but with a more local sphere of influence, limited evidence of trade and an impoverished culture, from which it took centuries to recover."),
 ("wp_Hittites.txt", "the Hittites splintered into several small independent states, some of which survived until the 8th century BC"),
 ("wp_Anatolian_hieroglyphs.txt", "After some two centuries of sparse material, the hieroglyphs resume in the Early Iron Age, ca. 10th to 8th centuries BC."),
 ("wp_Indus_Valley_Civilisation.txt", "There are some circular stamp seals with geometric designs, but lacking the Indus script which characterised the mature phase of the civilisation. Script is rare and confined to potsherd inscriptions. There was also a decline in long-distance trade, although the local cultures show new innovations in faience and glass making, and carving of stone beads."),
 ("wp_Indus_Valley_Civilisation.txt", "Bet Dwarka was fortified and continued to have contacts with the Persian Gulf region, but there was a general decrease of long-distance trade."),
 ("wp_Indus_Valley_Civilisation.txt", "The small surplus produced in these small communities did not allow the development of trade, and the cities died out."),
 ("wp_Classic_Maya_collapse.txt", "Archaeologically, this decline is indicated by the cessation of monumental inscriptions and the reduction of large-scale architectural construction at the primary urban centers of the Classic Period."),
 ("wp_Classic_Maya_collapse.txt", "with much less use of monumental hieroglyphic writing"),
 ("wp_Classic_Maya_collapse.txt", "the longevity of the northern regions was likely facilitated by access to the coast, and thus trade routes."),
 ("wp_Fall_of_the_Western_Roman_Empire.txt", "they maintained literacy in Latin and other identifiably Roman traits for some time although they sank to a level of material development inferior even to their pre-Roman Iron Age ancestors."),
 ("wp_Fall_of_the_Western_Roman_Empire.txt", "The supply of coinage to the Diocese of Britannia ceases with Honorius."),
 ("wp_Early_Middle_Ages.txt", "the loss of the established cursus honorum led to the collapse of the schools and to a rise of illiteracy even among the leadership."),
 ("wp_Early_Middle_Ages.txt", "With the fall of the Western Roman Empire and with urban centres in decline, literacy and learning decreased in the West. De-urbanization reduced the scope of education, and by the 6th century teaching and learning moved to monastic and cathedral schools"),
 ("wp_Early_Middle_Ages.txt", "As the import of papyrus ceased and knowledge of Greek declined, the Latin West found itself cut off from Greek learning."),
 ("wp_Sub-Roman_Britain.txt", "In the sub-Roman period, building in stone gradually came to an end; buildings were constructed of less durable materials than during the Roman period."),
 ("wp_Sub-Roman_Britain.txt", "no newly minted coins are believed to have entered circulation in Britain after the very early 5th century."),
 ("wp_Sub-Roman_Britain.txt", "This uncovered rectangular structures and much Mediterranean pottery."),
 ("wp_Sub-Roman_Britain.txt", "The population of Britain may have decreased by between 1.5 and 3 million after the Roman period"),
 ("wp_Byzantine_Dark_Ages.txt", "Apart from fortifications —often quite hastily executed — almost all building activity ceased during this period."),
 ("wp_Ancestral_Puebloans.txt", "Late 14th- and 15th-century pottery from central Arizona, widely traded in the region"),
 ("wp_Easter_Island.txt", "By the time of European arrival in 1722, the island's population was estimated to be 2,000 to 3,000."),
 ("wp_Easter_Island.txt", "After the Spanish visit, from 1770 onward, a period of statue toppling, huri mo'ai, commenced."),
 ("wp_Norse_settlements_in_Greenland.txt", "The Norse settlements existed for about half a millennium before they were abandoned for reasons that are still not entirely clear."),
 ("wp_Norse_settlements_in_Greenland.txt", "The Kalmar Union was to prove disastrous for trade with Greenland because the remote outpost was of little interest to the Danish royal family and trade dried up."),
 ("wp_Olmecs.txt", "Whatever the cause, within a few hundred years of the abandonment of the last Olmec cities, successor cultures became firmly established."),
 ("wp_Tiwanaku.txt", "The society appeared to have collapsed around AD 1000; the reason for this is still open to debate."),
 ("wp_Angkor.txt", "During the course of the 15th century, nearly all of Angkor was abandoned, except for Angkor Wat, which remained a Buddhist shrine."),
 ("wp_First_Intermediate_Period_of_Egypt.txt", "record a decline in international relations and a general impoverishment in Egypt."),
 ("wp_Akkadian_Empire.txt", "material that is thought to be Akkadian continues to be in use into the Ur III period."),
 ("wp_Cahokia.txt", "The inhabitants left no written records beyond symbols on pottery, marine shell, copper, wood, and stone"),
]
out = []
for f, q in Q:
    t = open(os.path.join(H, f), encoding="utf8").read()
    out.append({"url": t.splitlines()[0].replace("URL: ", ""), "file": "sources/" + f, "quote": q, "found_in_fetched_text": q in t})
json.dump({"retrieved": "2026-10-18T18:4xZ (MediaWiki API, en.wikipedia.org)", "excerpts": out}, open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(sum(not o["found_in_fetched_text"] for o in out), "not found of", len(out))
for o in out:
    if not o["found_in_fetched_text"]: print(o["file"], o["quote"][:90])
