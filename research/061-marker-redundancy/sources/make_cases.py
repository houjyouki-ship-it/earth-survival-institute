"""研究061 の事例と抜き出し（保存本文から逐語）。年は西暦（紀元前は負）。gap は位置の知識が失われた年数の区間 (L, U]、U=None は右打ち切り。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
W = "https://en.wikipedia.org/wiki/"
cases = [
 {"id": "NorseGreenlandEast", "class": "remote", "ceased": 1410, "L": 0, "U": 1721 - 1410, "src": "wp_Norse_settlements_in_Greenland.txt", "url": W + "Norse_settlements_in_Greenland",
  "ex": ["A wedding took place there on 14 September 1408.", "The guests came from Iceland in 1408 and returned in 1410."]},
 {"id": "NorseGreenlandEast_Egede", "class": "note", "src": "wp_Hans_Egede.txt", "url": W + "Hans_Egede",
  "ex": ["One was that the lost Eastern Settlement lay on Greenland's east coast, though it was later shown to have been located among the fjords of the island's far southwest."]},
 {"id": "Vinland", "class": "remote", "ceased": 1000, "L": 0, "U": 1880 - 1000, "src": "wp_Vinland.txt", "url": W + "Vinland",
  "ex": ["According to the 1880 Sephton translation of the saga, Rafn and other Danish scholars placed Kjalarnes at Cape Cod, Straumfjörð at Buzzards Bay, Massachusetts, and Straumsey at Martha's Vineyard.",
         "L'Anse Aux Meadows was a small and short-lived encampment; perhaps it was primarily used for timber-gathering forays and boat repair, rather than permanent settlements like those in Greenland."]},
 {"id": "Vinland_LAM", "class": "note", "src": "wp_LAnse_aux_Meadows.txt", "url": W + "L%27Anse_aux_Meadows",
  "ex": ["L'Anse aux Meadows is an archaeological site, first excavated in the 1960s, of a Norse settlement dating to approximately 1,000 years ago.",
         "The common hypothesis before the Ingstads was that Vinland could not be north of the Massachusetts coast, the northern limit of wild grapes"]},
 {"id": "Helike", "class": "inhabited", "ceased": -373, "L": 174 + 373, "U": 1826 + 373, "src": "wp_Helike.txt", "url": W + "Helike",
  "ex": ["was an ancient Greek polis or city-state that was submerged by a tsunami in the winter of 373 BC.",
         "Around 174 AD, the traveler Pausanias visited a coastal site still called Helike, located seven kilometres southeast of Aigio, and reported that the walls of the ancient city were still visible underwater",
         "Later the site silted over and the location was lost to memory.",
         "In 1826, French diplomat and archaeologist François Pouqueville, who wrote the Voyage en Grèce; in 1851 Ernst Curtius the German archaeologist and historian who speculated about its location"]},
 {"id": "Nineveh", "class": "inhabited", "ceased": -612, "L": 1761 + 612, "U": None, "src": "wp_Nineveh.txt", "url": W + "Nineveh",
  "ex": ["The location of Nineveh was known, to some, continuously through the Middle Ages.", "Carsten Niebuhr recorded its location during the 1761–1767 Danish expedition."]},
 {"id": "MachuPicchu", "class": "inhabited", "ceased": 1572, "L": 1911 - 1572, "U": None, "src": "wp_Machu_Picchu.txt", "url": W + "Machu_Picchu",
  "ex": ["The site was abandoned roughly a century later, likely during the Spanish conquest."]},
 {"id": "Pompeii", "class": "inhabited", "ceased": 79, "L": 0, "U": 1592 - 79, "src": "wp_Pompeii.txt", "url": W + "Pompeii",
  "ex": ["The area became known as La Civita (the city) due to the features in the ground.",
         "Francesco Bianchini pointed out the true meaning, and he was supported by Giuseppe Macrini, who in 1693 excavated some walls and wrote that Pompeii lay beneath La Civita."]},
 {"id": "MohenjoDaro", "class": "inhabited", "ceased": -1700, "L": 0, "U": 1920 + 1700, "src": "wp_Mohenjo-daro.txt", "url": W + "Mohenjo-daro",
  "ex": ["1700 BCE, Mohenjo-daro was abandoned."]},
]
markers = [
 {"id": "GeorgiaGuidestones", "src": "wp_Georgia_Guidestones.txt", "url": W + "Georgia_Guidestones", "years_stood": 42, "cost_usd2025_min": 400000,
  "ex": ["The Georgia Guidestones was  a granite monument that stood in Elbert County, Georgia, United States, from 1980 to 2022.",
         "The total cost of the project was not revealed, but it was over US$100,000 (equivalent to $400,000 in 2025).",
         "On July 6, 2022, an explosive device was detonated at the site, destroying the Swahili/Hindi language slab and causing significant damage to the capstone."]},
 {"id": "SeedVault", "src": "wp_Svalbard_Global_Seed_Vault.txt", "url": W + "Svalbard_Global_Seed_Vault", "construction_usd": 8.8e6,
  "ex": ["The Norwegian government entirely funded the Seed Vault's approximately 45 million kr (US$8.8 million in 2008) construction cost.", "In 2019, the Seed Vault cost about 2.4 million kr (US$282,000) to maintain."]},
 {"id": "Behistun", "src": "wp_Behistun_inscription.txt", "url": W + "Behistun_inscription",
  "ex": ["Authored by Darius the Great sometime between his coronation as king of the Persian Empire in the summer of 522 BC and his death in autumn of 486 BC"]},
 {"id": "MOM_token", "src": "wp_Memory_of_Mankind.txt", "url": W + "Memory_of_Mankind",
  "ex": ["On its surface, the token indicates a place on Earth with a precision of some ten meters.",
         "An instruction for a rite is attached to this [非公開] owners are supposed to gather every 50 years to commemorate and decide whether humanity still knows the content of MOM and if extensions are needed."]},
 {"id": "WIPP_markers", "src": "wp_Waste_Isolation_Pilot_Plant.txt", "url": W + "Waste_Isolation_Pilot_Plant",
  "ex": ["will include an outer perimeter of thirty-two 25-foot-tall (7.6 m) granite pillars built in a four-mile (6 km) square."]},
 {"id": "WIPP_PIC_archives", "src": "pic.txt", "url": "https://www.wipp.energy.gov/library/CRA/BaselineTool/Documents/Appendices/PIC.PDF",
  "ex": ["should be widely distributed in a number of locations", "Detailed maps describing the exact location of the", "The 53 Federal Regional Depository Libraries",
         "However, the most likely", "strategy for long term protection of the information is", "through widespread distribution.",
         "rate of 0.1% per lo,000 hours. If this is a constant", "remaining after about 2700 years."]},
 {"id": "WIPP_PMIP", "src": "pmip.txt", "url": "https://www.wipp.energy.gov/library/PermanentMarkersImplementationPlan.pdf",
  "ex": ["Permanent Markers Implementation Plan", "DOE/WIPP 04-3302"]},
]
json.dump({"cases": cases, "markers": markers}, open(os.path.join(H, "cases.json"), "w"), indent=1, ensure_ascii=False)
print(len(cases), len(markers))
