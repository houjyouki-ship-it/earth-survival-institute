"""研究062 の資料の数字と抜き出し（保存本文から逐語）。
editions：版ごとの刷り部数 N0（幅）と現存部数 n、刷った年と数えた年。
events：1つの施設・地域が戦乱・解散・火事にあったときに残った割合。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
W = "https://en.wikipedia.org/wiki/"
editions = [
 {"id": "Gutenberg_B42", "N0": [158, 185], "n": 49, "year": 1455, "count_year": 2026, "kind": "valued", "src": "wp_Gutenberg_Bible.txt", "url": W + "Gutenberg_Bible",
  "ex": ["Out of either 158 or 180 copies that were originally printed, 49 survive in at least substantial portion, 21 of them in entirety",
         "Scholars today think that examination of surviving copies suggests that somewhere between 160 and 185 copies were printed"]},
 {"id": "Shakespeare_FirstFolio", "N0": [700, 800], "n": 235, "year": 1623, "count_year": 2026, "kind": "valued", "src": "wp_First_Folio.txt", "url": W + "First_Folio",
  "ex": ["Of perhaps 750 copies printed, 235 are known to remain, most of which are kept in either public archives or private collections"]},
 {"id": "BayPsalmBook", "N0": [1500, 1900], "n": 11, "year": 1640, "count_year": 2026, "kind": "used", "src": "wp_Bay_Psalm_Book.txt", "url": W + "Bay_Psalm_Book",
  "ex": ["one of eleven known surviving copies of the first edition sold at auction", "An estimated 1,700 copies of the first edition were printed"]},
 {"id": "Copernicus_1543", "N0": [400, 500], "n": 276, "year": 1543, "count_year": 2004, "kind": "valued", "src": "wp_De_revolutionibus_orbium_coelestium.txt", "url": W + "De_revolutionibus_orbium_coelestium",
  "ex": ["with an initial print run of 400 failing to sell out", "His census included 276 copies of the first edition", "published in 2004 by Walker & Co"]},
 {"id": "Newton_Principia_1687", "N0": [400, 750], "n": 387, "year": 1687, "count_year": 2020, "kind": "valued", "src": "wp_Philosophiæ_Naturalis_Principia_Mathematica.txt", "url": W + "Philosophi%C3%A6_Naturalis_Principia_Mathematica",
  "ex": ["It has been estimated that as many as 750 copies of the first edition were printed by the Royal Society",
         "concludes that Macomber's earlier estimation of 500 copies is likely correct",
         "A 2020 study established a new census of owners of the first edition of the Principia, identifying 387 copies"]},
 {"id": "NurembergChronicle_Latin", "N0": [1400, 1500], "n": 1287, "year": 1493, "count_year": 2026, "kind": "valued", "src": "wp_Nuremberg_Chronicle.txt", "url": W + "Nuremberg_Chronicle",
  "ex": ["An estimated 1,400 to 1,500 Latin and 700 to 1,000 German copies were published", "Paul Needham's census documents 1,287 copies of the Latin edition and 343 of the German"]},
 {"id": "NurembergChronicle_German", "N0": [700, 1000], "n": 343, "year": 1493, "count_year": 2026, "kind": "valued", "src": "wp_Nuremberg_Chronicle.txt", "url": W + "Nuremberg_Chronicle", "ex": []},
 {"id": "MagnaCarta_1215", "N0": [13, 40], "n": 4, "year": 1215, "count_year": 2026, "kind": "manuscript_spread", "src": "wp_Magna_Carta.txt", "url": W + "Magna_Carta",
  "ex": ["At least thirteen original copies of the charter of 1215 were issued by the royal chancery during that year", "one of four surviving originals of the 1215 Magna Carta"]},
]
# 施設・地域の出来事。s＝その施設の所蔵のうち出来事を越えて残った割合、b＝地域の施設のうち焼かれた割合
events = [
 {"id": "Zaluski_1944", "type": "library_burned", "s": 31800 / 400000, "src": "wp_Załuski_Library.txt", "url": W + "Za%C5%82uski_Library",
  "ex": ["which grew to some 400,000 printed items, maps and manuscripts by the end of the 1780s", "Only 1800 manuscripts and 30,000 printed materials from the original library survived the war"]},
 {"id": "Worcester_Priory_dissolution", "type": "library_dispersed", "s": 6 / 600, "src": "wp_Dissolution_of_the_monasteries.txt", "url": W + "Dissolution_of_the_monasteries",
  "ex": ["Worcester Priory (now Worcester Cathedral) had 600 books at the time of the dissolution. Only six of them are known to have survived intact to the present day."]},
 {"id": "York_AugustinianFriars_dissolution", "type": "library_dispersed", "s": 3 / 646, "src": "wp_Dissolution_of_the_monasteries.txt", "url": W + "Dissolution_of_the_monasteries",
  "ex": ["At the abbey of the Augustinian Friars at York, a library of 646 volumes was destroyed, leaving only three known survivors"]},
 {"id": "LibraryOfCongress_1851_fire", "type": "library_burned", "s": 1 / 3, "src": "wp_Library_of_Congress.txt", "url": W + "Library_of_Congress",
  "ex": ["The collection grew slowly and suffered another major fire in 1851, which destroyed two-thirds of Jefferson's original books"]},
 {"id": "IraqNationalLibrary_2003_books", "type": "library_damaged", "s": 0.75, "src": "wp_Iraq_National_Library_and_Archive.txt", "url": W + "Iraq_National_Library_and_Archive",
  "ex": ["In total, an estimated 60 percent of its total archival materials, 25 percent of its books, newspapers, rare books, and most of its historical photographs and maps were destroyed"]},
 {"id": "Warsaw_1944_libraries", "type": "region_b", "b": [0.7, 0.8], "src": "wp_Destruction_of_Warsaw.txt", "url": W + "Destruction_of_Warsaw",
  "ex": ["During the German suppression of the Warsaw Uprising of 1944, around 70 to 80% of libraries were carefully burned by the Brandkommandos (burning detachments)",
         "14 libraries including the National Library"]},
 {"id": "England_dissolution_all_houses", "type": "region_b_note", "src": "wp_Dissolution_of_the_monasteries.txt", "url": W + "Dissolution_of_the_monasteries",
  "ex": ["Parts of 117 former monasteries (of over 800) survived (and mostly still remain) in use for parochial worship"]},
]
notes = [
 {"id": "Incunabula_total", "src": "wp_Incunable.txt", "url": W + "Incunable", "ex": ["Around 550,000 copies of around 27,500 different works have been preserved worldwide"]},
 {"id": "BirdsOfAmerica_no_N0", "src": "wp_The_Birds_of_America.txt", "url": W + "The_Birds_of_America", "ex": ["It is thought that no more than 120 complete sets exist today"]},
 {"id": "Fabrica_no_N0", "src": "wp_De_Humani_Corporis_Fabrica_Libri_Septem.txt", "url": W + "De_humani_corporis_fabrica", "ex": ["More than 700 copies survive from the 1543 and 1555 editions"]},
]
json.dump({"editions": editions, "events": events, "notes": notes}, open(os.path.join(H, "cases.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(editions), len(events), len(notes))
