"""H84a の資料：国家を持たない社会の移動の距離（英語版 Wikipedia 本文の km・mile の記述）を、事前登録の規則で分類する。
分類：indiv＝個人（一行）の旅の距離、network＝物・人の広がりや網の大きさ（検定に入れない）、geo＝地形の幅など旅でないもの（除く）。
quote は wiki_texts.json の本文にそのまま含まれる（verify_excerpts.py で照合）。"""
import json
C = [
 # 個人の旅
 dict(id="Hopewell_ObsidianCliff", page="Hopewell tradition", cls="indiv", km=3200, flag="hypothetical",
      quote="The Ohio Hopewell may have made pilgrimages to Obsidian Cliff in what is today Yellowstone National Park, a journey which would have required walking 2,000 miles (3,200 kilometers) each way."),
 dict(id="Tahitian_castaways_Atiu", page="Polynesian navigation", cls="indiv", km=1609, flag="involuntary_failed",
      quote="a castaway group of Tahitians who had become lost at sea in a gale and blown 1000 miles away to the island of Atiu"),
 dict(id="Tasmanian_offshore", page="History of Indigenous Australians", cls="indiv", km=10, flag="routine",
      quote="journeyed up to 10 kilometres offshore to visit islands and hunt for seals and muttonbirds"),
 # 広がり・網（検定に入れない）
 dict(id="Lapita_site_range", page="Lapita culture", cls="network", km=4000, flag="range_of_sites",
      quote="More than 200 Lapita sites have since been uncovered, ranging more than 4,000 km"),
 dict(id="Lapita_spread_east", page="Polynesian navigation", cls="network", km=6000, flag="colonisation_extent",
      quote="the Lapita culture spread 6,000 km (3,700 mi) farther to the east from the Bismarck Archipelago"),
 dict(id="Tupaia_chart_radius", page="Polynesian navigation", cls="network", km=3200, flag="knowledge_radius",
      quote="drew a chart of the islands within a 2,000 miles (3,200 km) radius (to the north and west) of his home island of Ra'iatea"),
 dict(id="Chile_chicken", page="Polynesian navigation", cls="network", km=8000, flag="genetic_contact",
      quote="both over 5000 miles (8000 kilometers) away from Chile"),
 dict(id="Ochre_Mungo", page="History of Indigenous Australians", cls="network", km=250, flag="object_transport",
      quote="Ochre was transported 250 kilometres from the Barrier Range to Lake Mungo 40,000 years ago."),
 dict(id="Shell_beads", page="History of Indigenous Australians", cls="network", km=500, flag="object_transport",
      quote="Shells (for decorative beads) were transported 500 kilometres by 30,000 BP."),
 # 失敗した遠征（国家のある社会、別系列）
 dict(id="Kentoshi_loss", page="Japanese missions to Tang China", cls="failure", km=None, flag="loss_share_1/3",
      quote="Approximately one third of those who embarked from Japan did not survive to return home."),
]
NOKM = ["Kula ring", "Moka exchange", "Inuit", "San people", "Aboriginal Australians"]  # km・mile の旅の記述なし（Aboriginal Australians は Torres Strait の幅だけ＝geo）
MISSING = {"Hxaro": "Pauline Wiessner", "Trade in Aboriginal Australia": "History of Indigenous Australians"}
json.dump({"rule": "事前登録 H84a：個人の旅の距離の中央値 < 803 km なら支持、3件未満なら評価不能",
           "cases": C, "pages_without_trip_km": NOKM, "missing_pages_and_replacements": MISSING},
          open("cases.json", "w"), ensure_ascii=False, indent=1)
print(len(C), "cases")
