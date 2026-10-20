"""本回開いた本文から事例表 cases.json を作る（数は本文の値をそのまま写す。符号の規則はこの冒頭に書く）。
規則（資料を読んだ後・sim.py の前に固定。一人の判断で【推測】を含む）：
- map：言葉の説明なしには使ったことのない地図を、字を読む前の子ども・学校のない大人が使った実験。値は「最初に探した場所が正しい」割合。
  chance は選べる場所の数の逆数。sim.py は (p − chance)/(1 − chance) に直して使う（本物の探索では当て推量で当たらない）。
  n は本文で決まる参加者の数（t 値の自由度 + 1）。決まらなければ null（sim.py は 30 と置く）。
- landmark：同じ実験のうち、目標が目立つ目印（色と形が地図にも描かれた箱）だった試行の割合。
- picto：字が読めないか学校の短い人々を含む集団で、絵文字・安全の標識の意味を自由回答で答えさせた試験の平均の正答の割合（当て推量の補正なし）。
  low=1 は字が読めない・学校の短い人々（農業の働き手・12年以下の学校）を主に含む例。範囲だけの本文は中点を使う（Martí：48〜64 → 56、Rother：「50%以上が誤り」→ 0.50 を上限として使う）。
- hist：作った文化の外の人が説明なしに見た、文字のない古い地図。y=1 は本文が地図と認め、0.5 は「地図と思われる」「異論あり」、0 は作った人しか読めない。
- other：数には使わず質的に使う。
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
D = {
 "map": [
  {"id": "Dehaene2006_Munduruku", "p": 0.71, "chance": 1/3, "n": None, "pop": "Mundurukú children and adults, little or no schooling, no maps", "feedback": True, "landmark_mixed": True,
   "ex": [["hal_00206261.txt", "Overall, the participants_ success rate averaged 71%, which was well above the chance level"],
          ["hal_00206261.txt", "of 33.3% (across subjects, P G 0.0001). Performance did not differ between children (73.3%"],
          ["hal_00206261.txt", "correct) and adults (70.4% correct; both P G"],
          ["hal_00206261.txt", "do not typically possess maps or spontaneously draw pictures of their houses, villages,"],
          ["pmc_PMC3572713.txt", "each trial of the above experiment provided informative feedback to the participants"]]},
  {"id": "Shusterman2008_E1", "p": 0.70, "chance": 1/3, "n": 18, "pop": "US 4-year-olds (pre-literate), landmark present", "feedback": False, "landmark_mixed": True,
   "ex": [["pmc_PMC3572713.txt", "mean 70% correct, compared to 33% chance, t(17) = 16.46, p < .0001"],
          ["pmc_PMC3572713.txt", "Two experiments tested whether 4-year-old children extract and use geometric information in simple maps without task instruction or feedback."]]},
  {"id": "Shusterman2008_E2", "p": 0.53, "chance": 1/3, "n": 18, "pop": "US 4-year-olds (pre-literate), purely geometric map", "feedback": False, "landmark_mixed": False,
   "ex": [["pmc_PMC3572713.txt", "53% correct, compared with 33% chance, t(17) = 4.95, p = .0001"]]},
  {"id": "Izard2014_E1_fullmap", "p": 0.729, "chance": 0.5, "n": 32, "pop": "US 4-year-olds (pre-literate), full geometric map, two targets", "feedback": False, "landmark_mixed": False,
   "ex": [["pmc_PMC3751975.txt", "Children performed above chance on the full map trials (72.9%, t(31)=6.3, p<0.001)"]]}
 ],
 "landmark": [
  {"id": "Dehaene2006_landmark", "p": 0.871, "chance": 1/3, "n": None, "ex": [["hal_00206261.txt", "landmark location (87.1% correct) but less well"], ["hal_00206261.txt", "two unmarked locations (54.3% correct). An"]]},
  {"id": "Shusterman2008_E1_landmark", "p": 0.99, "chance": 1/3, "n": 18, "ex": [["pmc_PMC3572713.txt", "99% vs. 56% correct, t(17) = 11.151, p < .0001"]]}
 ],
 "picto": [
  {"id": "Dowse2023_isiXhosa", "p": 0.79, "n": 90, "low": 1, "pop": "isiXhosa adults, max 12 years schooling, locally developed medicine pictograms",
   "ex": [["pm_37927939.txt", "The mean pictogram comprehension score was 7.9/10"], ["pm_37927939.txt", "Ninety isiXhosa-speaking adults with a maximum of 12 years"]]},
  {"id": "Rother2008_farmworkers", "p": 0.50, "n": 115, "low": 1, "pop": "South African vineyard farm workers, FAO pesticide pictograms (upper bound)",
   "ex": [["pm_18755451.txt", "Overall, 50% or more of the study farm workers had"], ["pm_18755451.txt", "misleading, incorrect and critically confused interpretations of the label"]]},
  {"id": "Patel2023_farmers", "p": (0.876 + 0.694) / 2, "n": None, "low": 1, "pop": "Indian farmers, pesticide pictograms, mean of largest (87.6%) and smallest (69.4%) size",
   "ex": [["pm_37497563.txt", "15 mm × 15 mm (87.6%) and minimum comprehension for the size 7 mm × 7 mm"]]},
  {"id": "Gungor2023_fiberboard", "p": 0.666, "n": 139, "low": 0, "pop": "Turkish fiberboard workers, 22 ISO 7010 safety signs (elementary-to-associate group 65.31%)",
   "ex": [["pm_37313152.txt", "The mean comprehension score for 22 signs was"], ["pmc_PMC10258412.txt", "(65.31%, SD = 13.51%)"]]},
  {"id": "MartiFernandez2015_cleaners", "p": (94 + 95 + 5 * 56 + 7 + 7) / 9 / 100, "n": 118, "low": 0, "pop": "Spanish cleaning workers, 9 GHS pictograms (midpoint 56% for the 48-64% group)",
   "ex": [["pm_25877796.txt", "\"acute toxicity\" and \"flammable\", were described correctly by 94% and 95% of the"],
          ["pm_25877796.txt", "of correct answers ranged from 48% to 64%, whereas those for pictograms"],
          ["pm_25877796.txt", "\"oxidizing\" and \"compressed gas\" were interpreted correctly by only 7% of"]]}
 ],
 "hist": [
  {"id": "Catalhoyuk_mural", "y": 0.5, "ex": [["wp_Çatalhöyük.txt", "is frequently cited as the world's oldest map, and the first landscape painting. However, some archaeologists question this interpretation."]]},
  {"id": "Bedolina", "y": 1.0, "ex": [["wp_Bedolina_Map.txt", "It is known as one of the most ancient topographic maps, interpreted as a depiction of cultivated plots, mountain paths and villages."]]},
  {"id": "Mezhyrich_bone", "y": 0.5, "ex": [["wp_Mezhyrich,_Cherkasy_Oblast.txt", "a map inscribed onto a bone, presumably showing the area around the settlement"]]},
  {"id": "Marshall_stick_chart", "y": 0.0, "ex": [["wp_Marshall_Islands_stick_chart.txt", "the individual navigator who made the chart was the only person who could fully interpret and use it"],
                                                 ["wp_Marshall_Islands_stick_chart.txt", "'convinced' the navigators to share how the stick charts were used"]]}
 ],
 "other": [
  {"id": "Dowse2001_after_explanation", "note": "説明の後の3週間後の試験で ANSI 85% を満たしたのは地元の絵 20/23、USP 11/23。説明が入るので数に使わない。",
   "ex": [["pm_11687321.txt", "The correct meaning of each"], ["pm_11687321.txt", "the Local pictograms complied with the ANSI criterion of >/=85% comprehension"]]},
  {"id": "Dowse2005_with_text", "note": "文字＋絵のラベルで理解 95%、文字だけで 70%。文字があるので数に使わない。", "ex": [["pm_15950838.txt", "Average percentages for"]]},
  {"id": "Hermans2011_IVD_symbols", "note": "抽象的な記号（取り決め）の理解は低い。絵の記号のほうがよい。数に使わない（読める検査技師、取り決めの記号）。",
   "ex": [["pm_22047089.txt", "'Authorized representative in the"], ["pmc_PMC3215670.txt", "pure pictorial symbols, such as the umbrella of the 'Keep dry' symbol, generally scored better compared to abstract symbols"]]},
  {"id": "Pioneer_plaque_arrow", "note": "矢印は文化の取り決めで、別の文化の発見者に通じないかもしれないという批判。", "ex": [["wp_Pioneer_plaque.txt", "finders with a different cultural heritage may find the arrow symbol obscure in meaning"]]},
  {"id": "Sandia1993_levels", "note": "遠い将来への警告は物の形・絵・文字の多層にする提言。", "ex": [["wp_Long-term_nuclear_waste_warning_messages.txt", "providing pictographic information attempting to convey some details of the danger, and written explanations for those able to read it"]]},
  {"id": "Tupaia_map", "note": "Tupaia は自分の道の知識を Cook の羅針盤の取り決めに訳す地図の仕組みを作った（説明と言葉つき）。", "ex": [["wp_Tupaia_(navigator).txt", "Tupaia invented a cartographic system for Cook and his men"]]},
  {"id": "Ammassalik_maps", "note": "文字のない社会の海岸の木の地図。売られて欧州の博物館に入った（使い方の記述は本文になし）。", "ex": [["wp_Ammassalik_wooden_maps.txt", "carved, tactile maps of the Greenlandic coastlines"]]}
 ],
 "switch_rule_count": {"rule": "本回実際に開いた公開の数値（理解の割合）の出典の数。5未満なら B（173）へ切り替える。",
                       "sources": ["Dehaene2006", "Shusterman2008", "Izard2014", "Dowse2023", "Rother2008", "Patel2023", "Gungor2023", "MartiFernandez2015", "Hermans2011"]}
}
json.dump(D, open(os.path.join(H, "cases.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("map", len(D["map"]), "picto", len(D["picto"]), "hist", len(D["hist"]), "count", len(D["switch_rule_count"]["sources"]))
