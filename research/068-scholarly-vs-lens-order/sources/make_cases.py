"""研究068の事例表。符号の規則（一人の判断で【推測】を含む）：
- R1（文献学の段）：自分が読める（ほぼ読める）文字の古い文書・碑文を、体系的に集め・写し・比べ・誤りを正す営みが本文で年づけられる最初の年。
- R2（読めない文字への体系的な取り組み）：読めない文字の文書に体系的に取り組み始めた年（067 の onset と同じ定義）。
- L（読取装置の段）：望遠鏡・顕微鏡の級のレンズをその社会で作った最初の年。眼鏡・拡大鏡は T2 に足りないので L に数えない（版で扱う）。
  diffused=True はヨーロッパから知識が伝わってから作った例。L が本文で決まらない社会は、History of the telescope の「最古の望遠鏡は 1608」を使い L>1608 の下限（右打ち切り）とする。
- 曝露：R1 の後・L（または打ち切り）の前に、読めない外国の文字が見える場所にあった年数。
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
S = [
 {"id": "Europe_Italy", "region": "Europe", "R1": 1421, "R1_alt": 1440, "L": 1608, "L_alt": 1590, "L_censored": False, "diffused": False,
  "R2": 1616, "R2_alt": 1643, "eyeglasses": 1290,
  "exposure": {"script": "Egyptian (Rome obelisks, Giza visited 1435)", "start": 1421, "end_preL": 1608, "k_sys_preL": 0, "attempts_preL": ["Hypnerotomachia 1499 (pseudo-hieroglyphs)", "Horapollo printed 1505 (wrong key; 067)"]},
  "note": "R1 は Cyriacus が碑文に向かった 1421（alt は Valla の Donation の論 1439〜1440）。L は 1608 の望遠鏡（alt は Janssen の顕微鏡の主張 1590）。R2 は 067 の Palmyrene 1616（alt は Kircher 1643）。眼鏡は約1290で R1 より前。",
  "ex": [["wp_Cyriacus_of_Ancona.txt", "His vocation dates back to 1421, when scaffolding was erected around Arch of Trajan in his city for restoration."],
         ["wp_Cyriacus_of_Ancona.txt", "was the most enterprising and prolific recorder of Greek and Roman antiquities, particularly inscriptions, in the fifteenth century"],
         ["wp_Cyriacus_of_Ancona.txt", "In 1435, Cyriacus went to Egypt and reached the Giza Plateau"],
         ["wp_Lorenzo_Valla.txt", "Between 1439 and 1440, Valla wrote the essay, De falso credita et ementita Constantini Donatione declamatio"],
         ["wp_History_of_the_telescope.txt", "the earliest known telescope, which appeared in 1608 in the Netherlands"],
         ["wp_Timeline_of_microscope_technology.txt", "1590: earliest date of a claimed Hans Martens/Zacharias Janssen invention of the compound microscope (claim made in 1655)."],
         ["wp_Glasses.txt", "The first glasses were estimated to have been made in central Italy, most likely in Pisa or Florence, by about 1290"],
         ["wp_Flaminio_Obelisk.txt", "The obelisk was rediscovered and excavated in 1471 by the architect Leon Battista Alberti"],
         ["wp_Hypnerotomachia_Poliphili.txt", "The work was first published in 1499 in Venice by Aldus Manutius"]]},
 {"id": "Song_China", "region": "nonEurope", "R1": 1132, "R1_alt": None, "L": 1635, "L_alt": None, "L_censored": False, "diffused": True,
  "R2": None, "R2_alt": None, "eyeglasses": None,
  "exposure": {"script": "Khitan (epitaphs and monuments)", "start": 1165, "end_preL": 1635, "k_sys_preL": 0, "attempts_preL": ["Tao Zongyi 1376: five Khitan large characters with Chinese glosses (a record, not decipherment)"]},
  "note": "R1 は Zhao Mingcheng の Jin Shi Lu の刊行 1132（Ouyang Xiu の金石学はそれより前だが本文に年がない）。L は薄珏の千里鏡（崇禎8年 1635、湯若望の書の影響＝伝播）。孫雲球（1628〜1662）は望遠鏡と察微鏡（顕微鏡）を作った。Khitan の文字は Liao の滅亡（1125）の後も数十年使われた：失われた年 1165 は【推測】。",
  "ex": [["wp_Zhao_Mingcheng.txt", "in which he recorded the details of nearly 2,000 antique inscriptions, with carefully researched analysis about their histories"],
         ["wp_Zhao_Mingcheng.txt", "which was finally published in 1132"],
         ["wp_Zhao_Mingcheng.txt", "He scrutinized epigraphy scholarship of his time and corrected many mistakes and textual errors of received texts"],
         ["wp_Ouyang_Xiu.txt", "he played an important role in establishing the use of epigraphy as a historiographic technique"],
         ["wpzh_薄珏.txt", "約於該年二月開始爲之製造銅炮"],
         ["wpzh_薄珏.txt", "同時爲輔助使用火炮配備有“千里鏡”，用以偵察敵人遠近"],
         ["wpzh_薄珏.txt", "研究指出其望遠鏡製造則受到汤若望《交食历指》、《远镜说》等書间接甚或直接之影响。"],
         ["wpzh_孙云球.txt", "他把磨制成的凸透镜和凹透镜组合成“千里镜”，成为中国最早的望远镜之一。"],
         ["wp_History_of_the_telescope.txt", "By 1626 knowledge of the telescope had spread to China when German Jesuit and astronomer Johann Adam Schall von Bell published Yuan jing shuo"],
         ["wp_Glasses.txt", "The earliest mentions of glasses in China occur in the 15th century and those Chinese sources state that glasses were imported."],
         ["wp_Khitan_large_script.txt", "Both Khitan scripts continued to be in use to some extent by the Jurchen people for several decades after the fall of the Liao dynasty"],
         ["wp_Khitan_large_script.txt", "aside from five example Khitan large characters with Chinese glosses in Tao Zongyi's 1376 Important Matters in the History of Calligraphy"]]},
 {"id": "Edo_Japan", "region": "nonEurope", "R1": 1683, "R1_alt": 1690, "L": 1790, "L_alt": 1831, "L_censored": False, "diffused": True,
  "R2": None, "R2_alt": None, "eyeglasses": None, "exposure": None,
  "note": "R1 は契沖が万葉代匠記に着手した 1683 頃（alt は精選本 1690）。L は岩橋善兵衛が自ら磨いたレンズで望遠鏡を作った 1790 頃（渡来品の研究＝伝播、alt は国友一貫斎の反射望遠鏡 1831）。1676 までに 150 本以上の望遠鏡が輸入された。読めない外国の文字の曝露は本文で決まらず数えない。",
  "ex": [["wpja_万葉代匠記.txt", "契沖が『代匠記』に着手したのは天和3年（1683年）の頃であり"],
         ["wpja_契沖.txt", "実証的学問法を確立して国学の発展に寄与するなど、古典研究史上において時代を画するものであった。"],
         ["wpja_岩橋善兵衛.txt", "渡来品の研究を行い寛政2年（1790年）頃、望遠鏡を作り評判を得た。"],
         ["wpja_岩橋善兵衛.txt", "自らレンズを磨いて製作した望遠鏡は"],
         ["wp_Rangaku.txt", "Until 1676 more than 150 telescopes were brought to Nagasaki."],
         ["wp_Kunitomo_Ikkansai.txt", "built Japan's first reflective telescope in 1831"]]},
 {"id": "Abbasid", "region": "nonEurope", "R1": 813, "R1_alt": 833, "L": None, "L_alt": None, "L_censored": True, "L_lb": 1608, "diffused": None,
  "R2": None, "R2_alt": None, "eyeglasses": None,
  "exposure": {"script": "Egyptian (Giza and other monuments)", "start": 813, "end_preL": 1258, "end_alt": 1600, "k_sys_preL": 0, "attempts_preL": ["Dhul-Nun / pseudo-Ibn Wahshiyya 9th-10th c. (067)"]},
  "note": "R1 は al-Ma'mun の治世の始め 813（alt は終わり 833）：外国の文書を集める遠征。L は本文で決まらず、最古の望遠鏡 1608 を下限とする右打ち切り。読む石（約1000年から）と Ibn al-Haytham の光学はあるが T2 の級でない。曝露の終わりは House of Wisdom が壊された 1258（alt は 1600）。",
  "ex": [["wp_House_of_Wisdom.txt", "al-Ma'mun would send expeditions of scholars from the House of Wisdom to collect texts from foreign lands"],
         ["wp_House_of_Wisdom.txt", "Under the sponsorship of Caliph al-Ma'mun (r. 813–833)"],
         ["wp_House_of_Wisdom.txt", "He also promoted Egyptology and personally participated in excavations at the pyramids of Giza."],
         ["wp_House_of_Wisdom.txt", "the House of Wisdom was destroyed by Hulagu's army during the Siege of Baghdad"],
         ["wp_Reading_stone.txt", "the regular use of reading stones did not begin until around 1000 AD"]]},
 {"id": "Byzantine", "region": "nonEurope", "R1": 850, "R1_alt": 880, "L": None, "L_alt": None, "L_censored": True, "L_lb": 1608, "diffused": None,
  "R2": None, "R2_alt": None, "eyeglasses": None,
  "exposure": {"script": "Egyptian (Obelisk of Theodosius, Hippodrome)", "start": 850, "end_preL": 1453, "k_sys_preL": 0, "attempts_preL": []},
  "note": "R1 は Photios の Bibliotheca（9世紀、279 冊の書評）：年は 850 と置いた【推測】（alt 880）。L は右打ち切り（1608 が下限）。Constantinople の Hippodrome に 390 から hieroglyph の obelisk が立っていた。曝露の終わりは 1453（Constantinople の陥落）。取り組みの記述は本回の資料にない（無いことの証拠は弱い）。",
  "ex": [["wp_Ottoman_Empire.txt", "The Ottomans ended the Byzantine Empire with the conquest of Constantinople in 1453 by Mehmed II."], ["wp_Bibliotheca_(Photius).txt", "was a ninth-century work of Byzantine Patriarch of Constantinople Photius, dedicated to his brother and composed of 279 reviews of books which he had read."],
         ["wp_Obelisk_of_Theodosius.txt", "remained in Alexandria until 390; when Theodosius I (379–395 AD) had it transported to Constantinople and put up on the spina of the Hippodrome there. It remained in place through the Byzantine period"]]},
]
OTHER = [
 {"id": "Qing_oracle_bones", "use": "qualitative", "note": "考証学の社会は 1899 に新しく手にした甲骨の文字を数か月で古い漢字と認め、1903 に 34 字を読んだ。自分の文字の祖先なので R2 に数えない（067 の new_access と同じ型）。",
  "ex": [["wp_Wang_Yirong.txt", "best known as the first to recognize that the symbols inscribed on oracle bones were an early form of Chinese writing in 1899"],
         ["wp_Oracle_bone_script.txt", "published the first collection of 1,058 rubbings entitled Tieyun Canggui"],
         ["wp_Kaozheng.txt", "was a Chinese school of thought emphasizing philology"]]},
 {"id": "Joseon_Korea", "use": "excluded", "note": "金正喜の金石学（北漢山の真興王巡狩碑の考証）は R1 だが、朝鮮で望遠鏡級のレンズを作った年が本文にない（眼鏡は17世紀に清の商人から）。",
  "ex": [["wp_Kim_Chŏnghŭi.txt", "He also pursued research by visiting and studying the inscriptions on ancient stele."],
         ["wpko_안경.txt", "17세기에 안경이 청나라 상인을 통해 국내에 유입되었다"]]},
]
json.dump({"societies": S, "other": OTHER}, open(os.path.join(H, "cases.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(S), "societies")
