"""wiki_texts.json・wiki_coords*.json から cases.json を作る（符号化の規則はここに固定）。
規則：
- 事例 = 本文で「文書（経典・写本・書物・文字）を得るために出かけた／取り寄せた」ことと、出発地と目的地が本文で分かるもの。
- 出発地は本文に出る出発の拠点（なければその人の拠点の町）。目的地は本文で文書を得た場所（複数なら本文で最も中心の場所）。
- 座標は Wikipedia の地名の頁の座標。地域名しかない目的地（Byzantium、Kashmir、Gandhara）は代表の町の座標で置き、proxy=True【推測】。
- 小地域は国連 M49（070 の iso_regions.csv と同じ表）。大陸は 070 の continent_of_region。
- 年は本文に出る出発と帰着の年（なければ None）。
"""
import json, math
T = json.load(open("wiki_texts.json", encoding="utf8"))
C = {}
for f in ("wiki_coords.json", "wiki_coords2.json", "wiki_coords3.json"):
    for k, v in json.load(open(f, encoding="utf8")).items():
        c = v["query"]["pages"][0].get("coordinates")
        if c: C[k] = (c[0]["lat"], c[0]["lon"])
SUB = {"Xi'an": "Eastern Asia", "Luoyang": "Eastern Asia", "Lhasa": "Eastern Asia", "Tholing_Monastery": "Eastern Asia", "Lhodrak_County": "Eastern Asia",
       "Mount_Hiei": "Eastern Asia", "Kyoto": "Eastern Asia", "Ningbo": "Eastern Asia",
       "Pataliputra": "Southern Asia", "Nalanda_district": "Southern Asia", "Peshawar": "Southern Asia", "Srinagar": "Southern Asia",
       "Baghdad": "Western Asia", "Constantinople": "Western Asia",
       "Sunderland": "Northern Europe", "Rome": "Southern Europe", "Cremona": "Southern Europe", "Toledo,_Spain": "Southern Europe",
       "Verona": "Southern Europe", "Savona": "Southern Europe", "Venice": "Southern Europe",
       "Avignon": "Western Europe", "Konstanz": "Western Europe", "Cluny_Abbey": "Western Europe", "Langres": "Western Europe",
       "Abbey_of_Saint_Gall": "Western Europe", "Bad_Hersfeld": "Western Europe", "Fulda": "Western Europe", "Ferrières-en-Gâtinais": "Western Europe"}
CONT = {"Eastern Asia": "Asia", "Southern Asia": "Asia", "Western Asia": "Asia", "Northern Europe": "Europe", "Southern Europe": "Europe", "Western Europe": "Europe"}
CASES = [
 dict(id="Faxian", page="Faxian", o="Xi'an", d="Pataliputra", y0=399, y1=412, mode="travel", proxy=False,
      q=["He set out from Chang'an, the capital of the Buddhist Later Qin dynasty, along with four others to locate sacred Buddhist texts",
         "Faxian learned Sanskrit, and collected Indian literature from Pataliputra (Patna), Oddiyana, and Taxila in Gandhara.",
         "The journey and return took from 399 to 412, with 10 years spent in India."]),
 dict(id="Song_Yun", page="Song_Yun", o="Luoyang", d="Peshawar", y0=518, y1=522, mode="travel", proxy=True,
      q=["left the Wei capital Luoyang on foot in 518 and returned in the winter of 522 with 170 Buddhist scriptures.",
         "They stayed two years in Udyana and Gandhara"]),
 dict(id="Xuanzang", page="Xuanzang", o="Xi'an", d="Nalanda_district", y0=629, y1=645, mode="travel", proxy=False,
      q=["the travelogue of his journey to the Indian subcontinent in 629–645, his efforts to bring at least 657 Indian texts to China",
         "He stayed and studied at Nalanda."]),
 dict(id="Thonmi_Sambhota", page="Thonmi_Sambhota", o="Lhasa", d="Srinagar", y0=632, y1=None, mode="travel_script", proxy=True,
      q=["to India in 632 with other youths, to learn the alphabet. The pattern chosen was the script of Kashmir.",
         "was prepared at Kukarmaru Palace in Lhasa"]),
 dict(id="Yijing", page="Yijing_(monk)", o="Luoyang", d="Nalanda_district", y0=671, y1=695, mode="travel", proxy=False,
      q=["In 671 he began his journey to India", "came to the famed monastery of Nalanda where he remained for the next ten years.",
         "finally returned to China at Luoyang", "He brought back some 400 Buddhist texts translated into Chinese."]),
 dict(id="Benedict_Biscop", page="Benedict_Biscop", o="Sunderland", d="Rome", y0=679, y1=None, mode="travel", proxy=True,
      q=["His second trip to Rome had focused on book-buying.", "Benedict made his fifth and final trip to Rome in 679 to bring back books for a library"]),
 dict(id="Kukai", page="Kūkai", o="Kyoto", d="Xi'an", y0=804, y1=806, mode="travel", proxy=False,
      q=["In 804, Kūkai took part in a government-sponsored expedition to China", "the party was asked to proceed to the capital of Chang'an (present day Xi'an)",
         "Kūkai arrived back in Japan in 806", "He also arrived with a large number of texts, many of which were new to Japan"]),
 dict(id="Saicho", page="Saichō", o="Mount_Hiei", d="Ningbo", y0=804, y1=805, mode="travel", proxy=False,
      q=["The emperor granted a petition by Saichō to journey to China to further study Tiantai doctrine in China and bring back more texts.",
         "Saichō and his party returned to Ningbo and after compiling further bibliographies, boarded the ship back for Japan"]),
 dict(id="Ennin", page="Ennin", o="Mount_Hiei", d="Xi'an", y0=838, y1=847, mode="travel", proxy=False,
      q=["His journey to Tang China (838–847)", "Later he went to Chang'an (Japanese: Chōan), then the capital of China",
         "Upon returning to Japan in 847, Ennin brought back numerous Buddhist texts"]),
 dict(id="Hunayn", page="Hunayn_ibn_Ishaq", o="Baghdad", d="Constantinople", y0=None, y1=None, mode="travel", proxy=True,
      q=["The caliph also gave Hunayn the opportunity to travel to Byzantium in search of additional manuscripts"]),
 dict(id="Rinchen_Zangpo", page="Rinchen_Zangpo", o="Tholing_Monastery", d="Srinagar", y0=None, y1=None, mode="travel", proxy=True,
      q=["to Kashmir and other Buddhist centres to study and bring back Buddhist teachings to Western Tibet"]),
 dict(id="Marpa", page="Marpa_Lotsawa", o="Lhodrak_County", d="Nalanda_district", y0=None, y1=None, dur=12, mode="travel", proxy=True,
      q=["Born as Marpa Chökyi Lodrö, in Lhodrak Chukhyer in the southern part of Tibet", "Pullahari, near Nalanda University, where Naropa taught.",
         "Marpa spent twelve years studying with Naropa", "Upon his return to Tibet, Marpa spent many years translating Buddhist scriptures"]),
 dict(id="Gerard_of_Cremona", page="Gerard_of_Cremona", o="Cremona", d="Toledo,_Spain", y0=None, y1=None, mode="travel", proxy=False,
      q=["Dissatisfied with the philosophies of his Italian teachers, Gerard went to Toledo.", "obtained the Arabic books in the libraries at Toledo."]),
 dict(id="Petrarch", page="Petrarch", o="Avignon", d="Verona", y0=1345, y1=None, mode="travel", proxy=False,
      q=["During his travels, he collected crumbling Latin manuscripts", "in the Chapter Library (Biblioteca Capitolare) of Verona Cathedral.",
         "He spent much of his early life at Avignon"]),
 dict(id="Poggio", page="Poggio_Bracciolini", o="Konstanz", d="Cluny_Abbey", y0=1415, y1=1417, mode="travel", proxy=False,
      extra_d=["Abbey_of_Saint_Gall", "Langres", "Bad_Hersfeld", "Fulda", "Reichenau_Island"],
      q=["when his duties called him to the Council of Constance in 1414, he employed his forced leisure in exploring the libraries of Swiss and Swabian abbeys.",
         "In 1415 at Cluny he found Cicero's complete great forensic orations", "At Langres in the summer of 1417 he discovered"]),
 dict(id="Guarino", page="Guarino_da_Verona", o="Verona", d="Constantinople", y0=None, y1=None, dur=5, mode="travel", proxy=False,
      q=["He was born in Verona", "studied Greek language and literature in Constantinople", "for five years he was the pupil",
         "he had with him two cases of precious manuscripts of ancient Greek texts"]),
 dict(id="Aurispa", page="Giovanni_Aurispa", o="Savona", d="Constantinople", y0=1418, y1=1423, mode="travel", proxy=False,
      q=["He returned to Italy in 1414, setting in Savona", "In 1418, Aurispa visited Constantinople, where he remained for some years, perfecting his knowledge of Greek and searching for manuscripts.",
         "On 15 December 1423 Aurispa arrived in Venice", "he brought back 238 manuscripts."]),
 dict(id="Filelfo", page="Francesco_Filelfo", o="Venice", d="Constantinople", y0=1420, y1=1427, mode="travel", proxy=False,
      q=["Immediately after his arrival in Constantinople at the end of 1420", "formed a large collection of Greek manuscripts.",
         "When Filelfo arrived at Venice with his family in 1427"]),
 dict(id="Lupus_Servatus", page="Lupus_Servatus", o="Ferrières-en-Gâtinais", d="Fulda", y0=None, y1=None, mode="letter", proxy=False,
      q=["from the library of the monastery of Fulda", "Lupus made a tireless quest of manuscripts of classic authors"]),
]
EXCLUDED = {"Adelard_of_Bath": "目的地が本文で一つに決まらない（Sicily・Antioch・possibly Spain）",
            "House_of_Wisdom": "al-Ma'mun の遠征は目的地が本文にない（foreign lands）",
            "Kibi_no_Makibi": "持ち帰ったものに文書が本文で明示されない", "Basilios_Bessarion": "旅でなく寄贈", "Atiśa": "教える人が招かれた例（Rinchen Zangpo と重複）",
            "Ceolfrith": "Benedict Biscop の 679 年の旅と同じ旅", "Hyecho": "本文に文書の取得がない", "Japanese_missions_to_Tang_China": "個別の旅でなく使節の総体（損失率だけ使う）"}
KENTOSHI_LOSS_Q = "Approximately one third of those who embarked from Japan did not survive to return home."
def hav(a, b):
    (la1, lo1), (la2, lo2) = a, b; R = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2); dp = p2 - p1; dl = math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))
out = []
for c in CASES:
    so, sd = SUB[c["o"]], SUB[c["d"]]
    dist = hav(C[c["o"]], C[c["d"]])
    if "extra_d" in c: dist_max = max([dist] + [hav(C[c["o"]], C[x]) for x in c["extra_d"]])
    else: dist_max = dist
    lit = "sub" if so == sd else ("cont" if CONT[so] == CONT[sd] else "global")
    adj = "cont" if (lit == "global" and dist < 2000) else lit
    dur = (c["y1"] - c["y0"]) if (c.get("y0") and c.get("y1")) else c.get("dur")
    out.append(dict(id=c["id"], page=c["page"], origin=c["o"], dest=c["d"], origin_sub=so, dest_sub=sd, dist_km=round(dist, 1), dist_max_km=round(dist_max, 1),
                    reach_literal=lit, reach_adj=adj, y0=c.get("y0"), y1=c.get("y1"), dur_years=dur, mode=c["mode"], proxy=c["proxy"], quotes=c["q"]))
J = {"rule": "本回開いて確認した、出発地と目的地が本文で分かる文書を求めた旅の事例 >= 8 なら A（185）を続ける",
     "cases": out, "excluded": EXCLUDED, "kentoshi_loss_quote": KENTOSHI_LOSS_Q, "coords": {k: list(v) for k, v in C.items()},
     "subregion": SUB, "continent": CONT, "adj_rule": "M49 の大陸をまたぐが 2,000 km 未満の旅は cont に数える（070 の『世界に届く』は全大陸の単位に届くことなので）【推測】"}
json.dump(J, open("cases.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print(len(out), Counter(x["reach_literal"] for x in out), Counter(x["reach_adj"] for x in out))
for x in out: print(f'{x["id"]:18s} {x["origin_sub"]:16s}->{x["dest_sub"]:16s} {x["dist_km"]:7.0f} km {x["reach_literal"]:6s} {x["reach_adj"]:5s} dur={x["dur_years"]}')
