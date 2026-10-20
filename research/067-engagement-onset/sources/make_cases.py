"""研究067の事例表（open_problems 175：崩壊の後の社会が古い文書に学者の速さで取り組み始める時点。174：教える人なしに文書だけから読み始めた非学者の例）。
符号の規則（PREREGISTRATION.md に同じ。資料を読んだ後・sim.py の前に固定。年の丸めと「取り組み」の判定は一人の判断で【推測】）：
- corpora（読めなくなった文字の文書の集まり）：
  loss＝読める人が消えた年（本文の最後の文・「読めた」とされる最後の世紀の中点）。
  access＝読める社会がその文書を見られた最初の年。地上に見えていた碑文・貨幣は loss と同じ（visible=True）。
          埋もれていた・知られていなかった文書は、本文の発見・初出版の年（visible=False）。
  attempt_any＝本文が「読もうとした」「解こうとした」と書く最初の年（誤った試みも含む）。世紀だけのときは中点。
  onset＝体系的な取り組みの始まり：本文が、読むための写し・分析の出版・解読の試みの出版・研究の開始を書く最初の年。
         本文に年がなく解読の年しかないときは None（onset_ub に解読の年を上限として入れる）。
  onset_alt＝別の読み（Kircher は 1628 の関心の始まり、Brahmi は 1834 の進展の再開、Indus は 1920年代の発掘の資料、など）。
  decipher＝本文の解読の年（なければ None）。
- near174（教える人なしに文書から読み始めた例の候補）：strict＝非学者・教える人なし・知らない文字・文書だけの4つを満たすか（1/0）。
抜き出しは保存した本文の部分文字列で、verify_excerpts.py で照合する。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
C = [
 dict(id="Egyptian", visible=True, loss=394, access=394, attempt_any=850, onset=1643, onset_alt=1628, decipher=1822,
      note="attempt_any は Dhul-Nun（9世紀）の中点850【推測】。Horapollo（4〜5世紀）は失われる途中の知識なので数えない。",
      ex=[("wp_Egyptian_hieroglyphs.txt", "the last known inscription is from Philae, known as the Graffito of Esmet-Akhom, from 394"),
          ("wp_Egyptian_hieroglyphs.txt", "Early attempts at decipherment were made by some such as Dhul-Nun al-Misri and pseudo-Ibn Wahshiyya (9th and 10th century, respectively)."),
          ("wp_Athanasius_Kircher.txt", "In his Lingua Aegyptiaca Restituta (1643)"),
          ("wp_Athanasius_Kircher.txt", "Kircher's interest in Egyptology began in 1628"),
          ("wp_Decipherment_of_ancient_Egyptian_scripts.txt", "Arab scholars were aware of the connection between Coptic and the ancient Egyptian language"),
          ("wp_Horapollo.txt", "The first printed edition of the text appeared in 1505")]),
 dict(id="Achaemenid_cuneiform", visible=True, loss=80, access=80, attempt_any=None, onset=1711, onset_alt=1621, decipher=1802,
      note="Persepolis・Behistun の三言語の碑文。loss は最後の楔形文字の文書 79/80。中世のペルシアの地理学者は遺跡を Jamshid のものとした（読もうとした記述なし）。onset_alt は della Valle の訪問 1621。",
      ex=[("wp_Cuneiform.txt", "The last known cuneiform inscription, an astronomical text, was written in 79/80 AD."),
          ("wp_Persepolis.txt", "Early Muslim geographers, including al-Masudi, Istakhri, al-Muqaddasi and Ibn Hawqal, described the ruins but attributed them to the legendary world-king Jamshid"),
          ("wp_Old_Persian_cuneiform.txt", "Proper attempts at deciphering Old Persian cuneiform started with faithful copies of cuneiform inscriptions, which first became available in 1711 when duplicates of Darius's inscriptions were published by Jean Chardin."),
          ("wp_Pietro_Della_Valle.txt", "In October 1621 he left Isfahan, visited Persepolis and Shiraz"),
          ("wp_Behistun_inscription.txt", "In 1598, Englishman Robert Sherley saw the inscription"),
          ("wp_Georg_Friedrich_Grotefend.txt", "His first discovery was communicated to the Royal Society of Göttingen in 1802")]),
 dict(id="Ashokan_Brahmi", visible=True, loss=400, access=400, attempt_any=None, onset=1785, onset_alt=1834, decipher=1837,
      note="loss は「4世紀まで読めた」から400。Feroz Shah（1351–1388 在位、歴史に関心）は未解読の柱を飾りとして都に移した（読もうとした記述なし）。onset は Wilkins の後期 Brahmi の解読 1785。",
      ex=[("wp_Edicts_of_Ashoka.txt", "remained readable down to the 4th century CE during the Gupta period"),
          ("wp_Delhi-Topra_pillar.txt", "till then undeciphered – and decided to shift them to his palatial Feruzabad palace in Delhi as \"totemic embellishments\""),
          ("wp_Delhi-Topra_pillar.txt", "was a keen historian"),
          ("wp_Brahmi_script.txt", "Inscriptions of the 6th century CE in late Brahmi were already deciphered in 1785 by Charles Wilkins"),
          ("wp_Brahmi_script.txt", "Progress resumed in 1834 with the publication of proper facsimiles of the inscriptions on the Allahabad pillar of Ashoka"),
          ("wp_Delhi-Topra_pillar.txt", "which was deciphered by James Princep, a renowned scholar in Indian antiquarian studies in 1837")]),
 dict(id="Palmyrene", visible=True, loss=274, access=274, attempt_any=None, onset=1616, onset_alt=1753, decipher=1754,
      note="onset は碑文の例の最初の印刷 1616、onset_alt は正確な二言語の写しの出版 1753。",
      ex=[("wp_Palmyrene_alphabet.txt", "The last surviving inscription dates to 274 CE"),
          ("wp_Jean-Jacques_Barthélemy.txt", "Examples of Palmyrene inscriptions were printed as far back as 1616, but accurate copies of Palmyrene/Greek bilingual inscriptions were not available until 1753"),
          ("wp_Palmyrene_alphabet.txt", "The Palmyrene alphabet was deciphered in 1754, literally overnight")]),
 dict(id="Meroitic", visible=True, loss=430, access=430, attempt_any=None, onset=None, onset_ub=1909, onset_alt=None, decipher=1909,
      note="loss は最後の碑文 410/450 の中点。onset の年は本文になく、解読 1909 を上限とする。",
      ex=[("wp_Meroitic_script.txt", "has recently been re-dated to AD 410/ 450 of the 5th century"),
          ("wp_Meroitic_script.txt", "The script was deciphered in 1909 by Francis Llewellyn Griffith")]),
 dict(id="Kharosthi", visible=True, loss=350, access=350, attempt_any=None, onset=None, onset_ub=1835, onset_alt=None, decipher=1835,
      note="loss は「2〜3世紀の後に捨てられた」と「故地で5世紀ごろ消えた」の間の 350【推測】。貨幣と碑文が地上にあったとした【推測】。",
      ex=[("wp_Kharosthi.txt", "Kharosthi seems to have been abandoned after the 2nd–3rd century AD"),
          ("wp_Kharosthi.txt", "Kharosthi remained in use until it died out in its homeland around the 5th century CE"),
          ("wp_Kharosthi.txt", "The Kharosthi script was deciphered separately almost concomitantly by James Prinsep (in 1835")]),
 dict(id="Maya", visible=True, loss=1575, access=1575, attempt_any=1566, onset=1862, onset_alt=None, decipher=1952,
      note="loss は 063 の 1550〜1600 の中点。Landa の表（1566）は読める書き手がいた時の鍵で、失われた後の取り組みでない。onset は Brasseur の Landa の出版 1862（065 の保存本文）。",
      ex=[("wp_Maya_script.txt", "he derived what he believed to be a Maya \"alphabet\" (the so-called de Landa alphabet)"),
          ("wp_Maya_script.txt", "In 1952 Knorozov published the paper"),
          ("wp_Diego_de_Landa.txt", "In 1862, French cleric Charles Etienne Brasseur de Bourbourg published the manuscript")]),
 dict(id="LinearB", visible=False, loss=-1200, access=1900, attempt_any=1900, onset=1900, onset_alt=None, decipher=1952,
      note="access は Knossos の発掘 1900（065 の保存本文）。Evans は続く数十年に解読を試みた。",
      ex=[("wp_Linear_B.txt", "excavation on the hill began on 23 March 1900"),
          ("wp_Michael_Ventris.txt", "Evans attempted to decipher both in the following decades, with little success."),
          ("wp_Linear_B.txt", "Linear B was deciphered in 1952 by English architect and self-taught linguist Michael Ventris")]),
 dict(id="LinearA", visible=False, loss=-1450, access=1900, attempt_any=1900, onset=1900, onset_alt=None, decipher=None,
      ex=[("wp_Linear_A.txt", "Since its modern rediscovery in 1900, no texts in the script have yet been deciphered.")]),
 dict(id="Indus", visible=False, loss=-1900, access=1875, attempt_any=1925, onset=1925, onset_alt=1925, access_alt=1922, decipher=None,
      note="access は最初の印章の出版 1875、access_alt は Mohenjo-daro の発掘 1922。onset は「1920年代から」の中点 1925【推測】。",
      ex=[("wp_Indus_script.txt", "The first publication of a seal with Harappan symbols dates to 1875"),
          ("wp_Indus_script.txt", "Over 100 (mutually exclusive) attempts at decipherment have been published since the 1920s"),
          ("wp_Mohenjo-daro.txt", "carried out by the Government of India between the years 1922 and 1927")]),
 dict(id="Orkhon_OldTurkic", visible=False, loss=None, access=1889, attempt_any=None, onset=None, onset_ub=1893, onset_alt=None, decipher=1893,
      note="loss の年は本文で決まらない。モンゴルの地上の碑だが、読める社会が見ていたかは本文にない。access は Yadrintsev の調査 1889。",
      ex=[("wp_Orkhon_inscriptions.txt", "The inscriptions were discovered by Nikolai Yadrintsev's expedition in 1889, published by Vasily Radlov."),
          ("wp_Orkhon_inscriptions.txt", "deciphered by the Danish philologist Vilhelm Thomsen in 1893")]),
 dict(id="Hittite", visible=False, loss=None, access=1906, attempt_any=None, onset=None, onset_ub=1915, onset_alt=None, access_alt=1893, decipher=1915,
      note="access は「1906年から回収」、access_alt は Chantre の断片 1893–94。",
      ex=[("wp_Hittite_language.txt", "Hittite texts, which were recovered from 1906 onwards"),
          ("wp_Hattusa.txt", "Significantly Chantre  discovered some fragments of clay tablets inscribed with cuneiform."),
          ("wp_Hittite_language.txt", "He presented his argument that the language is Indo-European in a paper published in 1915")]),
 dict(id="Tocharian", visible=False, loss=750, access=1900, attempt_any=None, onset=1907, onset_alt=None, decipher=None,
      note="loss は「6〜8世紀の写本」の後の 750【推測】。access は「20世紀初め」の 1900【推測】。onset は Sieg と Müller の 1907。",
      ex=[("wp_Tocharian_languages.txt", "brought to light fragments of manuscripts in an unknown language, dating from the 6th to 8th centuries AD"),
          ("wp_Tocharian_languages.txt", "In 1907, Emil Sieg and Friedrich W.")]),
]
N174 = [
 dict(id="Kaitai_Shinsho_1771", strict=0, nonscholar=1, no_teacher=0.5, unknown_script=0, docs_only=0.5, years=3, success=1,
      note="医者の仲間が図と解剖を照らしながらオランダ語の本を訳した。Maeno は少し読めた。辞書なし。図が鍵の代わり。",
      ex=[("wp_Kaitai_Shinsho.txt", "At first, Sugita and Nakagawa could not actually read Dutch; even with Maeno who could, their Dutch vocabulary was inadequate."),
          ("wp_Kaitai_Shinsho.txt", "naturally there were no dictionaries at the time"),
          ("wp_Kaitai_Shinsho.txt", "Sugita, marveling at the accuracy of the work while comparing it by eye with his autopsies, proposed to Maeno that it be translated."),
          ("wp_Kaitai_Shinsho.txt", "On 4 March 1771"), ("wp_Kaitai_Shinsho.txt", "In 1774, Kaitai Shinsho was published.")]),
 dict(id="Grotefend_1802", strict=0, nonscholar=0.5, no_teacher=1, unknown_script=1, docs_only=1, years=2, success=0.5,
      note="学校の教師が飲み仲間との賭けで解いた。大学で学んだ人。王の名の同定は正しく、音価は不完全。",
      ex=[("wp_Georg_Friedrich_Grotefend.txt", "Having a taste for puzzles, he made a bet with drinking friends around 1800 that he could decipher at least part of the Persepolis inscriptions."),
          ("wp_Georg_Friedrich_Grotefend.txt", "an assistant mastership in the Göttingen gymnasium in 1797")]),
 dict(id="Ventris_1952", strict=0, nonscholar=0.5, no_teacher=1, unknown_script=1, docs_only=0.5, years=16, success=1,
      note="建築家で独学の言語学者。Kober の研究に基づく。1936 の展示から 1952。",
      ex=[("wp_Linear_B.txt", "deciphered in 1952 by English architect and self-taught linguist Michael Ventris based on the research of the American classicist Alice Kober"),
          ("wp_Michael_Ventris.txt", "In 1936, Evans hosted an exhibition on Cretan archaeology at Burlington House")]),
 dict(id="Horapollo_false_key", strict=0, nonscholar=0, no_teacher=1, unknown_script=1, docs_only=1, years=None, success=0,
      note="鍵とされた本が 1505 から印刷され、Kircher まで象徴の読みに導いた（誤った鍵）。",
      ex=[("wp_Horapollo.txt", "inherited the preconception of the hieroglyphs as a magical, symbolic, ideographic script")]),
 dict(id="Feroz_Shah_pillars", strict=0, nonscholar=1, no_teacher=1, unknown_script=1, docs_only=1, years=None, success=0,
      note="歴史に関心のある支配者が未解読の碑文の柱を飾りとして運んだ。読もうとした記述なし。",
      ex=[("wp_Delhi-Topra_pillar.txt", "till then undeciphered – and decided to shift them to his palatial Feruzabad palace in Delhi as \"totemic embellishments\"")]),
]
json.dump({"corpora": C, "near174": N174}, open(os.path.join(H, "cases.json"), "w", encoding="utf8"), indent=1, ensure_ascii=False)
print(len(C), "corpora", len(N174), "near174")
