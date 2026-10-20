"""研究064の事例表（崩壊・人口の激減・弾圧の局面で、その文字を教える人が共同体の中にしかいなかった例）。
符号の規則（PREREGISTRATION.md）：
- breadth: narrow＝本文が読み書きを小さな書記・支配層・聖職者か行政の用途に限られたと書く／broad＝本文が普通の成員・子ども・多数が読み書きしたと書く／unknown＝どちらも書かれていない。
- depth: deep＝本文が中心の破壊・放棄、大きな人口の減少、強制移住、共同体が数百人以下になったことを書く／shallow＝国家が弱まって残ったと書く／suppression＝外からの禁止・閉鎖が主。
- y: 1＝本文がその文字が消えた・ほとんど誰も読めなかったと書く／0＝局面の後も1世代（25年）以上使われた／0.5＝本文の記述があいまい。y_alt はあいまいさの逆側の読み。
- years_after: 局面の後に使われ続けた年数（観察の終わりは置き換え・接触・現在）。本文の年・世紀からの丸めは【推測】。
- iso_world: 周りに文字の社会がなかった（Pitcairn 1790〜1808、Rapa Nui 1862〜）。
抜き出しは保存した本文の部分文字列で、verify_excerpts.py で照合する。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
C = [
 dict(id="LinearB_LBA", breadth="narrow", depth="deep", y=1, y_alt=1, years_after=0, iso_world=False,
      ex=[("wp_Linear_B.txt", "The application of Linear B texts appear to have been mostly confined to administrative contexts"),
          ("wp_Linear_B.txt", "a relatively small number of scribes have been detected: 45 in Pylos"),
          ("wp_Linear_B.txt", "Once the palaces were destroyed, the script disappeared."),
          ("wp_Greek_Dark_Ages.txt", "there were fewer, smaller settlements, which suggests widespread famine and depopulation")]),
 dict(id="Rongorongo_1862", breadth="narrow", depth="deep", y=1, y_alt=0.5, years_after=0, iso_world=True,
      ex=[("wp_Rongorongo.txt", "literacy was a privilege of the ruling families and priests, who were all kidnapped in the Peruvian slaving raids"),
          ("wp_Rongorongo.txt", "few if any of the Rapanui people remaining on the island in the 1870s could read the glyphs"),
          ("wp_Rongorongo.txt", "had reduced the Rapa Nui population to under two hundred by the 1870s")]),
 dict(id="Indus_1900BCE", breadth="unknown", depth="deep", y=1, y_alt=0.5, years_after=0, iso_world=False,
      ex=[("wp_Indus_script.txt", "was a period of fragmentation and localisation"),
          ("wp_Indus_script.txt", "the use of the script largely ceased as the use of inscribed seals ended"),
          ("wp_Indus_script.txt", "the use of the Indus script may have endured for a longer duration in other regions")]),
 dict(id="Maya_Classic", breadth="narrow", depth="deep", y=0, y_alt=0, years_after=650, iso_world=False,
      ex=[("wp_Maya_script.txt", "written by scribes—usually members of the Maya priesthood"),
          ("wp_Classic_Maya_collapse.txt", "violent collapse and abandonment of many southern lowlands city-states between the 7th and 9th centuries CE"),
          ("wp_Maya_script.txt", "Maya writing was in continuous use throughout Mesoamerica until the Spanish conquest of the Maya in the 16th and 17th centuries."),
          ("wp_Classic_Maya_collapse.txt", "with much less use of monumental hieroglyphic writing")]),
 dict(id="Hittite_Luwian", breadth="unknown", depth="deep", y=0, y_alt=0, years_after=500, iso_world=False,
      ex=[("wp_Hattusa.txt", "The city was destroyed, together with the Hittite state itself, around 1200 BC"),
          ("wp_Neo-Hittite_states.txt", "They arose following the collapse of the Hittite New Kingdom in the 12th century BCE"),
          ("wp_Anatolian_hieroglyphs.txt", "In the Neo-Hittite period, hieroglyphs may have also been used for \"everyday\" documents"),
          ("wp_Anatolian_hieroglyphs.txt", "In the early 7th century BC, the Anatolian hieroglyphic script, by then aged some 700 years, was marginalized by competing alphabetic scripts")]),
 dict(id="Cyprus_CM", breadth="unknown", depth="shallow", y=0, y_alt=0.5, years_after=750, iso_world=False,
      ex=[("wp_Cypriot_syllabary.txt", "used in Iron Age Cyprus, from about the 11th to the 4th centuries BCE, when it was replaced by the Greek alphabet"),
          ("wp_Cypro-Minoan_script.txt", "It is thought to be somehow related to the later Cypriot syllabary.")]),
 dict(id="Egypt_LBA", breadth="narrow", depth="shallow", y=0, y_alt=0, years_after=1500, iso_world=False,
      ex=[("wp_Literacy.txt", "probably less than 1% of the population was literate, as it was confined to a very small group"),
          ("wp_Late_Bronze_Age_collapse.txt", "states such as the Middle Assyrian Empire in Mesopotamia and the New Kingdom of Egypt survived in weakened forms"),
          ("wp_Egyptian_hieroglyphs.txt", "the last known inscription is from Philae, known as the Graffito of Esmet-Akhom, from 394")]),
 dict(id="Assyria_LBA", breadth="unknown", depth="shallow", y=0, y_alt=0, years_after=1250, iso_world=False,
      ex=[("wp_Late_Bronze_Age_collapse.txt", "The Middle Assyrian Empire survived intact throughout much of this period"),
          ("wp_Cuneiform.txt", "The latest firmly dateable tablet, from Uruk, dates to 79/80 AD.")]),
 dict(id="Samaritans", breadth="unknown", depth="deep", y=0, y_alt=0, years_after=1450, iso_world=False,
      ex=[("wp_Samaritans.txt", "their numbers were drastically reduced following a series of failed revolts and severe Byzantine repression in the 5th and 6th centuries CE"),
          ("wp_Samaritans.txt", "Their numbers reached a record low of around 100 people during the early modern period"),
          ("wp_Samaritans.txt", "both of which are written in the Samaritan script")]),
 dict(id="Pitcairn_1793", breadth="broad", depth="deep", y=0, y_alt=0, years_after=15, iso_world=True,
      ex=[("wp_John_Adams_(mutineer).txt", "Having taken effective control of the 19-member strong colony after the 1793 massacre"),
          ("wp_John_Adams_(mutineer).txt", "taught the children to read and write using the Bible"),
          ("wp_Mutiny_on_the_Bounty.txt", "Christian's group remained undiscovered on Pitcairn until 1808")]),
 dict(id="Mandaeans_2003", breadth="unknown", depth="deep", y=0, y_alt=0, years_after=19, iso_world=False,
      ex=[("wp_Mandaeans.txt", "which before the war numbered 60,000–70,000 people, collapsed"),
          ("wp_Mandaeans.txt", "By 2007, the population of Mandaeans in Iraq had fallen to approximately 5,000."),
          ("wp_Mandaic_alphabet.txt", "Classical Mandaic and its descendant Neo-Mandaic are still in limited use.")]),
 dict(id="Cherokee_1838", breadth="broad", depth="deep", y=0, y_alt=0, years_after=188, iso_world=False,
      ex=[("wp_Cherokee_syllabary.txt", "By 1825, the majority of Cherokees could read and write in their newly developed orthography."),
          ("wp_Trail_of_Tears.txt", "Approximately 4,000 Cherokees died in the ensuing trek to Oklahoma."),
          ("wp_Cherokee_syllabary.txt", "a bilingual paper printed in Indian Territory in the mid-19th century")]),
 dict(id="Kaifeng_1850s", breadth="unknown", depth="deep", y=0.5, y_alt=1, years_after=0, iso_world=False,
      ex=[("wp_Kaifeng_Jews.txt", "The dismantlement of the synagogue sometime between 1850 and 1866 led to the community's demise."),
          ("wp_Kaifeng_Jews.txt", "it was clearly copied by someone who did not understand them")]),
 dict(id="Kakure_1614", breadth="unknown", depth="suppression", y=1, y_alt=0.5, years_after=0, iso_world=False,
      ex=[("wp_Kakure_Kirishitan.txt", "The Bible and other parts of the liturgy were passed down orally, because printed works could be confiscated by authorities.")]),
 dict(id="Moriscos_1567", breadth="unknown", depth="suppression", y=0, y_alt=0, years_after=42, iso_world=False,
      ex=[("wp_Aljamiado.txt", "In 1567, Philip II of Spain issued a royal decree in Spain, which forced Moriscos to abandon using Arabic on all occasions"),
          ("wp_Aljamiado.txt", "some of the Moriscos kept their Islamic belief and traditions secretly, and this included the usage of Aljamiado"),
          ("wp_Morisco.txt", "between 1609 and 1614 they began to expel them systematically")]),
 dict(id="Egypt_temples_391", breadth="narrow", depth="suppression", y=1, y_alt=1, years_after=0, iso_world=False,
      ex=[("wp_Egyptian_hieroglyphs.txt", "the permanent closing of pagan temples across Roman Egypt  resulted in the loss of fluent readers and writers (called scribes) in hieroglyphs")]),
]
# 局面のない長い使用（drift の参考、数字には使わない）と literacy の率
OTHER = [
 dict(id="Dalecarlian", ex=[("wp_Dalecarlian_runes.txt", "in use in the Swedish province of Dalarna until the 20th century"),
                            ("wp_Dalecarlian_runes.txt", "The peasants in the community here, apart from using rune staves, still today write their names")]),
 dict(id="Szekely", ex=[("wp_Old_Hungarian_script.txt", "the script has remained in use by the Székely Magyars")]),
 dict(id="Nushu", ex=[("wp_Nüshu.txt", "used by ethnic Yao women for several centuries in Jiangyong")]),
 dict(id="literacy_rates", ex=[("wp_Literacy.txt", "estimated the global literacy rate at 87%"), ("wp_Literacy.txt", "where it was at 12% in 1820"),
                               ("wp_Literacy.txt", "level of literacy in classical Greece, which he argued was high for its time between 5% and 10%")]),
 dict(id="cherokee_speed", ex=[("wp_Cherokee_syllabary.txt", "After the syllabary was completed in the early 1820s, it achieved almost instantaneous popularity")]),
 dict(id="rongorongo_diffusion", ex=[("wp_Rongorongo.txt", "Known cases of the diffusion of writing, such as Sequoyah's invention of the Cherokee syllabary after observing English-language newspapers")]),
 dict(id="kakure_priests", ex=[("wp_Kakure_Kirishitan.txt", "retained many untranslated words from Latin, Portuguese, and Spanish")]),
]
json.dump({"cases": C, "other": OTHER}, open(os.path.join(H, "cases.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(C), "cases")
