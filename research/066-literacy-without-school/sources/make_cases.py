"""研究066の事例表（open_problems 172：現代の学校で教わる読み書きが学校なしで続くか、064 の重み w）。
符号の規則（PREREGISTRATION.md に同じ。資料を読んだ後・sim.py の前に固定。符号そのものは一人の判断で【推測】）：
- episodes（国・地域の学校の仕組みが無い・禁じられた・崩れた局面で、普通の人々の読みが次の世代に続いたか）：
  y＝1 本文が、国の学校なしで（家庭・教会・宗教の学校・共同体・秘密の学校で）広い読み書きが続いた・広まったと書く／
  0.5 秘密・家庭の学校はあったが本文が識字の低下も書く、または対象の世代の識字が本文で決まらない／
  0 本文が、その世代が読み書きできずに育った、または読み書きが聖職者など少数に狭まったと書く。
  y_alt＝より厳しい読み（数の裏付けがない継続は 0.5、低下の記述があれば 0）。
  group＝built（国の学校なしで広い読みが作られ保たれた。崩壊ではない）／collapse（あった学校・国が閉じた・禁じた・崩れた）。
  depth＝deep（国そのものが崩れた）／regime（政権が学校をなくした）／ban（特定の人・言葉の学校を禁じた）／none（崩壊でない）。
  relig＝普通の人が自分で宗教の文を読むことを求める・支える宗教の場が本文にある（1）、一部（0.5）、ない・宗教が抑えられた（0）。
- conflicts（世界銀行の年齢別識字率と突き合わせる内戦・戦争の年）：本文の年。進行中は end=None。
抜き出しは保存した本文の部分文字列で、verify_excerpts.py で照合する。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
E = [
 dict(id="Sweden_1686_1800", group="built", depth="none", relig=1, y=1, y_alt=1, years=114,
      ex=[("wp_Literacy.txt", "The 1686 church law (kyrkolagen) of the Kingdom of Sweden (modern Sweden, Finland, Latvia, and Estonia) made literacy compulsory, and by 1800, the percent of people able to read was close to 100%."),
          ("wp_Literacy.txt", "This was directly dependent on the need to read religious texts in the Lutheran faith in Sweden and Finland"),
          ("wp_Literacy.txt", "Graff also says, using the example of Sweden, that mass literacy can be achieved without formal schooling or instruction in writing."),
          ("wp_Swedish_Church_Law_1686.txt", "this law resulted in the need for every village and rural community to organise basic schooling for their children")]),
 dict(id="Iceland_premodern", group="built", depth="none", relig=1, y=1, y_alt=1, years=None,
      ex=[("wp_Literacy.txt", "Iceland was an exception, as it achieved widespread literacy without formal schooling, libraries, or printed books via informal tuition by religious leaders and peasant teachers.")]),
 dict(id="Finland_18_19c", group="built", depth="none", relig=1, y=1, y_alt=1, years=None,
      ex=[("wp_Education_in_Finland.txt", "Literacy is a key part of Lutheranism, the state and majority religion of Finland, as Christians are supposed to be able to read the Bible in their native language."),
          ("wp_Education_in_Finland.txt", "Literacy reached over 50% in the late 18th century and 80–90% in the mid-19th century."),
          ("wp_Education_in_Finland.txt", "Where there were no schools in a municipality, reading was taught in traveling schools (kiertokoulu)."),
          ("wp_Education_in_Finland.txt", "Confirmation, a rite of transition to adulthood, is only permissible for the literate")]),
 dict(id="Ireland_hedge_1723_1782", group="collapse", depth="ban", relig=0.5, y=1, y_alt=1, years=59,
      ex=[("wp_Hedge_school.txt", "Catholics and Presbyterians set up secret and illegal schools that met in private homes."),
          ("wp_Hedge_school.txt", "While all Catholic education was forbidden under the penal laws from 1723 to 1782, no hedge teachers are known to have been prosecuted."),
          ("wp_Hedge_school.txt", "Historians agree that the hedge schools provided education, occasionally at a very high level, for up to 400,000 students by the mid-1820s."),
          ("wp_Hedge_school.txt", "Reading was often taught using chapbooks, sold at village fairs")]),
 dict(id="Lithuania_press_ban_1864_1904", group="collapse", depth="ban", relig=1, y=0.5, y_alt=0, years=40,
      ex=[("wp_Lithuanian_press_ban.txt", "In the wake of the ruling, parish schools were closed."),
          ("wp_Lithuanian_press_ban.txt", "Many students were schooled at home or at the clandestine Lithuanian schools, although this practice also resulted in sanctions."),
          ("wp_Lithuanian_press_ban.txt", "The subsequent ban is thought to have contributed to illiteracy in 19th-century Lithuania."),
          ("wp_Lithuanian_press_ban.txt", "petitioned the government to exempt prayer books from the ban")]),
 dict(id="Afghanistan_girls_1996_2001", group="collapse", depth="ban", relig=0.5, y=0.5, y_alt=0, years=5,
      ex=[("wp_Women_in_Afghanistan.txt", "Women were denied formal education and were usually forced to stay at home."),
          ("wp_Women_in_Afghanistan.txt", "News about these secret schools spread through word of mouth from woman to woman."),
          ("wp_Women_in_Afghanistan.txt", "At these schools, young girls were taught basic literary skills, numeracy skills, and various other subjects such as biology, chemistry, English, Quranic Studies, cooking, sewing, and knitting."),
          ("wp_Golden_Needle_Sewing_School.txt", "Because women above the age of eight were not allowed to be educated")]),
 dict(id="Somalia_1991", group="collapse", depth="deep", relig=1, y=1, y_alt=0.5, years=35,
      ex=[("wp_Education_in_Somalia.txt", "Following the outbreak of the Somali Civil War in 1991, the task of running schools in Somalia was initially taken up by community education committees established in 94% of the local schools."),
          ("wp_Education_in_Somalia.txt", "Qur'anic schools (also known as duqsi) remain the basic system of traditional religious instruction in Somalia."),
          ("wp_Education_in_Somalia.txt", "Known as the most stable local, non-formal system of education")]),
 dict(id="Cambodia_KhmerRouge_1975_1979", group="collapse", depth="regime", relig=0, y=0, y_alt=0, years=4,
      ex=[("wp_Education_in_Cambodia.txt", "Schools were closed."),
          ("wp_Education_in_Cambodia.txt", "Only 50 of the 725 university instructors, 207 of the 2,300 secondary school teachers, and 2,717 of the 21,311 primary school teachers survived."),
          ("wp_Education_in_Cambodia.txt", "An entire generation of Cambodian children grew up illiterate."),
          ("wp_Education_in_Cambodia.txt", "Illiteracy had climbed to more than 40 percent, and most young people under the age of 14 lacked any basic education."),
          ("wp_Khmer_Rouge.txt", "In practice, primary schools were not set up in many areas")]),
 dict(id="Roman_West_fall", group="collapse", depth="deep", relig=0, y=0, y_alt=0, years=None,
      ex=[("wp_Literacy.txt", "Literacy was widespread under Roman rule but became very rare, limited almost entirely to churchmen, after the fall of the Western Roman Empire."),
          ("wp_Literacy.txt", "During the Dark Ages, the highest percentage of literacy was found among the clergy and monks")]),
]
C = [
 dict(iso="LBR", start=1989, end=2003, ex=[("wp_First_Liberian_Civil_War.txt", "lasted between 1989 and 1997"), ("wp_Second_Liberian_Civil_War.txt", "lasted from 1999 to 2003")]),
 dict(iso="RWA", start=1990, end=1994, ex=[("wp_Rwandan_Civil_War.txt", "from 1 October 1990 to 18 July 1994")]),
 dict(iso="TJK", start=1992, end=1997, ex=[("wp_Tajikistani_Civil_War.txt", "began in May 1992 and ended in June 1997")]),
 dict(iso="BIH", start=1992, end=1995, ex=[("wp_Bosnian_War.txt", "from 1992 until 1995")]),
 dict(iso="TCD", start=2005, end=2010, ex=[("wp_Chadian_Civil_War_(2005–2010).txt", "The Chadian Civil War of 2005–2010 began on 18 December 2005.")]),
 dict(iso="LKA", start=1983, end=2009, ex=[("wp_Sri_Lankan_civil_war.txt", "fought in Sri Lanka from 1983 to 2009")]),
 dict(iso="NPL", start=1996, end=2006, ex=[("wp_Nepalese_Civil_War.txt", "The Nepalese Civil War (1996–2006)")]),
 dict(iso="IRQ", start=2003, end=2011, ex=[("wp_Iraq_War.txt", "protracted armed conflict in Iraq from 2003 to 2011")]),
 dict(iso="BDI", start=1993, end=2005, ex=[("wp_Burundian_Civil_War.txt", "lasting from 1993 to 2005")]),
 dict(iso="MOZ", start=1977, end=1992, ex=[("wp_Mozambican_Civil_War.txt", "from 1977 to 1992")]),
 dict(iso="SLE", start=1991, end=2002, ex=[("wp_Sierra_Leone_Civil_War.txt", "The Sierra Leone Civil War (1991–2002)")]),
 dict(iso="AFG", start=1978, end=None, ex=[("wp_Afghan_conflict.txt", "near-continuous state of armed conflict since the 1970s"), ("wp_Women_in_Afghanistan.txt", "The Democratic Republic of Afghanistan (1978–1987)")]),
]
json.dump({"episodes": E, "conflicts": C, "not_used": ["wp_Cultural_Revolution.txt (学校の閉鎖の長さと識字が本文で決まらない)", "wp_Flying_University.txt (高等教育だけ)", "wp_Kuttab.txt (中世の宗教の学校、崩壊でない)", "wp_Nicaraguan_Literacy_Campaign.txt・wp_Cuban_literacy_campaign.txt (識字運動の後の保持の数がない)", "wp_Impact_of_the_COVID-19_pandemic_on_education.txt・wp_Summer_learning_loss.txt (数か月の閉鎖、質的に使う)", "wp_Education_in_Iraq/Liberia/Sierra_Leone/South_Sudan/Yemen/Syria/Iceland/Sweden.txt", "wp_Somali_Civil_War.txt", "wp_Ugandan_Bush_War.txt", "wp_Lithuanian_book_smugglers.txt", "wp_Mozambican_Civil_War.txt の識字の文"]},
          open(os.path.join(H, "cases.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(E), "episodes", len(C), "conflicts")
