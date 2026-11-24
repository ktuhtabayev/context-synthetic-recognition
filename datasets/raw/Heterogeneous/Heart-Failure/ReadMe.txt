# created_date:  18.10.2026
# updated_date:  18.10.2026

Obyektlar soni – 299 ta.

Alomatlar soni – 12 ta (sinf ustuni bu songa kirmaydi).

!!! 1-satr harakteristik vektor.
!!! Oxirgi satr alomat turlarini ko'rsatadi: 0 - nominal, 1 - miqdoriy.
!!! 18.10.2026: UCI fayli (heart_failure_clinical_records_dataset.csv) default.csv /
!!! default.dat formatiga o'tkazildi. Alomat qiymatlari o'zgartirilmagan (har bir qiymat
!!! asl fayl bilan tekshirilgan), obyektlar tartibi saqlangan. Faqat format o'zgargan:
!!!   - sarlavha satri (ustun nomlari) olib tashlangan - nomlar quyida [ ] ichida;
!!!   - sinf belgilari (DEATH_EVENT): 0 -> 1, 1 -> 2.
!!! Takrorlanuvchi qatorlar, bo'sh qiymatlar va o'zgarmas ustunlar yo'q.

#######################################################

Alomatlar hususiyatlari:

1 - miqdoriy. Yoshi (yil, 40...95) [age];

2 - nominal. Kamqonlik (anemiya): 0 - yo'q, 1 - bor [anaemia];

3 - miqdoriy. Qondagi kreatinfosfokinaza (KFK) fermenti darajasi (mkg/l)
    [creatinine_phosphokinase];

4 - nominal. Qandli diabet: 0 - yo'q, 1 - bor [diabetes];

5 - miqdoriy. Har bir qisqarishda yurakdan chiqadigan qon ulushi - otish fraksiyasi (%)
    [ejection_fraction];

6 - nominal. Yuqori qon bosimi (gipertoniya): 0 - yo'q, 1 - bor [high_blood_pressure];

7 - miqdoriy. Qondagi trombotsitlar (manbada: kilotrombotsit/ml) [platelets];

8 - miqdoriy. Qon zardobidagi kreatinin (mg/dl) [serum_creatinine];

9 - miqdoriy. Qon zardobidagi natriy (mEq/l) [serum_sodium];

10 - nominal. Jinsi: 0 - ayol (105 ta), 1 - erkak (194 ta) [sex];

11 - nominal. Chekadimi: 0 - yo'q, 1 - ha [smoking];

12 - miqdoriy. Kuzatuv davri (kun, 4...285) [time];

Maqsadli alomat (Target):

13 - nominal.	Sinf (DEATH_EVENT - kuzatuv davrida vafot etganmi):
		1 - yo'q (203 ta),
		2 - ha   (96 ta).

#######################################################

Tavsif:
Yurak yetishmovchiligi bo'lgan 299 nafar bemorning kuzatuv davridagi tibbiy yozuvlari
(Faysalobod yurak instituti va Allied kasalxonasi, Faysalobod, Pokiston).

Manba:
https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records (18.10.2026)
Maqola: D. Chicco, G. Jurman. Machine learning can predict survival of patients with heart
failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and
Decision Making 20, 16 (2020).
Litsenziya: CC BY 4.0, DOI: 10.24432/C5Z89R.
