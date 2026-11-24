# created_date:  18.10.2026
# updated_date:  18.10.2026

Obyektlar soni – 569 ta.

Alomatlar soni – 30 ta (sinf ustuni bu songa kirmaydi).

!!! 1-satr harakteristik vektor.
!!! Oxirgi satr alomat turlarini ko'rsatadi: 0 - nominal, 1 - miqdoriy.
!!! 18.10.2026: UCI fayli (wdbc.data) default.csv / default.dat formatiga o'tkazildi.
!!! Alomat qiymatlari o'zgartirilmagan (har bir qiymat asl fayl bilan tekshirilgan),
!!! obyektlar tartibi saqlangan. Faqat format o'zgargan:
!!!   - 1-ustun (bemorning ID raqami) alomat emas - olib tashlangan;
!!!   - sinf belgilari: B -> 1, M -> 2.
!!! Takrorlanuvchi qatorlar, bo'sh qiymatlar va o'zgarmas ustunlar yo'q.

#######################################################

Alomatlar hususiyatlari:

Ko'krak bezi o'smasidan ingichka igna bilan olingan namuna (FNA) tasviridagi
hujayra yadrolarining 10 ta xususiyati. Har bir tasvir uchun ularning o'rtacha
qiymati, standart xatosi va eng katta qiymati ("worst" - 3 ta eng katta
qiymatning o'rtachasi) hisoblangan:

1...10  - miqdoriy. O'rtacha qiymatlar (mean);

11...20 - miqdoriy. Standart xatolar (SE);

21...30 - miqdoriy. Eng katta qiymatlar (worst).

Har bir o'nlikda xususiyatlar tartibi (masalan, 1 - o'rtacha radius,
11 - radiusning standart xatosi, 21 - eng katta radius):
	1  - radius (markazdan kontur nuqtalarigacha masofalarning o'rtachasi);
	2  - tekstura (kulrang tus qiymatlarining standart og'ishi);
	3  - perimetr;
	4  - yuza;
	5  - silliqlik (radius uzunliklarining mahalliy o'zgarishi);
	6  - ixchamlik (perimetr^2 / yuza - 1.0);
	7  - botiqlik (konturning botiq qismlari chuqurligi);
	8  - botiq nuqtalar (konturning botiq qismlari soni);
	9  - simmetriya;
	10 - fraktal o'lchov ("qirg'oq chizig'i yaqinlashuvi" - 1).

Qiymatlar 4 ta qiymatli raqam bilan yozilgan.

Maqsadli alomat (Target):

31 - nominal.	Sinf:
		1 - xavfsiz o'sma  (benign, B)    (357 ta),
		2 - xavfli o'sma   (malignant, M) (212 ta).

#######################################################

Manba:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (18.10.2026)
Mualliflar: W. Wolberg, O. Mangasarian, N. Street, W. Street (1995).
Litsenziya: CC BY 4.0, DOI: 10.24432/C5DW2B.
