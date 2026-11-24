# created_date:  16.10.2026
# updated_date:  17.10.2026

Fayllar:
	Spambase (4204, 57, 2) - UCI faylining (4595 ta obyekt) takrorlanuvchi qatorlari
	                         o'chirilgan versiyasi.
	(4595, 57, 2) versiyasi 17.10.2026 da o'chirildi (sinf belgilari 0/1, 391 ta takror qator).

Obyektlar soni – 4204 ta.

Alomatlar soni – 57 ta (sinf ustuni bu songa kirmaydi).

!!! 1-satr harakteristik vektor.
!!! Oxirgi satr alomat turlarini ko'rsatadi: 0 - nominal, 1 - miqdoriy.
!!! 17.10.2026: .dat fayl default.dat formatiga o'tkazildi (LF qator oxiri, qiymatlar
!!! o'nli kasr ko'rinishida: 4 -> 4.0); qiymatlar o'zgarmagan, .csv bilan tekshirilgan.

#######################################################

Alomatlar hususiyatlari:

1...48  - miqdoriy. Xatdagi ma'lum so'zlarning chastotasi (%);

49...54 - miqdoriy. Ma'lum belgilarning chastotasi (%);

55 - miqdoriy. Bosh harflar ketma-ketligining o'rtacha uzunligi;

56 - miqdoriy. Bosh harflar ketma-ketligining eng katta uzunligi;

57 - miqdoriy. Bosh harflarning umumiy soni.

Maqsadli alomat (Target):

58 - nominal.	Sinf:
		1 - spam emas (2528 ta), 2 - spam (1676 ta).

#######################################################

Eslatmalar (17.10.2026):

!!! (4204, 57, 2) = (4595, 57, 2) ning har bir qatori birinchi uchraganida olingan,
!!!                tartibi saqlangan; sinflar 0/1 -> 1/2 (tekshirildi).

Eslatmalar (16.10.2026, asl (4595, 57, 2) fayl haqida, qiymatlar o'zgartirilmagan):

!!! (4595, 57, 2): sinf belgilari 0/1 (boshqa datasetlardagi 1..C emas).
!!! (4595, 57, 2): 391 ta takror qator.
!!! (4595, 57, 2): asl faylda 6 ta bo'sh satr bor edi (65, 150, 480, 1987, 3110, 3215-satrlar).
!!!                UCI versiyasida 4601 ta obyekt bor; bu faylda 6 tasi kam - ehtimol shu
!!!                bo'sh satrlar o'rnidagi obyektlar o'chirilgan.
!!! (4595, 57, 2): asl faylda ma'lumotlardan keyin 2 ta qo'shimcha satr bor edi
!!!                (faqat 1 lardan va 1/5 lardan iborat, sinf 0). Ular obyekt emas,
!!!                shuning uchun shablonga kiritilmagan.

Manba:
https://archive.ics.uci.edu/dataset/94/spambase
