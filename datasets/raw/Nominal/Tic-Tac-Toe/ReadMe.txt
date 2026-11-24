# created_date:  18.10.2026
# updated_date:  18.10.2026

Obyektlar soni – 958 ta.

Alomatlar soni – 9 ta (sinf ustuni bu songa kirmaydi).

!!! 1-satr harakteristik vektor.
!!! Oxirgi satr alomat turlarini ko'rsatadi: 0 - nominal, 1 - miqdoriy.
!!! 18.10.2026: UCI fayli (tic-tac-toe.data) default.csv / default.dat formatiga o'tkazildi.
!!! Obyektlar tartibi saqlangan. Faqat format o'zgargan:
!!!   - asl faylda qiymatlar harf bilan berilgan; ular alifbo tartibida 1-3 bilan
!!!     kodlangan (Molecular-Biology dagi A C G T -> 1-4 kabi): b -> 1, o -> 2, x -> 3;
!!!   - sinf belgilari: negative -> 1, positive -> 2.
!!! Takrorlanuvchi qatorlar, bo'sh qiymatlar va o'zgarmas ustunlar yo'q.

#######################################################

Alomatlar hususiyatlari:

"Krestik-nolik" o'yini oxiridagi taxtaning 9 ta katagi (x birinchi yurgan;
o'yin oxirida bo'lishi mumkin bo'lgan barcha holatlar):

1	nominal.	Yuqori chap katak;
2	nominal.	Yuqori o'rta katak;
3	nominal.	Yuqori o'ng katak;
4	nominal.	O'rta chap katak;
5	nominal.	Markaziy katak;
6	nominal.	O'rta o'ng katak;
7	nominal.	Pastki chap katak;
8	nominal.	Pastki o'rta katak;
9	nominal.	Pastki o'ng katak.

Har bir katak qiymati:
		1 - bo'sh (b),
		2 - o (nolik),
		3 - x (krestik).

Maqsadli alomat (Target):

10	nominal.	Sinf:
				1 - x yutmagan (negative) (332 ta),
				2 - x yutgan   (positive) (626 ta).

x yutgan - taxtada x ning 8 ta mumkin bo'lgan "uchtalik" chizig'idan biri bor.

#######################################################

Manba:
https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame (18.10.2026)
Muallif: D. W. Aha (1991).
Litsenziya: CC BY 4.0, DOI: 10.24432/C5688J.
