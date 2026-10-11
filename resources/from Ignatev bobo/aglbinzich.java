import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.Scanner;


    //	Формирование ансамбля алгоритмов с регуляризаторами по обобщённым оценкам
    //  Используется сигмоидная  мажорирующая функция  с корректирующими параметрами
    //    //  Результат получен  25  мая   2025 года
    class aglbaho {
        public static int m, n, nk;
        public static double teta, gamma;
        public static double[] w,wes;
    //    public static double[] wes;
    public static void mera(int v, int k7, double[] p, double[][] x, int[] f) {
        int i, k, ind1, ind2;
        double s, s1, s2, s_max, s1_max;
        double[] r2, r3, rr, p1;
        r2 = new double[k7 + 1];
        r3 = new double[k7 + 1];
        rr = new double[k7 + 1];
        w = new double[m + 1];
        p1 = new double[2];
        p1[0] = p[0];
        p1[1] = p[1];
        //   System.out.println("  Признак " + v + "  градаций " + k7);
        for (k = 0; k < k7 + 1; k++) {
            r2[k] = 0;
            r3[k] = 0;
        }
        for (k = 0; k < m; k++)
            //    Счёт идёт по локальной области
            if (x[k][v] != 999) {
                if (f[k] == 1) r2[(int) x[k][v]] = r2[(int) x[k][v]] + 1;
                else r3[(int) x[k][v]] = r3[(int) x[k][v]] + 1;
            } else p1[f[k] - 1] = p1[f[k] - 1] - 1;
        for (k = 1; k < k7 + 1; k++) {
            s = r2[k] / p1[0] - r3[k] / p1[1];
            //  writeln(' Градация ', k, ' s=', s:7:4);
            rr[k] = s; // вклад по градации признака
      wes[k-1]=s;
        }
        ind1 = k7;
        ind2 = k7; // Число градаций в 1 и 2-м класах
        for (k = 1; k < k7 + 1; k++) if (r2[k] == 0) ind1 = ind1 - 1;
        for (k = 1; k < k7 + 1; k++) if (r3[k] == 0) ind2 = ind2 - 1;
        s_max = 0;
        if (k7 <= 2) s_max = p1[0] * (p1[0] - 1) + p1[1] * (p1[1] - 1);
        else  // for (k = 1; k < nk + 1; k++)
        {
            s_max = s_max + (p1[0] - ind1) * (p1[0] - ind1 + 1);
            s_max = s_max + (p1[1] - ind2) * (p1[1] - ind2 + 1);
        }
        s1_max = p1[0] * p1[1];
        //   Различие  между классами
        s1 = 0;
        s2 = 0;
        for (k = 1; k < k7 + 1; k++) {
            s1 = s1 + r2[k] * r3[k];
            s2 = s2 + r2[k] * (r2[k] - 1) + r3[k] * (r3[k] - 1);   //   Внутрикл.сходство
        }
        s1 = 1 - s1 / s1_max;
        //  teta= 0;  //проверка
        s = s1 * s2 / s_max; // Вес  признака
        teta = s;
        for (k = 0; k < k7; k++)   wes[k]= wes[k] *s;
        for (k = 0; k < m; k++)
      if (x[k][v] != 999)  w[k] = wes[(int) x[k][v]-1];           //      w[k] = s * rr[(int) x[k][v]];
            else w[k] = 0;
    }// mera
        public static void mezon(double[] r3, double[] p, int[] f) {
            int i, k, u, ind2;
            //    baho.mezon(r3, s1, p, f);
            double s, a, b;
            double[] ma;
            ma = new double[2];
            ma[0] =0;
            ma[1] = 0;
       //     u= r3.length;
            for (k = 0; k < m; k++)
             if (k< (int)p[0] ) ma[0] = ma[0] + r3[k];
        else   ma[1] = ma[1] + r3[k];
           ma[0] = ma[0] / p[0];
           ma[1] = ma[1] / p[1];
            teta=0;gamma=0;
            for(k =0;k<m;k++)
            { if (f[k]==1 ) i=0; else i=1;
          teta = teta + Math.abs(r3[k] - ma[i]);
         gamma = gamma + Math.abs(r3[k] - ma[1 - i]);
            }
        }
    } //aglbaho
public class aglbinzich {
    public static void main(String[] args) throws IOException {
        // TODO Auto-generated method stub
        int n, nk, m, k7, k, i, j, u = 0, v, i1, j1, ukk=0;
        int ink, kr, q, kk, k1, nobl, ind, ind1 ;
        double coef, s, s1,s2, maxin;
        double[][] x, c,t,vazn;
        t=new double[][]
                { { 1,1},
           {1.5,0.5},
           {1,0.5,1.5},
           {1,0.5,0.5,2},
           {1,1,2,0.5,0.5},
           {1.5,1,1,0.5,0.5,1,5}
                };
        hizmat ob1;
        aglbaho ob2;
        erdam ob3;
        shahs ob5;
        hisobot ob6;
        int[] f, lt, l, lk7, ln ;
        String naz[];
        //     double[][] bij;
        double[][] x1;
        int[][] chg;  // число градаций номинальных признаков
        ob1 = new hizmat();
        ob2 = new aglbaho();
        ob3 = new erdam();
        ob5 = new shahs();
        ob6 = new hisobot();
        System.out.println();
 //   File f1 = new File("d:/raps/Tasvir25/dab_56tn.dat");  FileWriter ff1 = new FileWriter("d:/raps/Tasvir25/dab56tn.txt");
 //     File  f1 = new File("d:/raps/Derma25/C_der0.dat"); FileWriter  ff1 = new FileWriter("d:/raps/Derma25/Ag_der0.txt");
   //     File f1 = new File("d:/raps/BER1/ELP6-201.dat");  FileWriter ff1 = new FileWriter("d:/raps/BER1/AELP6-201.txt");
  //     File f1 = new File("d:/raps/BER2/ELP6_12.dat");  FileWriter ff1 = new FileWriter("d:/raps/BER2/Aelp6_12.txt");
//  File f1 = new File("d:/raps/Cerdse/her12dn.dat");  FileWriter ff1 = new FileWriter("d:/raps/Cerdse/hermdd12.txt");
  File f1 = new File("d:/raps/Cerdse/heart/Metric_Kanber/heartkanb.dat");  FileWriter ff1 = new FileWriter("d:/raps/Cerdse/herkanb.txt");
 //   File f1 = new File("d:/raps/tromboz/ltromb1.dat");         FileWriter ff1 = new FileWriter("d:/raps/tromboz/Atromb1.txt");
 //  File f1 = new File("d:/raps/molecular/moleculr.dat");  FileWriter ff1 = new FileWriter("d:/raps/molecular/mol10.txt");
//   File f1 = new File("d:/raps/UZBKOR/Uzb_K.dat");  FileWriter ff1 = new FileWriter("d:/raps/UZBKOR/Uzb24.txt");
//  File f1 = new File("d:/raps/german/ger_kanber.dat");   FileWriter ff1 = new FileWriter("d:/raps/german/Kanber20.txt");
        //   FileWriter ff1 = new FileWriter("d:/raps/Prostata/Probin.txt");
 //   File f1 = new File("d:/raps/spambase/Ds_1_223.dat"); FileWriter  ff1 = new FileWriter("d:/raps/spambase/Ds_1_223.txt");
    //    File f1 = new File("d:/raps/SARATON/sar44n.dat"); FileWriter ff1 = new FileWriter("d:/raps/SARATON/sar44.txt");
 //      File f1 = new File("d:/raps/Aritmia/La34.dat");         FileWriter ff1 = new FileWriter("d:/raps/Aritmia/Aa34.txt");
 //          File f1 = new File("d:/raps/mental/mol10.dat");  FileWriter ff1 = new FileWriter("d:/raps/mental/mol10.txt");
     //   FileWriter ff1 = new FileWriter("d:/raps/fizmat/Afizmat.txt");
          InputStreamReader fnom = new InputStreamReader(
               new FileInputStream("d:/raps/Derma25/naz.dat"), StandardCharsets.UTF_8);
        ff1.write("\n Имя файла " + f1.getName());
        Date data = new Date();
        ff1.write("\n" + data);
        try {
            System.out.println(" Размер " + f1.length() + " байт ");
            //String filename = new String ();
            //	System.out.println(" Время модификации; " + f1.lastModified() );
            Scanner src = new Scanner(f1);
            m = src.nextInt();
            System.out.println(" Число объектов " + m);
            n = src.nextInt();
            System.out.println(" Число признаков " + n);
            nk = src.nextInt();
            System.out.println(" Число классов " + nk);
            ff1.write("\n Число объектов " + m);
            ff1.write("\n Число признаков " + n);
            ff1.write("\n");
            x = new double[m][n];
            f = new int[m];
            lt = new int[n];
              lk7 = new int[n]; // количество градаций по n признакам
            naz = new String[n];
// src.next();
            for (i = 0; i < m; i++) {
                for (j = 0; j < n; j++) {
                    x[i][j] = src.nextDouble();
                }
                System.out.println("  " + i + " " + x[i][0] + " " + x[i][n - 1]);    //}
               f[i] = src.nextInt() ; //временно
                  System.out.println(" " + f[i]);
            }
            for (j = 0; j < n; j++) {
                lt[j] = src.nextInt();
              //  lt[j] =0;
                //       System.out.println (" "+lt[j]+" "+f[j]);
            }
            src.close();
        } catch (FileNotFoundException e) {
            System.out.println(" Ошибка при открытии входного файла");
            return;
        }
        try {
            int jjj = 0;
            try (BufferedReader br = new BufferedReader(fnom)) {
                for (String line; (line = br.readLine()) != null; ) {
              //      System.out.println(" " + jjj + ". " + line + " длина строки " + line.length() + " " + line.indexOf(""));
                    if (jjj < n)
                        naz[jjj++] = line;
                }
            }

//            for (j = 0; j < n; j++) {
//                System.out.println("  " + sb);
//                if (nom.hasNext())
//                    sb = nom.nextLine();  //.toString();
//                sb = "a" + j;
//                naz[j] = sb;
//            }
//            nom.close();
        } catch (FileNotFoundException e) {
            System.out.println(" Ошибка при открытии входного файла");
            return;
        }

       for (i = 0; i < n; i++) naz[i]="x"+i;
        double[] r3 = new double[m];
        double[] rr = new double[m];
        double[] r2 = new double[n];
        int[] r1 = new int[r3.length + 1];
        //a = new int[nk + 1];  // границы интервалов
        l = new int[nk + 1];
        double[] d = new double[nk];
        double[] p = new double[nk];
        int[] gate = new int[n]; // выходные данные
        String hulosa;
        x1 = new double[m][n];
        ob3.m = m;
        ob2.m = m;
        ob2.nk = nk;
        ob3.n = n;
        ob2.n = n;
//	nlda =new strr [n] ; // последовательность объединения признаков
        ob5.m = m;
        ob6.m = m;
        ob5.n = n;
        ob6.n = n;
        p[0] = 0;
        p[1] = 0;
        // здесь моделируется классификация выборки
        for (i = 0; i < m; i++) p[f[i] - 1] = p[f[i] - 1] + 1;
        ff1.write("\n Первый класс " + (int) p[0] + " Второй  класс " + (int) p[1]);
        System.out.println(" Первый класс " + (int) p[0] + " Второй  класс " + (int) p[1]);
        ob1.m = m;
        ob1.n = n;
         x1 = new double[m][n];
        // hizmat.valg(x, lt);  // нормирование данных
        k7 = 0;
        for (j = 0; j < n; j++) {

            if (lt[j] != 5) {
                ff1.write(" \n  Признак  " + j + ". " + naz[j]);
                System.out.println("   Признак  " + j + ". " + naz[j]);
            }
            if (lt[j] == 0) {
                ff1.write(" номинальный ");
                System.out.println(" номинальный ");
                for (i = 0; i <= m; i++) r1[i] = 0;
                k = 0;
                u = 0;
                k7 = 0; // номер градации
                do {
                    s = x[k][j];
                    if (s != 999) {
                        k7 = k7 + 1;
                        for (i = k; i < m; i++)
                            if (s == x[i][j]) r1[i] = k7;
                    } else {
                        r1[k] = (int) s;
                        u++;
                    }
                    while (r1[k] != 0) k++;
                } while (k < m);
                System.out.println(" Число градаций признака " + j + "  равно  " + k7);
                ff1.write("\n Число градаций признака  равно  " + k7);
                lk7[j]=k7;  // запомнил число градаций по n признакам
                if (u > 0) ff1.write(" пропусков " + u);
                for (i = 0; i < m; i++)
                   x1[i][j] = r1[i];  // Запись упорядоченных от 1 до k7 номинальных признаков
                c = new double[k7 + 1][2];
// Вычисление значений функций принадлежности
                if (k7 > 1) {
                    for (i = 0; i < k7; i++) {
                        c[i][0] = 0;
                        c[i][1] = 0;
                    }
                    for (i = 0; i < m; i++) c[r1[i] - 1][f[i] - 1] = c[r1[i] - 1][f[i] - 1] + 1;
                    //Функция принадлежности к K1
                    for (i = 0; i < k7; i++) r3[i] = c[i][0] / p[0] / (c[i][0] / p[0] + c[i][1] / p[1]);
                            ff1.write("\n Функция принадлежности по первой градации   " + r3[0] + " " + r3[1]);
           //         System.out.println(" Функция принадлежности по первой градации   " + r3[0] + " " + r3[1]);
                                      // Поиск границы между классами
                    s = 10;
                    s1 = -10;
                    for (i = 0; i < k7; i++) {
                        if (s > 0.5 - r3[i] & 0.5 - r3[i] > 0) s = 0.5 - r3[i];
                        if (s1 < 1 - r3[i] & r3[i] > 0.5) s1 = 1 - r3[i];
                    }
                    s = 0.5 - s;
                    s1 = 1 - s1;
                    ff1.write("\n Граница между классами   " + s + " " + s1 + " " + (s + s1) / 2);
                    s = (s + s1) / 2;
          // переход в бинарное пространство
                    for (i = 0; i < m; i++) {
                        i1 = r1[i] - 1;
                 x[i][j] = r3[i1];
       //                 r1[i] = (int) x1[i][j];
                    }// i
                    coef = 0;
                    for (i = 0; i < m; i++) {
                        if (x[i][j] < s & f[i] == 2) coef = coef + 1;
                        if (x[i][j] > s & f[i] == 1) coef = coef + 1;
                    }
                ff1.write("\n Точность на признаке " + j + " " + coef / m);
                c[0][0] = 0;
                c[0][1] = 0;
                c[1][0] = 0;
                c[1][1] = 0;
                r2[j] = coef / m;
            }   else r2[j]  =0;
        }// if
        } //j
        kk=0;
        hulosa=" ";
        k = -9;
        s = 0;
        for (i = 0; i < n; i++)
            if (s < r2[i]) {
                s = r2[i];
                k = i;
            }
    //          hulosa=hulosa + naz[k];
            ff1.write("\n  Максимальный признак " + k+ " "  + naz[k] +" точность " + s);
             System.out.println("   Максимальный признак " + k + " " + naz[k] + "  точность " + s);
        //====================================================================
        l = new int[n];
        x = new double[m][n+2];
           double kreg; // коэффициент регуляризации один для всех
      //         k=5;
          k7 = lk7[k];
        hulosa=hulosa + naz[k];
       kk=1; //  порядковый номер включения признака в ансамбль
        s = 0;
             maxin = 10000;
         v=k;   j1=-100;
     kreg=0.2;  // значение коэффициента регуляризации
            ff1.write("\n   Общие оценки для   класса" );
                        k=v;  kr=-1;
            ob2.wes= new double[lk7[v]];
            ff1.write("\n Формирование латентного признака для классов k="+ k+ " k7="+k7);
            l[v] = 1; // организатор обший для двух классов
                   rr = new double[m];
                    r3 = new double[m];
           ob2.mera(v,k7,p,x1,f);
        System.out.println("  Вес признака " + ob2.teta+ " k7="+k7);
         s1=1000; kr=-1; coef=0;
                 for (i = 0; i < m; i++)
                    {   r3[i] =  ob2.w[i];  }
                    ob2.mezon(r3, p, f);
                    s = ob2.teta / ob2.gamma;
        System.out.println(" Значение критерия " + s+ "  teta= "+ob2.teta+ "  gamma="+ ob2.gamma );
     ff1.write("\n  Признак " + v + " " + naz[v] + " teta/gamma=" + s );
               // Первая оценка по плотности
if (s>=1) { s2=0;     s = 1/s;}//s>1
        System.out.println(" Значение критерия " + s+ "  teta= "+ob2.teta+ "  gamma="+ ob2.gamma );
        for (i = 0; i < m; i++)    rr[i]=r3[i]; //ob2.w[i];
                maxin=s; // Включил из-за того что организатор между двумя классами
           ind1=0;
        i1 = 0;
           do {
               do {
                   ind = 0;
                   s1 = s;
                   s2 = 0;
                   coef = 0;
                   for (j = 0; j < n; j++)
                       if (l[j] == 0 & lt[j] != 5) {
                           k7 = lk7[j];
                           ob2.wes = new double[lk7[j]];
                           aglbaho.mera(j, k7, p, x1, f);
                           ind1 = 0;
                           j1 = -100;
                           for (i = 0; i < m; i++) {
                               s2 = rr[i] + ob2.w[i];
                               coef = kreg / (1 + Math.exp(s2));  //    s2=
                               if (f[i] == 1) r3[i] = s2 + coef;
                               if (f[i] == 2) r3[i] = s2 - coef;
                           }//i
                           ob2.mezon(r3, p, f);
                            s = ob2.teta / aglbaho.gamma;
                      if (s>1)     s=1/s;   //временно
                           if (s1 > s) {
                               s1 = s;
                               ind1 = 1;
                               j1 = j;
                               System.out.println("  Признак " + j + "  s1=" + s1 + "  teta=" + ob2.teta + "  gamma=" + ob2.gamma);
                           }//d
                           if (ind1 == 1)
                               ff1.write("\n  Признак " + j + " " + naz[j] + " teta/gamma=" + s1);
                           if (maxin > s1) {
                               maxin = s1;
                               k = j;
                               ind = 1;}
                           } //j
                   //  Признак нужно включить в комбинацию
                  //         if (ind==1 & maxin<1 ){
                               if (ind==1 & maxin<1 & i1<n-1  ){
                       System.out.println(" Включил  признак " + k + " maxin=" + maxin);
                       ff1.write("\n   Включил  признак " + k + " " + naz[k] + " teta/gamma=" + maxin + "  точность " + r2[k]);
                       hulosa = hulosa + ",x" + k;
                       //      hulosa = hulosa + "," + naz[k];
                       aglbaho.wes = new double[lk7[k]];
                       k7 = lk7[k];
                       aglbaho.mera(k, k7, p, x1, f);
                       ff1.write("\n Веса признака " + k + " ob2.teta=" + ob2.teta + " k7=" + k7);
                       for (i = 0; i < m; i++) {
                           s2 = rr[i] + ob2.w[i];   // латентный признак для корректировки
                           coef = kreg / (1 + Math.exp(s2));
                           if (f[i] == 1) rr[i] = s2 + coef;
                           else rr[i] = s2 - coef;
                           if (rr[i] > 0) x[i][kk] = 1;
                           else x[i][kk] = 2; // Разделение (дискретизация) по латентному признаку
                       }
                       kk = kk + 1;
                       l[k] = kk;
                   } //if ind=1
       if (s1>1  )   {i1=1000;  ind=0;
         ff1.write("\n  Значение   критерия " +  s1 + " Сходимости  к минимуму нет" ); }
                 if (maxin<0.01  ) ind=0;   // Искусственное ограничение
           //        if (maxin>0.1  ) ind=0;
               } while (ind == 1);
       ff1.write("\n  maxin="+maxin+" i1="+i1);
       if (maxin<0.01 || i1>50) {  //временная проверка
           ff1.write("\n  i1="+i1 ); i1=1000;   }
       //    ff1.write("\n  Есть равномерная сходимость критерия  к минимуму " ); i1=1000;    }
  maxin=1000; i1=i1+1; ind=0;
           } while (i1<n);

                           i1 = 0;
                        for (i = 0; i < n; i++) {
                if (l[i]>0)     i1 = i1 + 1;
          //      ff1.write("\n " + i + ". " + l[i] + " r2=" + r2[i]);
                        }
         //       for (i = 0; i < m; i++)
           //     ff1.write("\n  i="+i+" "+rr[i]+" "+f[i]);
                        System.out.println("  Не попали в группу " + (n - i1) + "  признаков");
                        System.out.println(hulosa);
                        ff1.write("\n " + hulosa);
              // Тестирование классов. Выбор решаюшей функции
              s1=p[0]; s2=p[1];
                for (i = 0; i < m; i++)
                 { //x[i][0] =rr[i];
          if (f[i]==1 & rr[i]<0) s1=s1-1;
       if (f[i]==2 & rr[i]>0) s2=s2-1;
                 }
         s=100*(s1+s2)/m;
        ff1.write("\n Точность " + s + "%  Класс 1 " + s1 + " Класс 2 " + s2+" Регуляризатор "+kreg);
        System.out.println(" Точность " + s + "%  Класс 1 " + s1 + " Класс 2 " + s2 + " reg="+kreg);
      ln= new int[kk]; // порядковые номера признаков
     k=1;  do {
            for (j = 0; j < n; j++)
                if (l[j] == k) {
                    ln[k - 1] = j;
                    }
     System.out.println(" ln=" + ln[k - 1]);
            k = k + 1;
        }while(k<=kk);
        l = new int[m];
   coef=m; // Число ошибок на обучаюшей выборк
    //  тестирование на другой выборке
   //====================================================================================

// ===================================================================
        for (i = 0; i < m; i++) {
     k=ln[0]; s= x1[i][k];  // выбрал первый признак
       for (j = 0; j < m; j++)  if ( s==x1[j][k] ) l[j]=1; else l[j]=0;
  //  ff1.write("\n i=" + i );
    i1=1;
  do {  k1=ln[i1]; s1= x1[i][k1]; // два признака выбрал
 // Сколько есть похожих вариантов?
 s2=0; k7=0; j1=0;
      for (j = 0; j < m; j++) if ( s1== x1[j][k1] & l[j]==1)
      { if (x[j][i1]==1) j1=j1+1; else k7=k7+1;   } else l[j]=0;
   //          ff1.write(" k=" + k1+ " - j1="+ j1+ " k7="+k7 );
      i1=i1+1;
  } while (i1<kk );
            s=j1/p[0] -  k7/p[1];
 if (s>0 & f[i]==2) coef =coef - 1;
      if (s<0 & f[i]==1) coef =coef - 1;
        //    s=j1/p[0] -  k7/p[1];
 //    ff1.write("\n  i="+ i +" " + s +" " + f[i]);
        }
        s=100*(coef)/m;

          ff1.write("\n Точность  на обучении " + s+ "%  kk="+kk);
        System.out.println(" Точность на обучении " + s+ "%  kk="+kk);
          for (j = 0; j <    1; j++)
     //     ff1.write("   " + x[k][j]);
    //       ff1.write("   " + f[k]);
   //   }
        ff1.close();
    }
}//binar
