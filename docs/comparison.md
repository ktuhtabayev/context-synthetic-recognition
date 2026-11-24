# Standard classifiers (Tables 7 and 8)

Tables 7 and 8 of the article compare standard classifiers on the article's training and test
samples: SVM and random forest (Table 7) and k-NN with three metrics (Table 8). This part is kept
apart from the CS-model's pipeline (ADR-079). It reads the split files through the data layer and
classifies with the project's baselines and metrics. It writes the tables with the project's table
model and changes none of them. A third table shows the CS-model on the same samples (ADR-080).

**No setting was tuned, and nothing is chosen on the test samples.** Every setting below was fixed
before the first run. Each method is fitted on the training sample only and classifies the test
objects without their class.

!!! warning "No method reaches the majority-class rate"
    On every split, the share of the larger class in the test sample (75.90 %, 79.97 %,
    80.57 %) is higher than the test accuracy of every method, the CS-model included. The best
    standard classifier is SVM on the Zhuravlyov split, at 74.85 %. The CS-model's balanced
    accuracy stays between 48 % and 57 %: near chance on both classes.

## The article's training and test samples

The splits are kept in `datasets/splits/` under neutral names, in the author's split-file
convention ([File layout](#file-layout)). A test pins their checksums. The dataset catalogue does
not read this folder, so they are not listed among the datasets.

| File (`datasets/splits/…`) | Split made with | Features | Training (K1, K2) | Test (K1, K2) |
| :--- | :--- | :--- | :--- | :--- |
| `German/German (1000, 20, 2) — training 336 (zhuravlyov).dat` | Zhuravlyov metric | all 20: 7 quantitative (min–max over the 1000 objects), 13 nominal | 336 (196, 140) | 664 (504, 160) |
| `German/German (1000, 7, 2) — training 341 (euclidean).dat` | Euclidean metric | the 7 quantitative, raw values | 341 (173, 168) | 659 (527, 132) |
| `German/German (1000, 7, 2) — training 367 (chebyshev).dat` | Chebyshev metric | the 7 quantitative, raw values | 367 (190, 177) | 633 (510, 123) |
| `Heart-Disease/Heart-Disease (270, 13, 2) — training 76 (zhuravlyov).dat` | Zhuravlyov metric | 13: 7 quantitative (the slope too, ADR-072), 6 nominal | 76 (36, 40) | 194 (114, 80) |
| `Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat` | Euclidean metric | the 57 quantitative, raw values | 601 (361, 240) | 1501 (903, 598) |
| `Spambase/Spambase (2102, 57, 2) — training 722 (euclidean).dat` | Euclidean metric | the 57 quantitative, min–max over the 2102 objects | 722 (467, 255) | 1380 (797, 583) |

Each German and Heart file holds the same objects as the project's German (1000, 20, 2) or
Heart-Disease (270, 13, 2), in another order. Object Sᵢ is the file's i-th row. The two Spambase
files hold the same 2102 objects — half of each class of Spambase (4204, 57, 2); some of them carry
a value rounded otherwise, mostly in the third decimal — and the 722 file's values are those of the
601 file min–max scaled over the 2102 objects (ADR-088). Heart's split and Spambase's two can be
read, run on the *Classifiers* page and predicted with `csr compare predict`, but are not part of
Tables 7 and 8.

### File layout

The split files follow the author's convention (ADR-082):

- row 1: the object count m, the feature count n and the class count c, e.g. `1000 7 2`. The
  header counts the features only; empty cells may follow on that row;
- rows 2 … m + 1, one per object: the n feature values, then the class (1 or 2) in column n + 1
  and the split marker (0 = training, 1 = test) in column n + 2;
- last row: the n feature types (0 = nominal, 1 = quantitative). Its cells in columns n + 1 and
  n + 2 are empty, because those columns are not features.

Cells are tab-separated, lines end with LF, there is no BOM, and decimal commas are allowed (the
German 336 and Heart files use them). No line starts with an empty cell.

`data.splits.load_split_file` reads the files into a `FixedSplit` (the dataset and the indices of
the two samples). It takes n from the header and checks the convention: three numbers in the
header, m object rows of exactly n + 2 values, a class among the c classes, a marker 0 or 1, and
exactly n type flags 0/1 in the last row with no other non-empty cell. A departure is reported with
the file and the row (the file's line number). A header that still counts the marker column, such
as `1000 8 2` for seven features, is reported at the first object row. The loader also reads CRLF
line ends and empty trailing cells; spaces do not separate cells. `csr compare` and the
*Classifiers* page stop at the first departure and show the loader's message, for example:

```text
German (1000, 7, 2) — training 341 (euclidean).dat: row 2 (object 1) has 9 values, expected n + 2 = 10 (8 features, the class, the split marker); the header's n counts the features only
```

A split file is not an ordinary dataset: the *Dataset* page and `csr data info` read it in the
dataset layout (n features and the class on each row) and stop at the first object row, which has
one value too many.

## Methods and settings

| Method | Setting |
| :--- | :--- |
| SVM | `SVC(kernel="rbf", C=1.0, gamma="scale")`: the project's `svm` baseline |
| Random forest | `RandomForestClassifier(n_estimators=200, random_state=42)`: the `random-forest` baseline |
| Their features | every feature of the file: quantitative ones min–max scaled on the training sample, nominal ones one-hot encoded with the training sample's categories (an unseen category encodes as all zeros) |
| k-NN | majority vote with k = 3 and k = 5: the `knn-vote` baseline |
| Its features and metrics | the 7 quantitative features, min–max scaled on the training sample; Euclidean, Chebyshev and Canberra (Σ \|a − b\| / (\|a\| + \|b\|), a term is 0 when both values are 0) |
| Its conventions | distances rounded to 10 decimals; equal distances go to the smaller training index (ADR-008) |
| Extra row (not printed) | k-NN with the Zhuravlyov metric on all 20 features of the 336 split |

Reported for every split and method:

- **accuracy on the test sample**: the tables' cells;
- **balanced accuracy**: the mean recall of the two classes;
- **majority-class rate**: the share of the larger class in the test sample;
- **accuracy on the training sample**: resubstitution as in Definition 2. A k-NN training object is
  not its own neighbour; SVM and random forest classify the training objects with the model fitted
  on them.
- **sample sizes**.

## Three stages and the blind test

The comparison runs in three stages (`comparison.stages`, ADR-084):

1. **Training** (`fit_split`): every method is fitted on the training sample — scaling, one-hot
   categories, the k-NN neighbours and the CS-model — and classifies the training objects
   (resubstitution, Definition 2).
2. **Test** (`classify_split`): every fitted method classifies the test objects from an
   `UnlabelledSample`: their place in the file, their feature values, and the feature types and
   names. **It has no class field and no split marker.** `FixedSplit.test_objects()` builds it from
   the test sample by dropping the class column.
3. **Evaluation** (`evaluate_split`): the decisions are compared with the true classes, which are
   kept apart in an `AnswerKey` (`FixedSplit.test_answers()`) that only this stage receives.

`compare_split` is the three stages in a row, so `csr compare run` and the pinned numbers go
through them. The blindness is tested, not only designed (ADR-085):

- shuffling, flipping or deleting the classes of the test objects changes no decision and no score
  of SVM, random forest, any k-NN row or any CS-model;
- a `Dataset` (which carries the classes) is rejected by `classify` at run time and by mypy, which
  the test suite runs on such a call.

SVM and random forest come from the project's baselines, which fit and predict in one call. Their
fitted state is the training sample with the fixed options, and the test stage fits them again on
it. The fit is deterministic (`random_state = 42`), so the decisions are those of a single fit. The
k-NN vote keeps the training sample as its model; the CS-model keeps the fitted model and
classifies with it (the Theorem). The three stages on the three article splits give the files of
`experiments/comparison/` byte for byte.

## Results

Test accuracy, %. These tables repeat `experiments/comparison/table-7.md` and `table-8.md`; a test
keeps them equal.

**Table 7.** SVM and random forest.

| Method | 336 (196, 140) — Zhuravlyov split | 341 (173, 168) — Euclidean split | 367 (190, 177) — Chebyshev split |
| :--- | ---: | ---: | ---: |
| SVM | 74.85 | 50.68 | 61.14 |
| Random Forest | 74.25 | 56.45 | 59.87 |

**Table 8.** k-NN, k = 3 / k = 5. The draft prints this table as a second "Table 7"; here it is
numbered 8.

| Metric | 336 (196, 140) — Zhuravlyov split | 341 (173, 168) — Euclidean split | 367 (190, 177) — Chebyshev split |
| :--- | :--- | :--- | :--- |
| Euclidean | 57.38 / 60.09 | 58.42 / 54.78 | 57.50 / 59.87 |
| Chebyshev | 54.37 / 56.78 | 58.88 / 59.64 | 59.08 / 60.19 |
| Canberra | 57.08 / 60.09 | 54.63 / 56.75 | 58.61 / 60.03 |

Majority-class rate of the test sample: 75.90 % (336), 79.97 % (341), 80.57 % (367). The extra row
(k-NN, Zhuravlyov metric, all 20 features of the 336 split) gives 68.83 / 73.49 %. The balanced
accuracies and the training accuracies of every method are in `results.csv`, `results.md` and
`summary.json`.

An independent run of exactly these settings on the same files (scikit-learn 1.9.1, numpy 2.5.3)
gives the same value for all 55 numbers: 26 test accuracies, 26 balanced accuracies and 3
majority-class rates.

## The CS-model on the same samples

The CS-model is fitted on the same training samples and classifies the same test objects without
their class (the Theorem). Its settings were approved before the first run (ADR-080):

- **Operators:** one base operator per metric, as in the k-NN rows. Euclidean, Chebyshev and Canberra
  run on the quantitative features of every split. Zhuravlyov runs on all 20 features, for the 336
  split only. Every operator is min–max scaled on the training sample.
- **The article's experiments:** odd k = 3 … 41, and synthetic features 0, 1, 2 with the features
  that take 0 on a training object left out (Table 1). α = 0.1 and α = 0.2.
- **Otherwise the article's defaults:** final class means and one majorizing pass. None of the four
  ⚠ switches is on.

Test accuracy, %, with refusals counted as errors (Definition 3); each cell is α = 0.1 / α = 0.2.
This repeats `experiments/comparison/cs-model.md`.

| Metric | 336 (196, 140) — Zhuravlyov split | 341 (173, 168) — Euclidean split | 367 (190, 177) — Chebyshev split |
| :--- | :--- | :--- | :--- |
| Euclidean | 62.95 / 69.13 | 56.90 / 56.15 | 77.41 / 77.41 |
| Chebyshev | 30.87 / 30.87 | 39.15 / 35.51 | 78.36 / 29.07 |
| Canberra | 70.18 / 69.88 | 77.39 / 77.39 | 21.33 / 21.33 |
| Zhuravlyov | 71.54 / 68.07 | — | — |

**How to read these numbers.**

- Every model has r = 20 synthetic features. None takes the value 0 on a training object, as in
  the study of Tables 1–6 (ADR-070).
- **Each model assigns most test objects to one class**, and which class depends on the metric and
  on α. The training samples are almost balanced and the test samples are not, so test accuracy
  swings with that choice:
  - high values (77–78 %) come from assigning almost every object to K1;
  - low values (21–31 %) come from assigning almost every object to K2.
- **Balanced accuracy is 48–57 %**, near chance. The standard classifiers reach 70.63 % (SVM) and
  69.81 % (random forest) on the 336 split, and 52–58 % on the two others.
- Refusals occur only on the 336 split (2 to 21 test objects). The accuracy on the answered objects,
  the training accuracy (Definition 2), TUPLAM and the refusals of every model are in `results.csv`
  and `summary.json`.

## Running it

```powershell
.\.venv\Scripts\csr.exe compare run                      # writes experiments\comparison\
.\.venv\Scripts\csr.exe compare run -o C:\temp\tables    # … or another folder
.\.venv\Scripts\csr.exe compare check                    # checksums + a new run, byte for byte
.\.venv\Scripts\csr.exe compare predict --split 341      # the blind predictions of one split
.\.venv\Scripts\python.exe scripts\run_comparison.py     # the same as compare run
```

`--splits` names another folder of split files. In the desktop application, the *Classifiers* page
runs the same three stages, step by step or all at once, and writes and checks the same files
([Desktop application](gui.md)). The Windows build carries scikit-learn and a copy of the split
files, so the page works there too (ADR-081).

`csr compare predict [--split KEY] [--out FILE]` trains the methods on the training sample of one
split (336 by default; 341, 367, 76, 601 or 722) and classifies its test objects without their class
(ADR-086). It writes one CSV file (with a byte-order mark, ADR-083): the object (Sᵢ, its place in
the file), then one column per method with the decision 1, 2 or 0, empty where a CS-model is
undefined. There is no class column. The loader reads the whole split file and checks the
convention, the classes included; the command keeps only the training sample and the test
objects' feature values, builds no answer key and writes no class. Without `--out` the file is
`predictions-<KEY>.csv` in the working folder, so `experiments/comparison/` keeps exactly the files
of `csr compare run`.

The folder `experiments/comparison/` holds:

| File | Content |
| :--- | :--- |
| `table-7.csv`, `.md`, `.tex` | Table 7, with the majority-class rates and the settings in its note (the LaTeX caption) |
| `table-8.csv`, `.md`, `.tex` | Table 8, the same way |
| `cs-model.csv`, `.md`, `.tex` | the CS-model on the same samples, α = 0.1 / α = 0.2, with its settings in the note |
| `results.csv`, `results.md` | every split × method and CS-model: sizes, majority rate, test accuracy, balanced accuracy, refusals, accuracy on the answered objects, training accuracy |
| `summary.json` | the settings, the versions of numpy and scikit-learn, the SHA-256 of every split file, every number with its counts |
| `checksums.sha256` | SHA-256 of the files above |

The CSV files start with a byte-order mark, so that Excel reads them as UTF-8 and shows "—" and α
correctly (ADR-083). The other files are UTF-8 without it.

The run is deterministic. `tests/test_comparison.py` checks the committed checksums and pins every
number, and a new run must give every file byte for byte. `summary.json` records the versions of
numpy and scikit-learn, not the package version, so a version bump of the package changes no file.
When the installed versions differ from the recorded ones, the byte-for-byte test is skipped and
shows both. The pinned numbers are still checked.
