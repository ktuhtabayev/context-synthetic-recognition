# Standard classifiers

The comparison with standard classifiers puts them on the article's training and test samples:
SVM and Random Forest (the *Standard classifiers* table) and k-NN with three metrics (the *k-NN by
metric* table). The tables are named by their content; the article numbers its tables. This part is kept
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
convention ([File layout](#file-layout)). The dataset catalogue does not read this folder, so they
are not listed among the datasets. **Every split file of the folder is a sample** (ADR-090): the
project finds the files by their names ([Adding a split file](#adding-a-split-file)), and nothing
lists them in code. A test pins the checksums of the six files below; a file added later is
checked, not pinned.

| File (`datasets/splits/…`) | Key | Split made with | Features | Training (K1, K2) | Test (K1, K2) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `German/German (1000, 20, 2) — training 336 (zhuravlyov).dat` | `German-336` | Zhuravlyov metric | all 20: 7 quantitative (min–max over the 1000 objects), 13 nominal | 336 (196, 140) | 664 (504, 160) |
| `German/German (1000, 7, 2) — training 341 (euclidean).dat` | `German-341` | Euclidean metric | the 7 quantitative, raw values | 341 (173, 168) | 659 (527, 132) |
| `German/German (1000, 7, 2) — training 367 (chebyshev).dat` | `German-367` | Chebyshev metric | the 7 quantitative, raw values | 367 (190, 177) | 633 (510, 123) |
| `Heart-Disease/Heart-Disease (270, 13, 2) — training 76 (zhuravlyov).dat` | `Heart-Disease-76` | Zhuravlyov metric | 13: 7 quantitative (the slope too, ADR-072), 6 nominal | 76 (36, 40) | 194 (114, 80) |
| `Spambase/Spambase (2102, 57, 2) — training 601 (euclidean).dat` | `Spambase-601` | Euclidean metric | the 57 quantitative, raw values | 601 (361, 240) | 1501 (903, 598) |
| `Spambase/Spambase (2102, 57, 2) — training 722 (euclidean).dat` | `Spambase-722` | Euclidean metric | the 57 quantitative, min–max over the 2102 objects | 722 (467, 255) | 1380 (797, 583) |

Each German and Heart file holds the same objects as the project's German (1000, 20, 2) or
Heart-Disease (270, 13, 2), in another order. Object Sᵢ is the file's i-th row. The two Spambase
files hold the same 2102 objects — half of each class of Spambase (4204, 57, 2); some of them carry
a value rounded otherwise, mostly in the third decimal — and the 722 file's values are those of the
601 file min–max scaled over the 2102 objects (ADR-088).

Every split can be run on the *Classifiers* page, written with `csr compare run --split KEY --out
DIR` and predicted with `csr compare predict`. The comparison tables are those of **German's three
splits** — the Zhuravlyov, Euclidean and Chebyshev splits — the only list of splits in the code
(`comparison.COMPARISON_SPLITS`).

A split's **key** is `<Dataset>-<t>`, its dataset and its training size: it names the split on the
command line and inside the project, and the desktop application never shows it. Everything a
user reads names a split by its dataset: "German · Euclidean split · training 341 (173, 168)".
The keys of the earlier versions — the training size alone, `336` … `722` — still name a split on
the command line while exactly one split has that size; a size that two splits share is reported
as ambiguous.

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

### Adding a split file

Save the file in the split-file convention ([File layout](#file-layout)) anywhere below
`datasets/splits/` — by convention in a folder named after its dataset — under the name

```text
<Dataset> (<m>, <n>, <c>) — training <t> (<metric>).dat
```

for example `Iris (150, 4, 2) — training 60 (euclidean).dat`: the dataset's name, its object,
feature and class counts, an em dash, the size of the training sample, and the registered metric
the split was made with (`zhuravlyov`, `euclidean`, `chebyshev`, …). Nothing else is needed: the
next time the *Classifiers* page reads the folder (or `csr compare` runs), the file is a sample of
its dataset with the key `<Dataset>-<t>`. The project checks it first (ADR-090):

- the name follows the pattern;
- the file follows the convention, and its header holds the m, n and c of the name;
- its split markers give t training objects;
- the metric is registered: one of the metrics of the
  [algorithm reference](algorithm-reference.md), by its registry name.

A file that fails a check is listed as unavailable under its dataset, with its file, its row and
the reason; it is never used and never stops the others. Two files with one key — the same
dataset and training size — are both unavailable. The comparison needs two classes.

## Methods and settings

Every method has its own settings (ADR-094); what a group of methods shares is said once.

| Method | Its own settings |
| :--- | :--- |
| SVM | SVC, RBF kernel, C = 1.0, gamma = scale — `SVC(kernel="rbf", C=1.0, gamma="scale")`, the project's `svm` baseline |
| Random Forest | 200 trees, random_state = 42 — `RandomForestClassifier(n_estimators=200, random_state=42)`, the `random-forest` baseline |
| k-NN (one row per metric and k) | its metric (Euclidean, Chebyshev or Canberra: Σ \|a − b\| / (\|a\| + \|b\|), a term is 0 when both values are 0), k = 3 or k = 5, the quantitative features; equal distances go to the smaller training index (ADR-008) — the `knn-vote` baseline, a majority vote |
| CS-model (one row per metric and α) | its metric, α = 0.1 or 0.2, odd k = 3 … 41, synthetic features 0, 1, 2 (those that take 0 on a training object left out) — see [below](#the-cs-model-on-the-same-samples) |

| Shared by | What they share |
| :--- | :--- |
| SVM and Random Forest | every feature of the file: quantitative ones min–max scaled on the training sample, nominal ones one-hot encoded with the training sample's categories (an unseen category encodes as all zeros) |
| the k-NN rows | the quantitative features, min–max scaled on the training sample; distances rounded to 10 decimals |

The methods of a split follow one rule (ADR-090), not a list per file:

- every split: SVM and Random Forest; the k-NN vote and the CS-model with the Euclidean,
  Chebyshev and Canberra metrics on the quantitative features;
- a split of the comparison set whose own metric is none of these three also runs the k-NN vote
  and the CS-model with its own metric on every feature: German's Zhuravlyov split, whose
  Zhuravlyov k-NN rows are in neither comparison table;
- a method a split cannot support is listed as unavailable with the reason, never an error: no
  quantitative features (the Euclidean, Chebyshev and Canberra k-NN and CS-model), or fewer than
  k + 1 training objects for the k-NN vote.

Reported for every split and method:

- **accuracy on the test sample**: the tables' cells;
- **balanced accuracy**: the mean recall of the two classes;
- **majority-class rate**: the share of the larger class in the test sample;
- **accuracy on the training sample**: resubstitution as in Definition 2. A k-NN training object is
  not its own neighbour; SVM and Random Forest classify the training objects with the model fitted
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
  of SVM, Random Forest, any k-NN row or any CS-model;
- a `Dataset` (which carries the classes) is rejected by `classify` at run time and by mypy, which
  the test suite runs on such a call.

SVM and Random Forest come from the project's baselines, which fit and predict in one call. Their
fitted state is the training sample with the fixed options, and the test stage fits them again on
it. The fit is deterministic (`random_state = 42`), so the decisions are those of a single fit. The
k-NN vote keeps the training sample as its model; the CS-model keeps the fitted model and
classifies with it (the Theorem). The three stages on the three article splits give the files of
`experiments/comparison/` byte for byte.

## Results

Test accuracy, %. These tables repeat `experiments/comparison/standard-classifiers.md` and
`knn-by-metric.md`; a test keeps them equal.

**Standard classifiers.** SVM and Random Forest.

| Method | German · Zhuravlyov split · training 336 (196, 140) | German · Euclidean split · training 341 (173, 168) | German · Chebyshev split · training 367 (190, 177) |
| :--- | ---: | ---: | ---: |
| SVM | 74.85 | 50.68 | 61.14 |
| Random Forest | 74.25 | 56.45 | 59.87 |

**k-NN by metric.** k-NN, k = 3 / k = 5.

| Metric | German · Zhuravlyov split · training 336 (196, 140) | German · Euclidean split · training 341 (173, 168) | German · Chebyshev split · training 367 (190, 177) |
| :--- | :--- | :--- | :--- |
| Euclidean | 57.38 / 60.09 | 58.42 / 54.78 | 57.50 / 59.87 |
| Chebyshev | 54.37 / 56.78 | 58.88 / 59.64 | 59.08 / 60.19 |
| Canberra | 57.08 / 60.09 | 54.63 / 56.75 | 58.61 / 60.03 |

Majority-class rate of the test sample: 75.90 % (German · Zhuravlyov split · training 336),
79.97 % (German · Euclidean split · training 341), 80.57 % (German · Chebyshev split · training
367). The extra row (k-NN, Zhuravlyov metric, all 20 features of German's Zhuravlyov split) gives
68.83 / 73.49 %. The balanced
accuracies and the training accuracies of every method are in `results.csv`, `results.md` and
`summary.json`.

An independent run of exactly these settings on the same files (scikit-learn 1.9.1, numpy 2.5.3)
gives the same value for all 55 numbers: 26 test accuracies, 26 balanced accuracies and 3
majority-class rates.

## The CS-model on the same samples

The CS-model is fitted on the same training samples and classifies the same test objects without
their class (the Theorem). Its settings were approved before the first run (ADR-080):

- **Operators:** one base operator per metric, as in the k-NN rows. Euclidean, Chebyshev and Canberra
  run on the quantitative features of every split. Zhuravlyov runs on all 20 features, for German's
  Zhuravlyov split only. Every operator is min–max scaled on the training sample.
- **The article's experiments:** odd k = 3 … 41, and synthetic features 0, 1, 2 with the features
  that take 0 on a training object left out (Table 1). α = 0.1 and α = 0.2.
- **Otherwise the article's defaults:** final class means and one majorizing pass. None of the four
  ⚠ switches is on.

Test accuracy, %, with refusals counted as errors (Definition 3); each cell is α = 0.1 / α = 0.2.
This repeats `experiments/comparison/cs-model.md`.

| Metric | German · Zhuravlyov split · training 336 (196, 140) | German · Euclidean split · training 341 (173, 168) | German · Chebyshev split · training 367 (190, 177) |
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
  69.81 % (Random Forest) on German's Zhuravlyov split, and 52–58 % on the two others.
- Refusals occur only on German's Zhuravlyov split (2 to 21 test objects). The accuracy on the answered objects,
  the training accuracy (Definition 2), TUPLAM and the refusals of every model are in `results.csv`
  and `summary.json`.

## Evaluation designs

A split file holds three sets: the training sample T, the test sample Q and all its m objects.
The comparison above fits on T and classifies Q, as the article does. An **evaluation design**
uses the same file another way (ADR-095), so that the numbers can be read against other points
of view: are they a property of these two samples, of the dataset, or of chance? Every design runs
the same methods with the same fixed settings, and every design keeps the blind rule: the objects
being classified reach the methods without their class, and normalization, encoding, R-IPS and
every fit use only that design's training objects.

| Design | What it does | What it shows |
| :--- | :--- | :--- |
| **Swapping the subsets** | | |
| Training → Test (as given) | fit on T, classify Q | the article's evaluation; the default |
| Test → Training (reversed) | fit on Q, classify T | whether the result holds with the roles of the two samples swapped |
| Training → Training (self-check) | T on itself, by resubstitution (Definition 2): a k-NN object is not its own neighbour, a CS-model object is left out of its own context, SVM and Random Forest classify the objects they were fitted on | how well each method explains its own training sample; equals the comparison's training accuracy |
| Test → Test (self-check) | Q on itself, the same way | how well the test objects explain each other |
| **Comparison points** | | |
| Random training samples (control) | 100 samples drawn from the m objects with T's size and class counts (seeds 42 … 141); each fitted, the other objects classified | the spread of chance samples, and where T's result ranks among them (percentile) |
| Whole dataset, cross-validation | stratified 10-fold on the m objects (seed 42); the split markers ignored | the usual estimate on the whole dataset: mean and SD over the folds |
| **Honest estimate** | | |
| Selection inside folds | stratified 10-fold (seed 42); in each fold R-IPS (the article's defaults, the split's metric) chooses the training sample from the fold's training objects only; fitted on it, the held-out fold classified | selection and fit together, without the held-out objects ever touching the selection; with the sizes R-IPS chose |
| **Optional extras** | | |
| Training-size curve | stratified random subsamples of T of 25, 50 and 75 % (20 repetitions each, seeds 42 … 61), and T itself; Q classified | how the accuracy grows with the size of the training sample |
| Split to split | fit on split A's T, classify split B's Q without the objects of A's T, for every ordered pair of compatible splits of one dataset | whether a training sample chosen with one metric serves another split |

**Settings.** They are fixed in `services.designs` and shown read-only on the page: the seeds,
100 random samples, 10 folds, the shares 25 / 50 / 75 % (rounded half up per class, at least one
object) and 20 repetitions each. Split to split needs the same objects with identical values and
types, matched one to one by their rows; otherwise the pair is shown as unavailable with the
reason. German's Zhuravlyov split (20 features) and its Euclidean split (7 features) are not
compatible; Spambase's two splits are not either, because the 722 file holds scaled values.

**What is reported.** For every design, split and method: the accuracy and the balanced
accuracy, the refusals, the majority-class rate of the classified objects and the sizes; a
repeated design (random control, cross-validation, selection inside folds, curve) adds the
standard deviation (n − 1) and the minimum and maximum over its rounds. A method or a case a
split cannot support — an undefined CS-model on a small training sample, an incompatible pair —
is shown with the reason; it never stops the rest.

**Two checks.** The design as given on German's three splits gives exactly the numbers of the
tables above, and the training self-check exactly the training accuracies of `results.csv`; the
tests pin both. The selection inside folds also reports — without using it — whether the
project's R-IPS on all m objects of a split file reproduces that file's training sample.

**Where.** On the *Classifiers* page, the *Design* chooser beside the samples ([Desktop
application](gui.md#8-classifiers)). In the CLI:

```powershell
.\.venv\Scripts\csr.exe compare run --design all                 # experiments\designs\<design>\
.\.venv\Scripts\csr.exe compare run --design reversed --split Heart-Disease-76 -o C:\temp\heart
.\.venv\Scripts\csr.exe compare check --design all               # every design folder, byte for byte
.\.venv\Scripts\csr.exe compare predict --design reversed --split German-341
```

`experiments/designs/<design>/` holds `<design>.csv | md | tex` (every split, case and method),
`rounds.csv` for a repeated design (every round), `summary.json` (the settings, the versions, every
number) and `checksums.sha256`.

### Results on German's three splits

From `experiments/designs/` (`csr compare run --design all`, 9.9 minutes here; tests pin the
numbers). Four of the methods, the mean of a repeated design; every method, case and round is in
the folders, and `experiments/designs/all-designs/` puts all of them side by side.

**German · Zhuravlyov split · training 336 (196, 140)** — accuracy / balanced accuracy, %

| Method | as given | reversed | T self | Q self | random (mean) | 10-fold CV | R-IPS in folds | 50 % of T |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 74.85 / 70.63 | 61.61 / 55.77 | 83.63 / 81.89 | 90.51 / 82.66 | 73.47 / 69.02 | 76.20 / 65.86 | 73.40 / 62.43 | 73.73 / 63.75 |
| Random Forest | 74.25 / 69.81 | 63.10 / 56.94 | 100.00 / 100.00 | 100.00 / 100.00 | 74.62 / 69.03 | 76.10 / 65.88 | 73.30 / 63.50 | 73.43 / 66.28 |
| k-NN Euclidean, k = 5 | 60.09 / 51.74 | 58.04 / 51.07 | 55.95 / 53.67 | 73.04 / 53.66 | 62.51 / 54.22 | 64.70 / 51.17 | 65.90 / 52.88 | 60.59 / 52.32 |
| CS-model Euclidean, α = 0.1 | 62.95 / 51.92 | 58.33 / 50.10 | 58.93 / 55.51 | 75.90 / 50.43 | 64.21 / 50.74 | 63.20 / 46.19 | 24.00 / 40.00 | 53.76 / 47.68 |

**German · Euclidean split · training 341 (173, 168)** — accuracy / balanced accuracy, %

| Method | as given | reversed | T self | Q self | random (mean) | 10-fold CV | R-IPS in folds | 50 % of T |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 50.68 / 52.41 | 49.56 / 48.85 | 59.53 / 59.65 | 81.03 / 52.94 | 62.39 / 58.90 | 69.90 / 50.50 | 70.00 / 51.62 | 53.42 / 53.01 |
| Random Forest | 56.45 / 52.61 | 51.91 / 51.31 | 100.00 / 100.00 | 100.00 / 100.00 | 60.72 / 59.31 | 69.60 / 57.43 | 66.10 / 57.21 | 55.58 / 52.71 |
| k-NN Euclidean, k = 5 | 54.78 / 49.58 | 49.27 / 48.67 | 53.08 / 53.02 | 77.24 / 52.83 | 57.62 / 54.68 | 64.80 / 51.90 | 61.70 / 52.55 | 54.44 / 51.38 |
| CS-model Euclidean, α = 0.1 | 56.90 / 48.07 | 0.00 / 0.00 | 53.08 / 52.97 | 0.00 / 0.00 | 58.02 / 53.05 | 65.60 / 50.38 | 57.40 / 49.67 | 50.27 / 47.97 |

**German · Chebyshev split · training 367 (190, 177)** — accuracy / balanced accuracy, %

| Method | as given | reversed | T self | Q self | random (mean) | 10-fold CV | R-IPS in folds | 50 % of T |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 61.14 / 57.99 | 52.04 / 50.36 | 62.67 / 62.48 | 81.83 / 53.56 | 64.36 / 59.17 | 69.60 / 50.10 | 70.30 / 53.45 | 61.39 / 56.58 |
| Random Forest | 59.87 / 55.97 | 53.68 / 52.21 | 100.00 / 100.00 | 100.00 / 100.00 | 62.05 / 60.11 | 70.90 / 59.02 | 68.10 / 59.40 | 60.01 / 54.70 |
| k-NN Euclidean, k = 5 | 59.87 / 56.59 | 51.50 / 50.07 | 50.68 / 50.49 | 78.20 / 52.85 | 58.52 / 54.78 | 65.00 / 51.86 | 61.80 / 53.57 | 57.35 / 53.36 |
| CS-model Euclidean, α = 0.1 | 77.41 / 48.66 | 0.00 / 0.00 | 53.95 / 52.34 | 0.00 / 0.00 | 60.74 / 54.01 | 57.50 / 49.55 | 58.10 / 51.31 | 45.79 / 49.64 |

How to read them:

- **As given = the comparison** above, and the training self-check = its training accuracy;
  Random Forest is at 100 % on its own training objects.
- **The self-checks of the test samples** look high for SVM (81–91 %), but their balanced accuracy
  is 53–83 %: the test samples are dominated by K1.
- **Reversed**, fitted on the larger, unbalanced test sample, the standard classifiers lose 10–15
  points on German's Zhuravlyov split and stay near 50 % on the others; the Euclidean CS-model
  refuses every training object of the Euclidean and Chebyshev splits (0.00: refusals count as
  errors).
- **Random control**: German's Zhuravlyov training sample ranks at the 76.5th percentile of 100
  random samples for SVM and the 38.5th for Random Forest; the Euclidean split's sample ranks at
  the 1st and 0.5th — almost every random sample of its size and classes does better — and the
  Chebyshev split's at the 14.5th and 15th.
- **Cross-validation and selection inside folds** on all 1000 objects give 69.6–76.2 % for SVM
  (balanced 50–66 %); R-IPS chooses 438–492 of a fold's 900 training objects. R-IPS on all 1000
  objects with the article's defaults chooses 524, 506 and 523 objects; the files hold 336, 341 and
  367, so it does not reproduce them.
- **Training-size curve**: on German's Zhuravlyov split SVM grows from 73.14 % (25 %) to 74.85 %
  (100 %); on the two others more training objects do not help.
- **Split to split**: only the Euclidean and the Chebyshev split hold the same objects; fitted on
  one and classifying the other's test objects outside its training sample, SVM reaches 50.26 % and
  61.78 %.

## Running it

```powershell
.\.venv\Scripts\csr.exe compare run                      # writes experiments\comparison\
.\.venv\Scripts\csr.exe compare run -o C:\temp\tables    # … or another folder
.\.venv\Scripts\csr.exe compare check                    # checksums + a new run, byte for byte
.\.venv\Scripts\csr.exe compare run --split Spambase-601 -o C:\temp\spambase   # one split
.\.venv\Scripts\csr.exe compare predict --split German-341   # the blind predictions of one split
.\.venv\Scripts\python.exe scripts\run_comparison.py     # the same as compare run
```

`--splits` names another folder of split files. In the desktop application, the *Classifiers* page
runs the same three stages, step by step or all at once, and writes and checks the same files
([Desktop application](gui.md)). The Windows build carries scikit-learn and a copy of the split
files, so the page works there too (ADR-081).

`csr compare run --split KEY --out DIR` runs one split, by key or by the training size of an
earlier version (`601`), and writes its evaluation — `results.csv`, `results.md`, `summary.json`
and `checksums.sha256` — to DIR. It never writes to `experiments/comparison/`, which holds
German's three splits only, and it needs `--out`. `csr compare check --out DIR` checks either
kind of folder: it reruns the splits its `summary.json` names.

`csr compare predict [--split KEY] [--out FILE]` trains the methods on the training sample of one
split (`German-336` by default; any key, or an earlier version's training size) and classifies its test objects without their class
(ADR-086). It writes one CSV file (with a byte-order mark, ADR-083): the object (Sᵢ, its place in
the file), then one column per method with the decision 1, 2 or 0, empty where a CS-model is
undefined. There is no class column. The loader reads the whole split file and checks the
convention, the classes included; the command keeps only the training sample and the test
objects' feature values, builds no answer key and writes no class. Without `--out` the file is
`predictions-<key>.csv` (`predictions-German-336.csv`) in the working folder, so `experiments/comparison/` keeps exactly the files
of `csr compare run`.

The folder `experiments/comparison/` holds:

| File | Content |
| :--- | :--- |
| `standard-classifiers.csv`, `.md`, `.tex` | the *Standard classifiers* table, with the majority-class rates and the settings in its note (the LaTeX caption) |
| `knn-by-metric.csv`, `.md`, `.tex` | the *k-NN by metric* table, the same way |
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
