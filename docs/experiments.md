# Computational experiment

The computational experiment of the article — R-IPS splits of Heart (270, 13, 2) and German
(1000, 20, 2), the number of binary synthetic features (Table 1), the training and test accuracy
of the CS-model (Tables 2–6) and the comparison with k-NN — is rebuilt from real runs of the
package by one script (ADR-070). The comparison with standard classifiers — SVM, Random Forest and k-NN on the
article's own training and test samples — are in [Standard classifiers](comparison.md).

**No setting was tuned.** The settings are fixed in the script before any run; where the article
leaves a setting open, the choice is written down in the script and repeated in the results. The
numbers the article prints are kept in the script only for the comparison; the computation never
reads them.

!!! warning "The results do not reach the article's accuracies"
    The article reports 96–100 % on the training samples and 100 % on the test samples. The best
    test accuracy obtained here is 79.7 %, and no training accuracy reaches 90 %. The R-IPS
    training sample of Heart has 73 objects (36 of K1, 37 of K2) against the article's 76
    (36, 40) — and only when blocked objects keep covering (⚠ `rips.keep_blocked_cover`).

## Running it

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py              # writes experiments\
.\.venv\Scripts\python.exe scripts\run_experiments.py C:\temp\study # … or another folder
```

The computation is deterministic: a new run writes the committed files byte for byte. The test
suite checks the committed checksums, and a slow test runs the script again and compares every
file (`tests/test_experiments.py`).

## The settings

| Part | Setting |
|---|---|
| Datasets | Heart (270, 13, 2) with the slope x₁₁ quantitative — the article's 7 quantitative and 6 nominal features, set through `dataset.feature_types` (ADR-072); German (1000, 20, 2) (ADR-073) |
| R-IPS | the Zhuravlyov metric on all features, quantitative features normalized to [0; 1]; for German also the Euclidean and the Chebyshev metric on the 7 quantitative features without normalization (Table 6). The training sample is the set of objects that remain in E₁, the test sample the rest. Both readings of blocking are computed: *literal* (the article's text) and *keep-covering* (⚠ `rips.keep_blocked_cover`, ADR-061) |
| Local contexts | one base operator per metric: Zhuravlyov on all features (min–max normalization); Euclidean, Chebyshev and Canberra on the quantitative features without normalization |
| Admissible k | k = 3, 5, …, 41 (`range`) |
| Synthetic features | the values {0, 1, 2}: features that take 0 are left out (`synthetic_features.zero_value`, ADR-064) |
| Additional features | α = 0.1 and 0.2; the rest as in the default configuration (final class means, one majorizing pass, δ = 0.1, ϰ = 5, cr1₀ = 10, φ = logistic sigmoid) |
| Accuracy | training sample: resubstitution (Definition 2); test sample: classified without its class (Definition 3). A refusal counts as an error; the accuracy on the answered objects is written too |
| k-NN | majority vote with k = 1, 3, 5 and the metric of the CS-model, on the same splits |

Settings the article leaves open, as the script records them (`OPEN_SETTINGS`):

- Normalization of the CS-model for the Euclidean, Chebyshev and Canberra metrics: none (the
  article states it only for the R-IPS splits).
- Features of the Canberra metric: the quantitative ones (the metric is defined on them).
- k-NN of the comparison: majority vote with k = 1, 3, 5 and the same metric as the CS-model.
- Table 6 gives one pair of accuracies for two values of α; both α are computed here.
- The article reports 31 synthetic features for k ∈ {3, …, 41}, which has 20 values; the 20 values
  give 20 synthetic features per metric here.
- R-IPS: both readings of blocking are computed; the training sample is the remaining set E₁.

## The files

| File | Content |
|---|---|
| `experiments/results/rips_samples.csv` | per dataset, R-IPS metric and reading: the size and class split of the training and the test sample, and the article's training sample |
| `experiments/results/table1_binary_features.csv` | per dataset, reading and metric: the synthetic features, the binary features left after the features that take 0 are left out, and the article's number |
| `experiments/results/tables2_6_accuracy.csv` | per dataset, R-IPS metric, reading, CS-model metric and α: training accuracy and refusals, test accuracy (refusals as errors), test accuracy on the answered objects, test refusals, TUPLAM (as operator·k) and, if the model is undefined, why |
| `experiments/results/knn_comparison.csv` | the test accuracy of the k-NN vote on the same splits |
| `experiments/results/summary.json` | the package version, the open settings, every run in full (r, left-out features, TUPLAM, training and test scores with their sizes) and the numbers the article prints |
| `experiments/checksums.sha256` | SHA-256 of the five result files |

The CSV files start with a byte-order mark, so that Excel reads them as UTF-8 (ADR-083).

## Comparison with the article

### R-IPS samples

Training sample (objects of K1, K2) / test sample:

| Dataset | ρ of R-IPS | Literal blocking | Blocked objects keep covering ⚠ | Article, training sample |
|---|---|---|---|---|
| Heart | Zhuravlyov | 161 (89, 72) / 109 (61, 48) | 73 (36, 37) / 197 (114, 83) | 76 (36, 40) |
| German | Zhuravlyov | 524 (375, 149) / 476 (325, 151) | 335 (195, 140) / 665 (505, 160) | 336 (196, 140) |
| German | Euclidean | 497 (327, 170) / 503 (373, 130) | 340 (173, 167) / 660 (527, 133) | 341 (173, 168) |
| German | Chebyshev | 518 (337, 181) / 482 (363, 119) | 368 (189, 179) / 632 (511, 121) | 367 (190, 177) |

The literal reading keeps far more objects than the article; the reading in which blocked objects
keep covering comes within a few objects of it.

### Table 1 — binary synthetic features

Here every metric gives 20 synthetic features (one per k = 3, 5, …, 41) on both datasets and in
both readings, and none takes the value 0, so all 20 are binary. The article reports, out of 31:

| Metric | Heart | German |
|---|---|---|
| Zhuravlyov | 30 | 21 |
| Euclidean | 30 | 25 |
| Chebyshev | 30 | 23 |
| Canberra | 29 | 27 |

### Tables 2–5 — accuracy on the Zhuravlyov split

Training / test accuracy in %, refusals counted as errors:

| Dataset | Metric | α | Article | Literal blocking | Blocked objects keep covering ⚠ |
|---|---|---|---|---|---|
| Heart | Zhuravlyov | 0.1 | 97.37 / 100 | 82.61 / 77.06 | 64.38 / 79.7 |
| Heart | Zhuravlyov | 0.2 | 100 / 100 | 82.61 / 77.06 | 71.23 / 78.68 |
| Heart | Euclidean | 0.1 | 97.37 / 100 | 67.7 / 60.55 | 60.27 / 54.31 |
| Heart | Euclidean | 0.2 | 100 / 100 | 69.57 / 62.39 | 61.64 / 54.82 |
| Heart | Chebyshev | 0.1 | 96.05 / 100 | 67.08 / 57.8 | 54.79 / 47.72 |
| Heart | Chebyshev | 0.2 | 100 / 100 | 68.32 / 62.39 | 67.12 / 50.76 |
| Heart | Canberra | 0.1 | 98.68 / 100 | 71.43 / 78.9 | 60.27 / 63.96 |
| Heart | Canberra | 0.2 | 100 / 100 | 73.29 / 77.98 | 73.97 / 67.51 |
| German | Zhuravlyov | 0.1 | 97.32 / 100 | 71.95 / 69.75 | 64.18 / 70.98 |
| German | Zhuravlyov | 0.2 | 100 / 100 | 71.95 / 69.75 | 63.88 / 67.97 |
| German | Euclidean | 0.1 | 100 / 100 | 71.95 / 67.44 | 60.6 / 70.38 |
| German | Euclidean | 0.2 | 100 / 100 | 71.95 / 67.44 | 61.79 / 60.45 |
| German | Chebyshev | 0.1 | 100 / 100 | 71.95 / 67.44 | 58.51 / 69.92 |
| German | Chebyshev | 0.2 | 100 / 100 | 71.95 / 67.44 | 56.72 / 45.41 |
| German | Canberra | 0.1 | 100 / 100 | 71.76 / 67.86 | 60.9 / 67.67 |
| German | Canberra | 0.2 | 100 / 100 | 71.76 / 67.86 | 61.79 / 62.56 |

The best test accuracy, 79.7 % (Heart, Zhuravlyov, α = 0.1, blocked objects keep covering), leaves
21 of the 197 test objects refused; on the answered objects it is 89.2 %.

### Table 6 — German, split by the Euclidean and the Chebyshev metric

The article reports 100 % on the training and on the test sample for both splits. Here, with the
Euclidean, Chebyshev and Canberra metrics in the CS-model, both readings and both α, the training
accuracy lies between 35.91 % and 67.66 % and the test accuracy between 24.07 % and 78.01 % (rows
with `split_metric` `euclidean` or `chebyshev` in `tables2_6_accuracy.csv`).

### k-NN

The article states that k-NN does not exceed 80 %. On the same splits the k-NN vote reaches at
most 80.71 % (Heart, Zhuravlyov split, blocked objects keep covering, k = 5); on that split the
CS-model reaches 79.7 % (α = 0.1).
