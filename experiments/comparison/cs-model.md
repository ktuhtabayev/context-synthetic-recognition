## CS-model on the same samples: accuracy on the test samples, α = 0.1 / α = 0.2, %

| Metric | 336 (196, 140) — Zhuravlyov split | 341 (173, 168) — Euclidean split | 367 (190, 177) — Chebyshev split |
| :--- | :--- | :--- | :--- |
| Euclidean | 62.95 / 69.13 | 56.90 / 56.15 | 77.41 / 77.41 |
| Chebyshev | 30.87 / 30.87 | 39.15 / 35.51 | 78.36 / 29.07 |
| Canberra | 70.18 / 69.88 | 77.39 / 77.39 | 21.33 / 21.33 |
| Zhuravlyov | 71.54 / 68.07 | — | — |

*Each cell: α = 0.1 / α = 0.2; — = not run on that split. Majority-class rate of the test sample: 75.90 % (336), 79.97 % (341), 80.57 % (367). One base operator per metric, min–max scaled on the training sample: Euclidean, Chebyshev and Canberra on the quantitative features, Zhuravlyov on all features (the 336 split only); odd k = 3 … 41; synthetic features 0, 1, 2 with the features that take 0 on a training object left out; final class means, one majorizing pass. Refusals count as errors (Definition 3). Fitted on the article's training samples; the test objects are classified without their class.*
