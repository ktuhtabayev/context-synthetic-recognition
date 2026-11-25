## Standard classifiers: accuracy of SVM and Random Forest on the test samples, %

| Method | 336 (196, 140) — Zhuravlyov split | 341 (173, 168) — Euclidean split | 367 (190, 177) — Chebyshev split |
| :--- | ---: | ---: | ---: |
| SVM | 74.85 | 50.68 | 61.14 |
| Random Forest | 74.25 | 56.45 | 59.87 |

*Each method is fitted on the training sample and classifies the test sample. Majority-class rate of the test sample: 75.90 % (336), 79.97 % (341), 80.57 % (367). SVM: SVC with the RBF kernel, C = 1.0, gamma = scale; Random Forest: 200 trees, random_state = 42. Every feature of the file: the quantitative ones min–max scaled on the training sample, the nominal ones one-hot encoded with the categories of the training sample (an unseen category encodes as all zeros).*
