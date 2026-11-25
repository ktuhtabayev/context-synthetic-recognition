## Split to split: accuracy and balanced accuracy, %

| Split | Case | Method | Training | Classified | Majority rate, % | Accuracy, % | Balanced accuracy, % | Refusals | Note |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| German · Zhuravlyov split · training 336 (196, 140) | → German · Euclidean split · training 341 (173, 168) | — | — | — | — | — | — | — | the features differ (20 and 7 features, or other types): the same objects need the same features |
| German · Zhuravlyov split · training 336 (196, 140) | → German · Chebyshev split · training 367 (190, 177) | — | — | — | — | — | — | — | the features differ (20 and 7 features, or other types): the same objects need the same features |
| German · Euclidean split · training 341 (173, 168) | → German · Zhuravlyov split · training 336 (196, 140) | — | — | — | — | — | — | — | the features differ (7 and 20 features, or other types): the same objects need the same features |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | SVM | 341 | 573 | 80.50 | 50.26 | 51.51 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | Random Forest | 341 | 573 | 80.50 | 56.72 | 52.15 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Euclidean, k = 3 | 341 | 573 | 80.50 | 58.46 | 54.58 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Euclidean, k = 5 | 341 | 573 | 80.50 | 54.80 | 48.93 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Chebyshev, k = 3 | 341 | 573 | 80.50 | 57.94 | 52.57 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Chebyshev, k = 5 | 341 | 573 | 80.50 | 58.99 | 52.54 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Canberra, k = 3 | 341 | 573 | 80.50 | 54.10 | 51.20 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | k-NN Canberra, k = 5 | 341 | 573 | 80.50 | 56.89 | 54.96 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Euclidean, α = 0.1 | 341 | 573 | 80.50 | 57.42 | 47.51 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Euclidean, α = 0.2 | 341 | 573 | 80.50 | 56.02 | 51.04 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Chebyshev, α = 0.1 | 341 | 573 | 80.50 | 39.62 | 50.98 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Chebyshev, α = 0.2 | 341 | 573 | 80.50 | 35.43 | 52.10 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Canberra, α = 0.1 | 341 | 573 | 80.50 | 77.84 | 51.08 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | → German · Chebyshev split · training 367 (190, 177) | CS-model Canberra, α = 0.2 | 341 | 573 | 80.50 | 77.84 | 51.08 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Zhuravlyov split · training 336 (196, 140) | — | — | — | — | — | — | — | the features differ (7 and 20 features, or other types): the same objects need the same features |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | SVM | 367 | 573 | 80.50 | 61.78 | 58.33 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | Random Forest | 367 | 573 | 80.50 | 60.21 | 56.01 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Euclidean, k = 3 | 367 | 573 | 80.50 | 57.94 | 54.60 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Euclidean, k = 5 | 367 | 573 | 80.50 | 60.38 | 56.79 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Chebyshev, k = 3 | 367 | 573 | 80.50 | 59.51 | 51.52 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Chebyshev, k = 5 | 367 | 573 | 80.50 | 60.03 | 52.52 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Canberra, k = 3 | 367 | 573 | 80.50 | 59.51 | 54.22 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | k-NN Canberra, k = 5 | 367 | 573 | 80.50 | 60.38 | 56.45 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Euclidean, α = 0.1 | 367 | 573 | 80.50 | 77.66 | 48.94 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Euclidean, α = 0.2 | 367 | 573 | 80.50 | 77.66 | 48.94 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Chebyshev, α = 0.1 | 367 | 573 | 80.50 | 78.88 | 54.77 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Chebyshev, α = 0.2 | 367 | 573 | 80.50 | 29.67 | 52.24 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Canberra, α = 0.1 | 367 | 573 | 80.50 | 21.29 | 49.06 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | → German · Euclidean split · training 341 (173, 168) | CS-model Canberra, α = 0.2 | 367 | 573 | 80.50 | 21.29 | 49.06 | 0 |  |

*Fit on one split's training sample, classify another split's test objects of the same dataset. Every ordered pair of ticked splits of one dataset whose files hold the same objects with identical values and types (matched one to one by their rows); the test objects that are in the first split's training sample are left out. Training and Classified: the numbers of objects (the mean over the rounds). Majority rate: the share of the larger class among the classified objects. Balanced accuracy: the mean recall of the two classes. Refusals count as errors (Definition 3).*
