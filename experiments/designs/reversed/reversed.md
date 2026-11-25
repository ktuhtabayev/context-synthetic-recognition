## Test → Training (reversed): accuracy and balanced accuracy, %

| Split | Method | Training | Classified | Majority rate, % | Accuracy, % | Balanced accuracy, % | Refusals | Note |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| German · Zhuravlyov split · training 336 (196, 140) | SVM | 664 | 336 | 58.30 | 61.61 | 55.77 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | Random Forest | 664 | 336 | 58.30 | 63.10 | 56.94 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Euclidean, k = 3 | 664 | 336 | 58.30 | 57.74 | 51.63 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Euclidean, k = 5 | 664 | 336 | 58.30 | 58.04 | 51.07 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Chebyshev, k = 3 | 664 | 336 | 58.30 | 56.85 | 50.97 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Chebyshev, k = 5 | 664 | 336 | 58.30 | 58.63 | 51.99 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Canberra, k = 3 | 664 | 336 | 58.30 | 58.04 | 51.68 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Canberra, k = 5 | 664 | 336 | 58.30 | 58.93 | 51.94 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Zhuravlyov, k = 3 | 664 | 336 | 58.30 | 59.52 | 54.29 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | k-NN Zhuravlyov, k = 5 | 664 | 336 | 58.30 | 61.31 | 55.71 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Euclidean, α = 0.1 | 664 | 336 | 58.30 | 58.33 | 50.10 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Euclidean, α = 0.2 | 664 | 336 | 58.30 | 58.33 | 50.10 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Chebyshev, α = 0.1 | 664 | 336 | 58.30 | 58.63 | 50.46 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Chebyshev, α = 0.2 | 664 | 336 | 58.30 | 58.63 | 50.46 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Canberra, α = 0.1 | 664 | 336 | 58.30 | 58.33 | 50.10 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Canberra, α = 0.2 | 664 | 336 | 58.30 | 58.33 | 50.10 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Zhuravlyov, α = 0.1 | 664 | 336 | 58.30 | 59.23 | 51.68 | 4 |  |
| German · Zhuravlyov split · training 336 (196, 140) | CS-model Zhuravlyov, α = 0.2 | 664 | 336 | 58.30 | 59.52 | 51.94 | 3 |  |
| German · Euclidean split · training 341 (173, 168) | SVM | 659 | 341 | 50.70 | 49.56 | 48.85 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | Random Forest | 659 | 341 | 50.70 | 51.91 | 51.31 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Euclidean, k = 3 | 659 | 341 | 50.70 | 46.92 | 46.41 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Euclidean, k = 5 | 659 | 341 | 50.70 | 49.27 | 48.67 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Chebyshev, k = 3 | 659 | 341 | 50.70 | 48.39 | 47.88 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Chebyshev, k = 5 | 659 | 341 | 50.70 | 49.56 | 48.98 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Canberra, k = 3 | 659 | 341 | 50.70 | 50.44 | 49.87 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | k-NN Canberra, k = 5 | 659 | 341 | 50.70 | 49.27 | 48.62 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Euclidean, α = 0.1 | 659 | 341 | 50.70 | 0.00 | 0.00 | 341 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Euclidean, α = 0.2 | 659 | 341 | 50.70 | 0.00 | 0.00 | 341 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Chebyshev, α = 0.1 | 659 | 341 | 50.70 | 49.27 | 50.00 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Chebyshev, α = 0.2 | 659 | 341 | 50.70 | 49.27 | 50.00 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Canberra, α = 0.1 | 659 | 341 | 50.70 | 50.15 | 49.42 | 1 |  |
| German · Euclidean split · training 341 (173, 168) | CS-model Canberra, α = 0.2 | 659 | 341 | 50.70 | 50.15 | 49.42 | 1 |  |
| German · Chebyshev split · training 367 (190, 177) | SVM | 633 | 367 | 51.80 | 52.04 | 50.36 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | Random Forest | 633 | 367 | 51.80 | 53.68 | 52.21 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Euclidean, k = 3 | 633 | 367 | 51.80 | 49.86 | 48.56 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Euclidean, k = 5 | 633 | 367 | 51.80 | 51.50 | 50.07 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Chebyshev, k = 3 | 633 | 367 | 51.80 | 52.59 | 51.33 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Chebyshev, k = 5 | 633 | 367 | 51.80 | 52.04 | 50.67 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Canberra, k = 3 | 633 | 367 | 51.80 | 51.50 | 50.12 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | k-NN Canberra, k = 5 | 633 | 367 | 51.80 | 51.23 | 49.63 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Euclidean, α = 0.1 | 633 | 367 | 51.80 | 0.00 | 0.00 | 367 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Euclidean, α = 0.2 | 633 | 367 | 51.80 | 0.00 | 0.00 | 367 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Chebyshev, α = 0.1 | 633 | 367 | 51.80 | 0.00 | 0.00 | 367 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Chebyshev, α = 0.2 | 633 | 367 | 51.80 | 0.00 | 0.00 | 367 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Canberra, α = 0.1 | 633 | 367 | 51.80 | 51.77 | 50.00 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | CS-model Canberra, α = 0.2 | 633 | 367 | 51.80 | 51.77 | 50.00 | 0 |  |

*Fit on the test sample, classify the training sample: the two samples swapped. Training and Classified: the numbers of objects (the mean over the rounds). Majority rate: the share of the larger class among the classified objects. Balanced accuracy: the mean recall of the two classes. Refusals count as errors (Definition 3).*
