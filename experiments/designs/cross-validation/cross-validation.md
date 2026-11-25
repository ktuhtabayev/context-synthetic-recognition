## Whole dataset, cross-validation: accuracy and balanced accuracy, %

| Split | Case | Method | Rounds | Training | Classified | Majority rate, % | Accuracy, % | SD | Min | Max | Balanced accuracy, % | Balanced SD | Balanced min | Balanced max | Refusals | Note |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | SVM | 10 | 900 | 100 | 70 | 76.20 | 4.13 | 70.00 | 81.00 | 65.86 | 4.56 | 59.52 | 71.19 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | Random Forest | 10 | 900 | 100 | 70 | 76.10 | 2.92 | 71.00 | 80.00 | 65.88 | 4.18 | 59.29 | 70.95 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Euclidean, k = 3 | 10 | 900 | 100 | 70 | 64.70 | 3.06 | 60.00 | 69.00 | 54.12 | 3.74 | 49.52 | 59.76 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Euclidean, k = 5 | 10 | 900 | 100 | 70 | 64.70 | 2.58 | 60.00 | 69.00 | 51.17 | 3.19 | 45.71 | 55.95 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Chebyshev, k = 3 | 10 | 900 | 100 | 70 | 64.40 | 3.27 | 59.00 | 71.00 | 54.95 | 3.85 | 48.81 | 61.19 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Chebyshev, k = 5 | 10 | 900 | 100 | 70 | 64.10 | 3.35 | 57.00 | 68.00 | 51.79 | 4.09 | 42.62 | 58.10 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Canberra, k = 3 | 10 | 900 | 100 | 70 | 66.70 | 3.50 | 61.00 | 72.00 | 56.79 | 3.97 | 51.90 | 62.38 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Canberra, k = 5 | 10 | 900 | 100 | 70 | 65.80 | 2.62 | 60.00 | 69.00 | 53.19 | 3.28 | 47.62 | 59.76 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Zhuravlyov, k = 3 | 10 | 900 | 100 | 70 | 70.00 | 4.27 | 65.00 | 76.00 | 60.00 | 5.37 | 52.86 | 67.62 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | k-NN Zhuravlyov, k = 5 | 10 | 900 | 100 | 70 | 72.70 | 5.01 | 64.00 | 79.00 | 62.60 | 6.07 | 51.43 | 69.05 | 0 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Euclidean, α = 0.1 | 10 | 900 | 100 | 70 | 63.20 | 21.90 | 1.00 | 73.00 | 46.19 | 15.77 | 1.67 | 55.95 | 9.90 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Euclidean, α = 0.2 | 10 | 900 | 100 | 70 | 63.20 | 21.90 | 1.00 | 73.00 | 46.19 | 15.77 | 1.67 | 55.95 | 9.90 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Chebyshev, α = 0.1 | 10 | 900 | 100 | 70 | 66.20 | 13.47 | 28.00 | 72.00 | 50.90 | 2.18 | 46.67 | 54.29 | 0.10 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Chebyshev, α = 0.2 | 10 | 900 | 100 | 70 | 66.20 | 13.47 | 28.00 | 72.00 | 50.90 | 2.18 | 46.67 | 54.29 | 0.10 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Canberra, α = 0.1 | 10 | 900 | 100 | 70 | 69.40 | 1.17 | 68.00 | 72.00 | 50.24 | 1.38 | 48.57 | 53.33 | 0.10 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Canberra, α = 0.2 | 10 | 900 | 100 | 70 | 69.40 | 1.17 | 68.00 | 72.00 | 50.24 | 1.38 | 48.57 | 53.33 | 0.10 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Zhuravlyov, α = 0.1 | 10 | 900 | 100 | 70 | 72.30 | 1.77 | 69.00 | 75.00 | 55.93 | 2.72 | 51.67 | 60.24 | 1.60 |  |
| German · Zhuravlyov split · training 336 (196, 140) | 10 folds | CS-model Zhuravlyov, α = 0.2 | 10 | 900 | 100 | 70 | 72.20 | 2.04 | 69.00 | 75.00 | 56.05 | 3.12 | 50.95 | 60.24 | 1.70 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | SVM | 10 | 900 | 100 | 70 | 69.90 | 0.99 | 69.00 | 72.00 | 50.50 | 1.41 | 49.29 | 53.33 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | Random Forest | 10 | 900 | 100 | 70 | 69.60 | 3.60 | 63.00 | 75.00 | 57.43 | 4.98 | 48.81 | 65.95 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Euclidean, k = 3 | 10 | 900 | 100 | 70 | 65.60 | 3.84 | 60.00 | 70.00 | 54.57 | 4.04 | 48.33 | 61.43 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Euclidean, k = 5 | 10 | 900 | 100 | 70 | 64.80 | 3.43 | 59.00 | 70.00 | 51.90 | 3.10 | 47.86 | 58.57 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Chebyshev, k = 3 | 10 | 900 | 100 | 70 | 64.30 | 3.09 | 60.00 | 71.00 | 54.12 | 3.84 | 49.29 | 63.10 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Chebyshev, k = 5 | 10 | 900 | 100 | 70 | 65.00 | 3.53 | 57.00 | 69.00 | 52.71 | 3.99 | 44.52 | 57.86 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Canberra, k = 3 | 10 | 900 | 100 | 70 | 65.70 | 3.27 | 61.00 | 73.00 | 55.69 | 5.40 | 50.24 | 69.29 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | k-NN Canberra, k = 5 | 10 | 900 | 100 | 70 | 66.60 | 5.38 | 60.00 | 79.00 | 54.24 | 7.34 | 46.67 | 71.67 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Euclidean, α = 0.1 | 10 | 900 | 100 | 70 | 65.60 | 12.88 | 29.00 | 71.00 | 50.38 | 1.63 | 48.33 | 54.05 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Euclidean, α = 0.2 | 10 | 900 | 100 | 70 | 65.60 | 12.88 | 29.00 | 71.00 | 50.38 | 1.63 | 48.33 | 54.05 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Chebyshev, α = 0.1 | 10 | 900 | 100 | 70 | 65.80 | 13.02 | 29.00 | 72.00 | 50.71 | 2.42 | 46.43 | 54.29 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Chebyshev, α = 0.2 | 10 | 900 | 100 | 70 | 65.80 | 13.02 | 29.00 | 72.00 | 50.71 | 2.42 | 46.43 | 54.29 | 0 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Canberra, α = 0.1 | 10 | 900 | 100 | 70 | 69.70 | 1.89 | 66.00 | 73.00 | 50.55 | 2.19 | 47.14 | 55.00 | 0.20 |  |
| German · Euclidean split · training 341 (173, 168) | 10 folds | CS-model Canberra, α = 0.2 | 10 | 900 | 100 | 70 | 69.70 | 1.89 | 66.00 | 73.00 | 50.55 | 2.19 | 47.14 | 55.00 | 0.20 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | SVM | 10 | 900 | 100 | 70 | 69.60 | 0.97 | 68.00 | 71.00 | 50.10 | 1.21 | 48.57 | 52.62 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | Random Forest | 10 | 900 | 100 | 70 | 70.90 | 2.18 | 66.00 | 73.00 | 59.02 | 2.34 | 55.00 | 61.67 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Euclidean, k = 3 | 10 | 900 | 100 | 70 | 65.40 | 4.67 | 57.00 | 71.00 | 54.81 | 4.59 | 48.33 | 61.19 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Euclidean, k = 5 | 10 | 900 | 100 | 70 | 65.00 | 3.53 | 58.00 | 69.00 | 51.86 | 2.77 | 47.14 | 55.95 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Chebyshev, k = 3 | 10 | 900 | 100 | 70 | 64.30 | 4.30 | 58.00 | 72.00 | 53.64 | 3.41 | 48.33 | 59.05 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Chebyshev, k = 5 | 10 | 900 | 100 | 70 | 65.20 | 4.05 | 58.00 | 71.00 | 52.67 | 3.68 | 47.14 | 59.29 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Canberra, k = 3 | 10 | 900 | 100 | 70 | 65.30 | 4.40 | 62.00 | 76.00 | 55.31 | 4.42 | 50.00 | 61.90 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | k-NN Canberra, k = 5 | 10 | 900 | 100 | 70 | 66.90 | 3.67 | 62.00 | 74.00 | 54.64 | 4.49 | 49.05 | 64.29 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Euclidean, α = 0.1 | 10 | 900 | 100 | 70 | 57.50 | 19.71 | 28.00 | 72.00 | 49.55 | 1.85 | 46.67 | 53.33 | 0.10 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Euclidean, α = 0.2 | 10 | 900 | 100 | 70 | 57.50 | 19.71 | 28.00 | 72.00 | 49.55 | 1.85 | 46.67 | 53.33 | 0.10 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Chebyshev, α = 0.1 | 10 | 900 | 100 | 70 | 69.90 | 1.29 | 68.00 | 72.00 | 51.74 | 1.92 | 49.52 | 55.00 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Chebyshev, α = 0.2 | 10 | 900 | 100 | 70 | 69.90 | 1.29 | 68.00 | 72.00 | 51.74 | 1.92 | 49.52 | 55.00 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Canberra, α = 0.1 | 10 | 900 | 100 | 70 | 69.60 | 0.84 | 68.00 | 71.00 | 50.19 | 1.21 | 48.57 | 51.90 | 0 |  |
| German · Chebyshev split · training 367 (190, 177) | 10 folds | CS-model Canberra, α = 0.2 | 10 | 900 | 100 | 70 | 69.60 | 0.84 | 68.00 | 71.00 | 50.19 | 1.21 | 48.57 | 51.90 | 0 |  |

*Stratified 10-fold cross-validation on all objects; the split markers are ignored. Stratified 10-fold, seed 42; mean and SD over the folds. Training and Classified: the numbers of objects (the mean over the rounds). Majority rate: the share of the larger class among the classified objects. Balanced accuracy: the mean recall of the two classes. Refusals count as errors (Definition 3). Repeated: the mean over the rounds, the standard deviation (n − 1), the minimum and the maximum.*
