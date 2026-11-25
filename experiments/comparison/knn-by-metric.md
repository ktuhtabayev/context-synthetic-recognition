## k-NN by metric: accuracy of the k-NN vote on the test samples, k = 3 / k = 5, %

| Metric | German · Zhuravlyov split · training 336 (196, 140) | German · Euclidean split · training 341 (173, 168) | German · Chebyshev split · training 367 (190, 177) |
| :--- | :--- | :--- | :--- |
| Euclidean | 57.38 / 60.09 | 58.42 / 54.78 | 57.50 / 59.87 |
| Chebyshev | 54.37 / 56.78 | 58.88 / 59.64 | 59.08 / 60.19 |
| Canberra | 57.08 / 60.09 | 54.63 / 56.75 | 58.61 / 60.03 |

*Each cell: k = 3 / k = 5. Majority-class rate of the test sample: 75.90 % (German · Zhuravlyov split · training 336), 79.97 % (German · Euclidean split · training 341), 80.57 % (German · Chebyshev split · training 367). Majority vote of the k nearest training objects on the quantitative features, min–max scaled on the training sample; distances rounded to 10 decimals, equal distances to the smaller training index. Training: the article's training samples; nothing is chosen on the test samples.*
