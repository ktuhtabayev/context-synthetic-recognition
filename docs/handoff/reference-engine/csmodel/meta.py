"""Meta-algorithm of the article (Steps 1–5) — classification of an object described by (a_0, …, a_p)."""

def classify(query_a, A, D, y):
    """query_a: gradations (a_0..a_p) of the TUPLAM features; A[i]: training gradations; D[i]: latent (d_i1..d_ip).
    Returns dict(class, score1, score2, B1 per step, B2 per step)."""
    n1, n2 = y.count(1), y.count(2); p = len(query_a) - 1
    B1 = [i for i in range(len(y)) if y[i] == 1 and A[i][0] == query_a[0]]
    B2 = [i for i in range(len(y)) if y[i] == 2 and A[i][0] == query_a[0]]
    steps = [(list(B1), list(B2))]
    for j in range(1, p + 1):
        B1 = [i for i in B1 if A[i][j] == query_a[j] and D[i][j - 1] > 0]
        B2 = [i for i in B2 if A[i][j] == query_a[j] and D[i][j - 1] < 0]
        steps.append((list(B1), list(B2)))
    s1, s2 = len(B1) / n1, len(B2) / n2
    cls = 1 if s1 > s2 else (2 if s1 < s2 else 0)
    return dict(cls=cls, s1=s1, s2=s2, steps=steps)
