"""Evaluation helpers: confusion counts, accuracy/precision/recall/F1, AUC, margins, k-NN baseline."""

def confusion(pred, y, pos=1):
    tp = sum(1 for p, t in zip(pred, y) if t == pos and p == pos)
    fn = sum(1 for p, t in zip(pred, y) if t == pos and p not in (pos,))
    fp = sum(1 for p, t in zip(pred, y) if t != pos and p == pos)
    tn = sum(1 for p, t in zip(pred, y) if t != pos and p not in (pos, 0))
    ref = sum(1 for p in pred if p == 0)
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, refusals=ref)

def scores(pred, y):
    n = len(y); acc = sum(p == t for p, t in zip(pred, y)) / n
    cov = sum(p != 0 for p in pred) / n
    out = dict(accuracy=acc, coverage=cov)
    for c in (1, 2):
        tp = sum(1 for p, t in zip(pred, y) if p == c and t == c); pp = sum(1 for p in pred if p == c); ap = sum(1 for t in y if t == c)
        pr = tp / pp if pp else 0.0; rc = tp / ap if ap else 0.0
        out[c] = dict(precision=pr, recall=rc, f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return out

def auc(score, y, pos=1):
    score = [round(s, 10) for s in score]          # equal scores must tie exactly (0.5 − 2/3 vs 0 − 1/6)
    P = [s for s, t in zip(score, y) if t == pos]; N = [s for s, t in zip(score, y) if t != pos]
    return sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in P for b in N) / (len(P) * len(N))

def margins(v, y):
    lo1 = min(a for a, c in zip(v, y) if c == 1); hi2 = max(a for a, c in zip(v, y) if c == 2)
    b = (lo1 + hi2) / 2
    return dict(b=b, width=lo1 - hi2, m=[(1 if c == 1 else -1) * (a - b) for a, c in zip(v, y)],
                pred=[1 if a > b else 2 for a in v])

def knn_vote(order, ytr, k):
    c1 = sum(1 for j in order[:k] if ytr[j] == 1)
    return 1 if c1 > k // 2 else 2
