"""Hierarchical agglomerative grouping (HAG) of the article, Steps 1-5.

Inputs : contribution matrix C[t][u] = eta_u(a_tu) (objects x features), weights w[u], classes y[t] in {1, 2}.
Params : alpha (0<alpha<1), delta (0<delta<0.5), kappa (max |TUPLAM|), cr1 = 10,
         phi = logistic sigmoid, sign +1 for K1 and -1 for K2  (b <- b +/- alpha * phi(-b)).
Switches (template vs article):
  centres = 'final'   : M1, M2 are the class means of b over all objects (article and the template's formula box)
            'running' : the template's Excel cells, where |b_t - M| uses the running partial mean up to row t
  step4_passes = 1    : Step 4 recomputes R = R + eta_q and applies the majorizer once (article)
                 2    : the template applies the majorizer again to the already-majorized b of Step 3
"""
import math

def phi(x):                       # logistic sigmoid σ(x) = 1 / (1 + e^-x)
    return 1.0 / (1.0 + math.exp(-x))

def majorize(b, cls, alpha):
    return b + (1 if cls == 1 else -1) * alpha * phi(-b)

def theta_gamma(b, y, centres='final'):
    n1 = y.count(1); n2 = y.count(2)
    if centres == 'final':
        M1 = sum(v for v, c in zip(b, y) if c == 1) / n1
        M2 = sum(v for v, c in zip(b, y) if c == 2) / n2
        th = sum(abs(v - (M1 if c == 1 else M2)) for v, c in zip(b, y))
        ga = sum(abs(v - (M2 if c == 1 else M1)) for v, c in zip(b, y))
        return th, ga, M1, M2
    s1 = s2 = th = ga = 0.0                      # template cells: running partial means
    for v, c in zip(b, y):
        if c == 1: s1 += v
        else: s2 += v
        m1, m2 = s1 / n1, s2 / n2
        th += abs(v - (m1 if c == 1 else m2)); ga += abs(v - (m2 if c == 1 else m1))
    return th, ga, s1 / n1, s2 / n2

def hag(C, w, y, alpha=0.3, delta=0.1, kappa=5, cr1_init=10.0, centres='final', step4_passes=1, majorizer=True):
    m, n = len(C), len(w)
    P = list(range(n))
    u = max(P, key=lambda j: (w[j], -j))         # Step 2: argmax weight, first index on ties
    tuplam = [u]; P.remove(u)
    R = [C[t][u] for t in range(m)]              # R(S_t) = eta_u(a_tu)
    trace = dict(first=u, iterations=[], R0=list(R))
    latent = []
    while True:
        cands = []
        cr1, q = cr1_init, None
        for cu in P:                              # Step 3
            b = [R[t] + C[t][cu] for t in range(m)]
            bm = [majorize(b[t], y[t], alpha) if majorizer else b[t] for t in range(m)]
            th, ga, M1, M2 = theta_gamma(bm, y, centres)
            ratio = th / ga
            cands.append(dict(u=cu, b=b, bm=bm, theta=th, gamma=ga, M1=M1, M2=M2, ratio=ratio))
            if ratio < cr1: cr1, q = ratio, cu
        if q is None: break
        crit = cr1                                # Step 4
        P.remove(q); tuplam.append(q)
        sel = next(c for c in cands if c['u'] == q)
        if step4_passes == 2 and majorizer:
            Rn = [majorize(v, y[t], alpha) for t, v in enumerate(sel['bm'])]
        else:
            Rn = [majorize(R[t] + C[t][q], y[t], alpha) if majorizer else R[t] + C[t][q] for t in range(m)]
        R = Rn; latent.append(list(R))
        go = len(tuplam) < kappa and crit > delta and len(P) > 0
        trace['iterations'].append(dict(cands=cands, q=q, crit=crit, R=list(R), size=len(tuplam), go_on=go))
        if not go: break
    return dict(tuplam=tuplam, latent=latent, trace=trace)
