"""Fit the CS-model on a training sample and represent / classify new objects without their class."""
from . import core
from .hag import hag, majorize
from .meta import classify

DEFAULTS = dict(alpha=0.3, delta=0.1, kappa=5, cr1=10.0, centres='running', step4_passes=2)

def fit(X, y, types, **kw):
    P = dict(DEFAULTS, **kw)
    m = len(X)
    lo, hi = core.scale_params(X, types)
    Z = [core.unify(x, types, lo, hi) for x in X]
    ks, k_max = core.permitted_k(y)
    ops = {}
    for op, _ in core.OPERATORS:
        D = [[core.rho(Z[i], Z[j], X[i], X[j], types, op) for j in range(m)] for i in range(m)]
        order = [core.neighbours(D[i], exclude=i) for i in range(m)]
        ops[op] = dict(D=D, order=order)
    feats = []                                    # Ψ(r): one synthetic feature per (operator, k)
    for op, _ in core.OPERATORS:
        for k in ks:
            order = ops[op]['order']
            same = [sum(1 for j in order[i][:k] if y[j] == y[i]) for i in range(m)]         # μ (training side)
            c1 = [core.chi1(order[i], y, k) for i in range(m)]
            a = [core.binary_a(c1[i], k) for i in range(m)]                                # formula (5)
            fk = core.membership(same, y, k)
            g = core.stability(fk, m, k)
            G, q1, q2 = core.boundary(fk)
            w = core.informativeness(same, y, fk, G)
            eta = core.contributions(a, y, w)
            feats.append(dict(op=op, k=k, same=same, chi1=c1, a=a, f=fk, g=g, G=G, q1=q1, q2=q2, w=w, eta=eta,
                              contrib=[eta[v]['eta'] for v in a]))
    C = [[f['contrib'][t] for f in feats] for t in range(m)]
    W = [f['w'] for f in feats]
    H = hag(C, W, y, P['alpha'], P['delta'], P['kappa'], P['cr1'], P['centres'], P['step4_passes'])
    T = H['tuplam']
    A = [[feats[u]['a'][t] for u in T] for t in range(m)]
    Dl = [[H['latent'][j][t] for j in range(len(H['latent']))] for t in range(m)]
    # latent features without majorizer (same TUPLAM order): cumulative generalized estimate Σ η
    Dplain = [[sum(C[t][u] for u in T[:j + 2]) for j in range(len(T) - 1)] for t in range(m)]
    return dict(P=P, X=X, y=y, types=types, lo=lo, hi=hi, Z=Z, ks=ks, k_max=k_max, ops=ops, feats=feats,
                C=C, W=W, hag=H, tuplam=T, A=A, D=Dl, Dplain=Dplain)

def represent(model, x, exclude=None):
    """Synthetic-feature description of an object relative to the training sample — no class label used."""
    z = core.unify(x, model['types'], model['lo'], model['hi'])
    out = {}
    for op, _ in core.OPERATORS:
        d = [core.rho(z, model['Z'][j], x, model['X'][j], model['types'], op) for j in range(len(model['X']))]
        order = core.neighbours(d, exclude=exclude)
        out[op] = dict(d=d, order=order)
    a_all = []
    for f in model['feats']:
        c1 = core.chi1(out[f['op']]['order'], model['y'], f['k'])
        a_all.append(core.binary_a(c1, f['k']))
    return dict(ctx=out, a_all=a_all, a=[a_all[u] for u in model['tuplam']])

def predict(model, x, exclude=None):
    rep = represent(model, x, exclude)
    res = classify(rep['a'], model['A'], model['D'], model['y'])
    res['rep'] = rep
    return res
