import numpy as np
def spearman(x, y):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        import collections
        d = collections.defaultdict(list)
        for i, val in enumerate(v): d[val].append(i)
        for val, idxs in d.items():
            if len(idxs) > 1:
                r[idxs] = np.mean(r[idxs])
        return r
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])
