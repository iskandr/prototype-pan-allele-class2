import numpy as np


def from_ic50(ic50):
    x = 1.0 - (np.log(ic50) / np.log(50000))
    return np.minimum(
        1.0,
        np.maximum(0.0, x))

def to_ic50(x):
    return 50000.0 ** (1.0 - x)

assert np.allclose(to_ic50(from_ic50(40)), 40)
assert from_ic50(50000) < 0.00001, from_ic50(50000)
assert from_ic50(1) > 0.999, from_ic50(1)
assert to_ic50(0) > 40000, to_ic50(0)
assert to_ic50(1) <= 1, to_ic50(1)

def shuffle_data(peptides, alleles, Y, weights, group_ids=None):
    n = len(peptides)
    assert len(alleles) == n
    assert len(Y) == n
    assert len(weights) == n
    # shuffle training set
    shuffle_indices = np.arange(n)
    np.random.shuffle(shuffle_indices)
    peptides = [peptides[i] for i in shuffle_indices]
    alleles = [alleles[i] for i in shuffle_indices]
    Y = Y[shuffle_indices]
    weights = weights[shuffle_indices]
    if group_ids is None:
        return peptides, alleles, Y, weights
    else:
        group_ids = np.array(group_ids)
        group_ids = group_ids[shuffle_indices]
        return peptides, alleles, Y, weights, group_ids
