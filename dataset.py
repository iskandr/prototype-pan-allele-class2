import math

import numpy as np


class Dataset(object):
    """
    Training data for single allele of Class II MHC
    """
    def __init__(self, alleles, peptides, labels=None, weights=None, group_ids=None):
        n_peptides = len(peptides)

        if n_peptides != len(alleles):
            raise ValueError(
                "Length mismatch between peptides (%d) and alleles (%d)" % (
                    n_peptides,
                    len(alleles)))

        if labels is None:
            # if labels are not given then assume that the dataset is
            # comprised of "hits"
            labels = np.ones(n_peptides, dtype="bool")
        elif isinstance(labels, (float, int, bool)):
            labels = np.array([labels] * n_peptides)
        elif n_peptides != len(labels):
            raise ValueError(
                "Length mismatch between peptides (%d) and labels (%d)" % (
                    n_peptides,
                    len(labels)))

        if group_ids is None:
            # if sequence groups aren't given then put every
            # peptide in a singleton group
            group_ids = np.arange(n_peptides, dtype="int32")
        elif isinstance(group_ids, (float, int, bool)):
            group_ids = np.array([group_ids] * n_peptides)
        elif n_peptides != len(group_ids):
            raise ValueError(
                "Length mismatch between peptides (%d) and groups (%d)" % (
                    n_peptides,
                    len(group_ids)))

        if weights is None:
            # if weights aren't given then every element has the same
            # weight
            # TODO: use group_ids and labels to weigh samples inversely with
            # group size (separately for positive vs. negative labels)
            weights = np.ones(n_peptides, dtype="float32")
        elif isinstance(weights, (float, int)):
            weights = np.array([weights] * n_peptides)
        elif n_peptides != len(weights):
            raise ValueError(
                "Length mismatch between peptides (%d) and weights (%d)" % (
                    n_peptides,
                    len(labels)))

        self.alleles = alleles
        self.peptides = peptides
        self.labels = np.array(labels)
        self.weights = np.array(weights)
        self.group_ids = np.array(group_ids)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, indices):
        if isinstance(indices, slice):
            peptides = self.peptides[:indices]
            alleles = self.alleles[:indices]
            # for other fields we want to pull out the indices
            # implied by a slice
            index_array = np.arange(*indices.indices(len(self)))
        else:
            index_array = np.array(indices)
            if index_array.dtype == "bool":
                # replace boolean mask with indices of True entries
                index_array = np.where(index_array)[0]
            peptides = [self.peptides[i] for i in index_array]
            alleles = [self.alleles[i] for i in index_array]

        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=self.labels[index_array],
            weights=self.weights[index_array],
            group_ids=self.group_ids[index_array])

    def split(self, train_idx, test_idx):
        dataset_train = self[train_idx]
        dataset_test = self[test_idx]
        return dataset_train, dataset_test

    def random_subset(self, fraction=0.5):
        subset, _ = self.random_split(fraction=fraction)
        return subset

    def random_split(self, fraction=0.5):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        n_left = int(math.ceil(len(self) * fraction))
        subset_left = self[indices[:n_left]]
        subset_right = self[indices[n_left:]]
        return subset_left, subset_right

    def shuffle(self):
        left, right = self.random_split(fraction=1.0)
        if len(left) != len(self):
            raise ValueError("Expected %d elements but got %d after shuffle" % (
                len(self),
                len(left)))
        assert len(right) == 0
        return left

    def distinct_sequence_groups(self):
        return set(self.group_ids)

    def distinct_alleles(self):
        return set(self.alleles)

    def distinct_peptides(self):
        return set(self.peptides)

    def groupby_allele(self):
        for allele in self.distinct_alleles():
            mask = [a_i == allele for a_i in self.alleles]
            yield (allele, self[mask])

    def groupby_peptide(self):
        for peptide in self.distinct_peptides():
            mask = [p_i == peptide for p_i in self.peptides]
            yield (peptide, self[mask])

    def combine(self, other, preserve_group_ids=False):
        peptides = list(self.peptides) + list(other.peptides)
        alleles = list(self.alleles) + list(other.alleles)
        labels = np.concatenate([self.labels, other.labels])
        weights = np.concatenate([self.weights, other.weights])
        if not preserve_group_ids:
            offset = (self.group_ids.max() + 1)
            other_group_ids = offset + other.group_ids
        else:
            other_group_ids = other.group_ids
        group_ids = np.concatenate([self.group_ids, other_group_ids])
        return Dataset(
            alleles=alleles,
            peptides=peptides,
            labels=labels,
            weights=weights,
            group_ids=group_ids)
