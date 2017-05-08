import csv

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from pepnet import Predictor, SequenceInput, Output

from helpers import shuffle_data
from data import (
    load_mass_spec_hits_and_decoys_grouped_by_nested_sets,
    load_pseudosequences
)

N_CV_SPLITS = 3
DECOY_FACTOR = 10
MIN_EPOCHS = 1
MAX_EPOCHS = 75

def make_predictors(
        widths=[9],
        layer_sizes=[32],
        n_conv_layers=[2],
        conv_dropouts=[0]):
    return {
        (width, layer_size, n_layers, dropout): Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=45,
                add_start_tokens=True,
                add_stop_tokens=True,
                variable_length=True,
                conv_filter_sizes=[width],
                n_conv_layers=n_layers,
                conv_output_dim=layer_size,
                conv_dropout=dropout,
                global_pooling=True),
            outputs=Output(1, activation="sigmoid"))
        for width in widths
        for layer_size in layer_sizes
        for n_layers in n_conv_layers
        for dropout in conv_dropouts
    }


def groupby_allele(mhc_alleles):
    allele_to_indices = {}
    unique_alleles = set(mhc_alleles)
    for allele in unique_alleles:
        indices = []
        for i, ai in enumerate(mhc_alleles):
            if ai == allele:
                indices.append(i)
        allele_to_indices[allele] = np.array(indices)
    return allele_to_indices

if __name__ == "__main__":
    peptides, mhc_alleles, Y, weights, group_ids = \
        load_mass_spec_hits_and_decoys_grouped_by_nested_sets(
            decoy_multiple=DECOY_FACTOR)
    peptides, mhc_alleles, Y, weights, group_ids = \
        shuffle_data(
            peptides=peptides,
            alleles=mhc_alleles,
            Y=Y,
            weights=weights,
            group_ids=group_ids)
    mhc_pseudosequences_dict = load_pseudosequences()
    mhc_seqs = [
        mhc_pseudosequences_dict[allele]
        for allele in mhc_alleles
    ]

    allele_to_indices = groupby_allele(mhc_alleles)

    cv = GroupKFold(n_splits=N_CV_SPLITS)

    with open('scores_conv_saturation_larger_models.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_training",
            "width", "layer_size", "n_layers", "dropout",
            "allele", "fold", "auc", "epochs",
            "n_hits"])
        writer.writeheader()
        for allele, indices in allele_to_indices.items():
            peptides_allele = [peptides[i] for i in indices]
            mhc_seqs_allele = [mhc_seqs[i] for i in indices]
            Y_allele = Y[indices]
            weights_allele = weights[indices]
            group_ids_allele = group_ids[indices]

            for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(
                        X=peptides_allele,
                        y=Y_allele,
                        groups=group_ids_allele)):
                peptides_allele_train = [peptides_allele[i] for i in train_idx]
                peptides_allele_test = [peptides_allele[i] for i in test_idx]
                mhc_seqs_allele_train = [mhc_seqs_allele[i] for i in train_idx]
                mhc_seqs_allele_test = [mhc_seqs_allele[i] for i in test_idx]
                Y_allele_train = Y_allele[train_idx]
                Y_allele_test = Y_allele[test_idx]
                weights_allele_train = weights_allele[train_idx]
                weights_allele_test = weights_allele[test_idx]

                for n_training in np.linspace(
                        100,
                        len(peptides_allele_train),
                        num=10,
                        dtype=int):
                    epochs = int(np.ceil(1000 ** 5 / n_training))
                    if epochs < MIN_EPOCHS:
                        epochs = MIN_EPOCHS
                    if epochs > MAX_EPOCHS:
                        epochs = MAX_EPOCHS
                    predictor_dict = make_predictors()
                    for key in sorted(predictor_dict.keys()):
                        model = predictor_dict[key]
                        (width, layer_size, n_conv_layers, dropout) = key
                        row_dict = {
                            "width": width,
                            "layer_size": layer_size,
                            "n_layers": n_conv_layers,
                            "dropout": dropout,
                            "n_training": n_training,
                            "allele": allele,
                            "fold": fold_idx,
                            "epochs": epochs,
                            "n_hits": Y_allele_train[:n_training].sum()
                        }
                        print("==> Training %s" % (row_dict,))
                        model.fit({
                            "peptide": peptides_allele_train[:n_training],
                            "mhc": mhc_seqs_allele_train[:n_training]},
                            Y_allele_train[:n_training],
                            sample_weight=weights_allele_train[:n_training],
                            epochs=epochs)
                        pred = model.predict({
                            "peptide": peptides_allele_test,
                            "mhc": mhc_seqs_allele_test})
                        auc = roc_auc_score(
                            y_true=Y_allele_test,
                            y_score=pred,
                            sample_weight=weights_allele_test)
                        print("==> %s %d/%d %s: %0.4f" % (
                            allele, fold_idx + 1, N_CV_SPLITS, row_dict, auc))
                        writer.writerow(row_dict)
                        f.flush()
