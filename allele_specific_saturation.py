import csv

import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from keras.callbacks import EarlyStopping


from pepnet import Predictor, SequenceInput, Output

from helpers import shuffle_data
from data import (
    load_mass_spec_hits_and_decoys_grouped_by_nested_sets,
    load_pseudosequences
)

N_CV_SPLITS = 3
DECOY_FACTOR = 4
MIN_EPOCHS = 2
MAX_EPOCHS = 100
N_TRAINING_SIZES = 10

def make_predictors(
        widths=[9],
        layer_sizes=[8],
        n_conv_layers=[2],
        conv_dropouts=[0.25],
        conv_activation="relu",
        global_pooling_batch_normalization=True):
    return {
        (width, layer_size, n_layers, dropout): Predictor(
            inputs=SequenceInput(
                name="peptide",
                length=45,
                add_start_tokens=True,
                add_stop_tokens=True,
                variable_length=True,
                conv_filter_sizes=[1, 3, 5, 7, width, 11],
                n_conv_layers=n_layers,
                conv_output_dim=layer_size,
                conv_activation=conv_activation,
                conv_dropout=dropout,
                global_pooling=True,
                global_pooling_batch_normalization=global_pooling_batch_normalization),
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
            "n_pos", "n_pos_weighted"])
        writer.writeheader()
        for allele, indices in allele_to_indices.items():
            peptides_allele = [peptides[i] for i in indices]
            mhc_seqs_allele = [mhc_seqs[i] for i in indices]
            Y_allele = Y[indices]
            weights_allele = weights[indices]
            group_ids_allele = group_ids[indices]
            assert len(peptides_allele) == len(mhc_seqs_allele)
            assert len(peptides_allele) == len(Y_allele)
            assert len(peptides_allele) == len(weights_allele)
            assert len(peptides_allele) == len(group_ids_allele)

            for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(
                        X=peptides_allele,
                        y=Y_allele,
                        groups=group_ids_allele)):
                peptides_allele_train = [peptides_allele[i] for i in train_idx]
                peptides_allele_test = [peptides_allele[i] for i in test_idx]
                mhc_seqs_allele_train = [mhc_seqs_allele[i] for i in train_idx]
                mhc_seqs_allele_test = [mhc_seqs_allele[i] for i in test_idx]

                """
                group_ids_train = group_ids_allele[train_idx]
                group_ids_test = group_ids_allele[test_idx]
                TODO: adjust weights to only use groups in train set -OR-
                use group IDs for CV using increasing folds of a 10-fold
                CV iterator
                """
                Y_allele_train = Y_allele[train_idx]
                Y_allele_test = Y_allele[test_idx]

                weights_allele_train = weights_allele[train_idx]
                weights_allele_test = weights_allele[test_idx]

                assert len(peptides_allele_train) == len(mhc_seqs_allele_train)
                assert len(peptides_allele_train) == len(Y_allele_train)
                assert len(peptides_allele_train) == len(weights_allele_train)

                assert len(peptides_allele_test) == len(mhc_seqs_allele_test)
                assert len(peptides_allele_test) == len(Y_allele_test)
                assert len(peptides_allele_test) == len(weights_allele_test)

                for n_training in np.linspace(
                        300 + np.random.randint(0, 20),
                        len(peptides_allele_train),
                        num=N_TRAINING_SIZES,
                        dtype=int):
                    epochs = int(np.ceil(2.5 * 10 ** 5 / n_training))
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
                            "n_pos": Y_allele_train[:n_training].sum(),
                            "n_pos_weighted": weights_allele_train[:n_training][
                                Y_allele_train[:n_training]].sum()
                        }
                        print("==> Training %s" % (row_dict,))
                        early_stopping_callback = EarlyStopping(
                            monitor="loss",
                            patience=1)
                        print("-- n_training = %d, n_pos = %d, n_hit_loci = %d" % (
                            n_training,
                            Y_allele_train[:n_training].sum(),
                            weights_allele_train[:n_training][
                                Y_allele_train[:n_training]].sum()))
                        model.fit({
                            "peptide": peptides_allele_train[:n_training],
                            "mhc": mhc_seqs_allele_train[:n_training]},
                            Y_allele_train[:n_training],
                            sample_weight=weights_allele_train[:n_training],
                            epochs=epochs,
                            callbacks=[early_stopping_callback])
                        row_dict["epochs"] = early_stopping_callback.stopped_epoch
                        if row_dict["epochs"] == 0:
                            row_dict["epochs"] = epochs

                        pred = model.predict({
                            "peptide": peptides_allele_test,
                            "mhc": mhc_seqs_allele_test})
                        auc = roc_auc_score(
                            y_true=Y_allele_test,
                            y_score=pred,
                            sample_weight=weights_allele_test)
                        print("==> %s %d/%d %s: %0.4f" % (
                            allele, fold_idx + 1, N_CV_SPLITS, row_dict, auc))
                        row_dict["auc"] = auc
                        writer.writerow(row_dict)
                        f.flush()
