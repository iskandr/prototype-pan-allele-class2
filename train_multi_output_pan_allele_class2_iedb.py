from pepnet import SequenceInput, Output, Predictor
import numpy as np

from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import random

from helpers import to_ic50, from_ic50
from data import (
    load_iedb_binding_data,
    load_mass_spec,
    generate_negatives_from_proteome,
    load_pseudosequences)

N_EPOCHS = 10
TRAINING_DECOY_FACTOR = 0.5
DECOY_WEIGHT_FOR_QUANTITATIVE_ASSAYS = 0.01
TEST_DECOY_FACTOR = 9
MIN_ASSAY_COUNT = 10000
ONLY_HLA_DR = False
ASSAY = None  # "half maximal inhibitory concentration (IC50):purified MHC/competitive/radioactivity"


def make_model(sufficiently_large_output_names):
    mhc = SequenceInput(
        length=34,
        name="mhc",
        encoding="index",
        variable_length=True,
        embedding_dim=20,
        embedding_mask_zero=False,
        dense_layer_sizes=[64],
        dense_batch_normalization=True)

    peptide = SequenceInput(
        length=50,
        name="peptide",
        encoding="index",
        embedding_dim=20,
        embedding_mask_zero=True,
        variable_length=True,
        conv_filter_sizes=[1, 9, 10],
        conv_activation="relu",
        conv_output_dim=32,
        n_conv_layers=2,
        # conv_weight_source=mhc,
        global_pooling=True,
        global_pooling_batch_normalization=True)

    outputs = []
    for output_name in sufficiently_large_output_names:
        if "IC50" in output_name or "EC50" in output_name:
            transform = from_ic50
            inverse = to_ic50
            activation = "sigmoid"
        elif "half life" in output_name:
            transform = (lambda x: np.log10(x + 1))
            inverse = (lambda x: (10.0 ** x) - 1)
            activation = "relu"
        else:
            transform = None
            inverse = None
            activation = "sigmoid"
        output = Output(
            name=output_name,
            transform=transform,
            inverse_transform=inverse,
            activation=activation)
        print(output)
        outputs.append(output)
    return Predictor(
        inputs=[mhc, peptide],
        outputs=outputs,
        merge_mode="concat",
        dense_layer_sizes=[32],
        dense_activation="tanh",
        dense_batch_normalization=True)

def main():
    mhc_pseudosequences_dict = load_pseudosequences()

    hit_peptides, hit_mhc_seqs, decoy_peptides, decoy_mhc_sequences = \
        load_mass_spec(mhc_pseudosequences_dict, decoy_factor=TEST_DECOY_FACTOR)  # restrict_allele="HLA-DRA10101-DRB10101")

    # Mass spec validation set
    mass_spec_test_peptides = hit_peptides + decoy_peptides
    mass_spec_test_mhc_sequences = hit_mhc_seqs + decoy_mhc_sequences
    Y_mass_spec = np.zeros(len(mass_spec_test_peptides), dtype="int32")
    Y_mass_spec[:len(hit_peptides)] = 1
    assert Y_mass_spec.sum() == len(hit_peptides)

    df = load_iedb_binding_data()  # restrict_allele="HLA-DRA10101-DRB10101")
    print("Allele names in binding data:\n%s" % (df.mhc.value_counts(),))

    # Restrict binding data to alleles for which we have pseudosequences
    valid_mhc_set = set(mhc_pseudosequences_dict.keys())
    valid_mhc_mask = df.mhc.isin(valid_mhc_set)

    print("Dropping %d rows without pseudosequences, invalid alleles:\n%s" % (
        len(df) - valid_mhc_mask.sum(),
        df[~valid_mhc_mask].mhc.value_counts()))
    df = df[valid_mhc_mask]

    if ONLY_HLA_DR:
        dr_mask = df.mhc.str.contains("DR")
        print("%d rows with HLA-DR" % dr_mask.sum())
        df = df[dr_mask]
    print("Kept %d rows" % (len(df)))

    df["output_name"] = df["assay_group"] + ":" + df["assay_method"]

    if ASSAY:
        n_old = len(df)
        df = df[df.output_name == ASSAY]
        print("Kept %d/%d rows with assay %s" % (len(df), n_old, ASSAY))

    output_counts = df.output_name.value_counts()

    sufficiently_large_output_counts = output_counts[output_counts >= MIN_ASSAY_COUNT]

    print("Keeping outputs: %s" % (sufficiently_large_output_counts,))

    sufficiently_large_output_names = set(sufficiently_large_output_counts.index)
    df_subset = df[df.output_name.isin(sufficiently_large_output_names)]

    print(df_subset.head())
    print("Number of measurements: %d" % len(df_subset))

    predictor = make_model(sufficiently_large_output_names)
    predictor.save_diagram()

    unique_pmhcs = set(zip(df_subset["peptide"], df_subset["mhc"]))
    print("Number of unique pMHCs: %d" % len(unique_pmhcs))

    pmhc_list = sorted(unique_pmhcs)
    random.shuffle(pmhc_list)

    iedb_peptides = [p for (p, _) in pmhc_list]
    iedb_mhc_names = [m for (_, m) in pmhc_list]
    iedb_mhc_sequences = [
        mhc_pseudosequences_dict[mhc_name]
        for mhc_name in iedb_mhc_names
    ]

    mhc_name_counts = Counter()
    for name in iedb_mhc_names:
        mhc_name_counts[name] += 1
    print("Top MHC names:\n%s" % (mhc_name_counts,))

    mhc_seq_counts = Counter()
    for seq in iedb_mhc_sequences:
        mhc_seq_counts[seq] += 1
    print("Top MHC seqs:\n%s" % (mhc_seq_counts,))

    n_unique_pmhc = len(pmhc_list)
    pmhc_index_dict = {key: i for (i, key) in enumerate(pmhc_list)}

    assert pmhc_index_dict[(iedb_peptides[100], iedb_mhc_names[100])] == 100

    output_name_list = [o.name for o in predictor.outputs]
    output_name_index_dict = {
        output_name: i for i, output_name in enumerate(output_name_list)}
    n_outputs = len(output_name_list)

    output_is_quantitative_dict = {
        output_name: any(
            [(substr in output_name) for substr in ("IC50", "EC50", "half life")])
        for output_name in output_name_list
    }

    print(output_is_quantitative_dict)

    sums = np.zeros((n_unique_pmhc, n_outputs), dtype="float64")
    counts = np.zeros_like(sums, dtype="float64")

    for i, (output_name, peptide, mhc, qual, meas) in enumerate(zip(
            df_subset.output_name, df_subset.peptide, df_subset.mhc,
            df_subset.qual, df_subset.meas)):
        row_idx = pmhc_index_dict[(peptide, mhc)]
        col_idx = output_name_index_dict[output_name]

        if output_is_quantitative_dict[output_name]:
            if np.isnan(meas):
                continue
            counts[row_idx, col_idx] += 1
            sums[row_idx, col_idx] += np.log10(1 + meas)
        else:
            counts[row_idx, col_idx] += 1
            sums[row_idx, col_idx] += qual.startswith("Positive")

    Y_iedb = sums / counts

    for name, col_idx in output_name_index_dict.items():
        if output_is_quantitative_dict[name]:
            Y_iedb[:, col_idx] = 10.0 ** Y_iedb[:, col_idx] - 1
        else:
            Y_iedb[:, col_idx] = Y_iedb[:, col_idx] > 0.5

    Y_iedb[counts == 0] = np.nan

    cv_iterator = StratifiedKFold(3)
    for train_idx, test_idx in cv_iterator.split(iedb_peptides, iedb_mhc_sequences):
        train_peptides = [iedb_peptides[i] for i in train_idx]
        test_peptides = [iedb_peptides[i] for i in test_idx]
        Y_train_iedb = Y_iedb[train_idx, :]

        train_mhc_seqs = [iedb_mhc_sequences[i] for i in train_idx]
        test_mhc_seqs = [iedb_mhc_sequences[i] for i in test_idx]
        Y_test_iedb = Y_iedb[test_idx, :]

        assert len(train_mhc_seqs) == len(train_peptides) == len(Y_train_iedb)
        assert len(test_mhc_seqs) == len(test_peptides) == len(Y_test_iedb)
        for epoch in range(N_EPOCHS):
            print("--- EPOCH %d/%d" % (epoch + 1, N_EPOCHS))
            negative_training_peptides = generate_negatives_from_proteome(
                peptides=train_peptides,
                factor=TRAINING_DECOY_FACTOR)
            # maintain the proportion of MHCs to not create a bias that rare MHCs are
            # more likely to produce non-binders
            negative_training_mhcs = list(np.random.choice(
                train_mhc_seqs,
                size=len(negative_training_peptides)))

            Y_training_negative = np.zeros((len(negative_training_peptides), len(predictor.outputs)))
            for i, output in enumerate(predictor.outputs):
                if output.inverse_transform:
                    Y_training_negative[:, i] = output.inverse_transform(Y_training_negative[:, i])

            combined_training_peptides = train_peptides + negative_training_peptides
            combined_training_mhc_sequences = train_mhc_seqs + negative_training_mhcs
            combined_training_output_values = np.vstack([Y_train_iedb, Y_training_negative])

            assert len(combined_training_peptides) == len(combined_training_mhc_sequences)
            assert len(combined_training_peptides) == len(combined_training_output_values)

            real_to_decoy_ratio = (
                len(Y_train_iedb) /
                float(1 + len(negative_training_peptides)))
            # print("Real sample to decoy ratio: %0.4f" % real_to_decoy_ratio)

            training_weights_dict = {}
            for o in predictor.outputs:
                output_name = o.name
                weights = np.ones(len(combined_training_peptides), dtype="float32")
                if output_is_quantitative_dict[output_name]:
                    weights[len(train_peptides):] = (
                        DECOY_WEIGHT_FOR_QUANTITATIVE_ASSAYS * real_to_decoy_ratio)
                else:
                    weights[len(train_peptides):] = real_to_decoy_ratio
                training_weights_dict[output_name] = weights

            predictor.fit({
                    "peptide": combined_training_peptides,
                    "mhc": combined_training_mhc_sequences},
                combined_training_output_values,
                epochs=1,
                sample_weight=training_weights_dict)

            Y_pred_mass_spec_dict = predictor.predict_scores({
                "peptide": mass_spec_test_peptides,
                "mhc": mass_spec_test_mhc_sequences})
            Y_pred_train_dict = predictor.predict_scores({
                "peptide": train_peptides,
                "mhc": train_mhc_seqs})
            Y_pred_test_dict = predictor.predict_scores({
                "peptide": test_peptides,
                "mhc": test_mhc_seqs})
            for output_name, Y_pred_mass_spec in Y_pred_mass_spec_dict.items():
                print("-- %s" % output_name)

                col_idx = output_name_index_dict[output_name]
                Y_train_iedb_curr_assay = Y_train_iedb[:, col_idx]
                Y_test_iedb_curr_assay = Y_test_iedb[:, col_idx]
                if "IC50" in output_name or "EC50" in output_name:
                    Y_train_label = Y_train_iedb_curr_assay <= 500
                    Y_test_label = Y_test_iedb_curr_assay <= 500
                elif "half life" in output_name:
                    Y_train_label = Y_train_iedb_curr_assay >= 120
                    Y_test_label = Y_test_iedb_curr_assay >= 120
                else:
                    assert not output_is_quantitative_dict[output_name], output_name
                    Y_train_label = Y_train_iedb_curr_assay > 0.5
                    Y_test_label = Y_test_iedb_curr_assay > 0.5

                print("----> Training AUC=%0.4f" % (roc_auc_score(
                    y_true=Y_train_label,
                    y_score=Y_pred_train_dict[output_name]),))
                print("----> Test Set AUC=%0.4f" % (roc_auc_score(
                    y_true=Y_test_label,
                    y_score=Y_pred_test_dict[output_name]),))

                auc_mass_spec = roc_auc_score(
                    y_true=Y_mass_spec,
                    y_score=Y_pred_mass_spec)
                descending_indices = np.argsort(-Y_pred_mass_spec)
                n_hits = Y_mass_spec.sum()
                ppv_mass_spec = Y_mass_spec[descending_indices[:n_hits]].mean()
                print("----> Mass Spec %s AUC=%0.4f, PPV=%0.4f" % (
                    output_name,
                    auc_mass_spec,
                    ppv_mass_spec))

if __name__ == "__main__":
    main()
