from pepnet import SequenceInput, Output, Predictor
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


from helpers import to_ic50, from_ic50
from data import (
    load_mass_spec_hits,
    generate_negatives_from_proteome,
    load_pseudosequences)

N_EPOCHS = 5
TRAINING_DECOY_FACTOR = 5
DECOY_WEIGHT_FOR_QUANTITATIVE_ASSAYS = 0.01
TEST_DECOY_FACTOR = 9
MASS_SPEC_OUTPUT_NAME = "neon mass spec"

def make_model(output_names):
    mhc = SequenceInput(
        length=34,
        name="mhc",
        encoding="index",
        variable_length=True,
        embedding_dim=20,
        embedding_mask_zero=False,
        dense_layer_sizes=[32],
        dense_activation="tanh",
        dense_batch_normalization=True)

    peptide = SequenceInput(
        length=50,
        name="peptide",
        encoding="index",
        embedding_dim=20,
        embedding_mask_zero=True,
        variable_length=True,
        conv_filter_sizes=[9],
        conv_activation="relu",
        conv_output_dim=32,
        n_conv_layers=2,
        # conv_weight_source=mhc,
        global_pooling=True,
        global_pooling_batch_normalization=True,
        dense_layer_sizes=[32],
        dense_activation="sigmoid",
        dense_batch_normalization=True)

    outputs = []
    for output_name in output_names:
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
        merge_mode="multiply",
        # dense_layer_sizes=[32],
        # dense_activation="tanh",
        # dense_batch_normalization=True,
        training_metrics=["accuracy"])

def main():
    mhc_pseudosequences_dict = load_pseudosequences()

    hits_dict = load_mass_spec_hits()
    hit_peptides = []
    hit_mhc_alleles = []
    for (allele, peptides) in hits_dict.items():
        hit_peptides.extend(peptides)
        hit_mhc_alleles.extend([allele] * len(peptides))
    n_hits = len(hit_peptides)
    assert set(hit_mhc_alleles) == set(hits_dict.keys())

    decoy_peptides = generate_negatives_from_proteome(
        hit_peptides, factor=TRAINING_DECOY_FACTOR)

    n_decoys = len(decoy_peptides)
    assert n_decoys == int(n_hits * TRAINING_DECOY_FACTOR)
    decoy_mhc_alleles = list(np.random.choice(hit_mhc_alleles, size=n_decoys))

    # Mass spec validation set
    mass_spec_peptides = hit_peptides + decoy_peptides
    mass_spec_mhc_alleles = hit_mhc_alleles + decoy_mhc_alleles
    n_mass_spec = len(mass_spec_peptides)
    assert n_mass_spec == n_hits + n_decoys

    Y_mass_spec = np.zeros(len(mass_spec_peptides), dtype="int32")
    Y_mass_spec[:len(hit_peptides)] = 1
    assert Y_mass_spec.sum() == len(hit_peptides)
    weights = np.ones(n_mass_spec, dtype="float32")
    hits_to_decoys = n_hits / float(n_decoys)
    weights[n_hits:] = min(1.0, hits_to_decoys)

    # shuffle training set
    shuffle_indices = np.arange(n_mass_spec)
    np.random.shuffle(shuffle_indices)
    mass_spec_peptides = [mass_spec_peptides[i] for i in shuffle_indices]
    mass_spec_mhc_alleles = [mass_spec_mhc_alleles[i] for i in shuffle_indices]
    Y_mass_spec = Y_mass_spec[shuffle_indices]
    weights = weights[shuffle_indices]

    # get the pseudosequences for all samples
    mass_spec_mhc_seqs = [mhc_pseudosequences_dict[allele] for allele in mass_spec_mhc_alleles]

    predictor = make_model(output_names=[MASS_SPEC_OUTPUT_NAME])
    predictor.save_diagram()
    output_name_list = [o.name for o in predictor.outputs]

    output_is_quantitative_dict = {
        output_name: any(
            [(substr in output_name) for substr in ("IC50", "EC50", "half life")])
        for output_name in output_name_list
    }
    print("Which outputs are quantitative: %s" % (output_is_quantitative_dict,))

    # draw more random decoys to do PPV calculations
    extra_decoy_peptides_for_ppv = generate_negatives_from_proteome(
        hit_peptides, factor=TEST_DECOY_FACTOR - TRAINING_DECOY_FACTOR)
    extra_decoy_mhc_seqs_for_ppv = list(
        np.random.choice(
            mass_spec_mhc_seqs,
            size=len(extra_decoy_peptides_for_ppv)))

    cv_iterator = LeaveOneGroupOut()
    for train_idx, test_idx in cv_iterator.split(
            mass_spec_peptides, Y_mass_spec, groups=mass_spec_mhc_alleles):
        assert len(train_idx) < n_mass_spec
        assert len(test_idx) < n_mass_spec
        training_alleles = set([mass_spec_mhc_alleles[i] for i in train_idx])
        left_out_alleles = set([mass_spec_mhc_alleles[i] for i in test_idx])
        assert len(left_out_alleles) == 1, left_out_alleles
        overlapping_alleles = training_alleles.intersection(left_out_alleles)
        assert len(overlapping_alleles) == 0, overlapping_alleles
        left_out_allele = left_out_alleles.pop()
        print("\n\n===> Left out allele: %s" % left_out_allele)
        print("===> Training alleles: %s" % (training_alleles,))

        train_peptides = [mass_spec_peptides[i] for i in train_idx]
        test_peptides = [mass_spec_peptides[i] for i in test_idx]

        Y_train = Y_mass_spec[train_idx]
        Y_test = Y_mass_spec[test_idx]

        train_weights = weights[train_idx]
        test_weights = weights[test_idx]

        train_mhc_seqs = [mass_spec_mhc_seqs[i] for i in train_idx]
        test_mhc_seqs = [mass_spec_mhc_seqs[i] for i in test_idx]

        assert len(train_peptides) == len(Y_train) == len(train_weights) == len(train_mhc_seqs)
        assert len(test_peptides) == len(Y_test) == len(test_weights) == len(test_mhc_seqs)

        for epoch in range(N_EPOCHS):
            print("--- EPOCH %d/%d" % (epoch + 1, N_EPOCHS))
            predictor.fit({
                    "peptide": train_peptides,
                    "mhc": train_mhc_seqs},
                Y_train,
                epochs=1,
                sample_weight={o.name: train_weights for o in predictor.outputs},
                validation_data=({"peptide": test_peptides, "mhc": test_mhc_seqs}, Y_test, test_weights))

            Y_pred_train_dict = predictor.predict_scores({
                "peptide": train_peptides,
                "mhc": train_mhc_seqs})
            Y_pred_test_dict = predictor.predict_scores({
                "peptide": test_peptides,
                "mhc": test_mhc_seqs})

            combined_peptides_for_ppv = test_peptides + extra_decoy_peptides_for_ppv
            combined_mhc_seqs_for_ppv = test_mhc_seqs + extra_decoy_mhc_seqs_for_ppv
            Y_combined_for_ppv = np.zeros(len(combined_peptides_for_ppv))
            Y_combined_for_ppv[:len(Y_test)] = Y_test
            Y_pred_for_ppv_dict = predictor.predict_scores({
                "peptide": combined_peptides_for_ppv,
                "mhc": combined_mhc_seqs_for_ppv})
            for output_name, Y_pred_train in Y_pred_train_dict.items():
                print("-- %s" % output_name)
                Y_pred_test = Y_pred_test_dict[output_name]

                print("----> Training AUC=%0.4f" % (roc_auc_score(
                    y_true=Y_train,
                    y_score=Y_pred_train,
                    sample_weight=train_weights)))
                print("----> Test Set AUC=%0.4f" % (roc_auc_score(
                    y_true=Y_test,
                    y_score=Y_pred_test,
                    sample_weight=test_weights),))

                Y_pred_for_ppv = Y_pred_for_ppv_dict[output_name]
                descending_indices = np.argsort(-Y_pred_for_ppv)
                n_hits = Y_test.sum()
                ppv = Y_combined_for_ppv[descending_indices[:n_hits]].mean()
                print("----> PPV @ %dX decoys for allele=%s %0.4f" % (
                    TEST_DECOY_FACTOR,
                    left_out_allele,
                    ppv))


if __name__ == "__main__":
    main()
