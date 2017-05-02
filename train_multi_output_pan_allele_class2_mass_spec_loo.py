from pepnet import SequenceInput, Output, Predictor
from pepnet.sequence_helpers import group_similar_sequences
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop

from helpers import to_ic50, from_ic50
from data import (
    load_mass_spec_hits,
    generate_negatives_from_proteome,
    load_pseudosequences)
from callback_auc import CallbackAUC

from seaborn import plt


N_EPOCHS = 10
TRAINING_DECOY_FACTOR = 8
DECOY_WEIGHT_FOR_QUANTITATIVE_ASSAYS = 0.01
TEST_DECOY_FACTOR = 99
MASS_SPEC_OUTPUT_NAME = "neon mass spec"
BATCH_SIZE = 32

INITIAL_LEARNING_RATE = RMSprop().lr.get_value() * 1.25
print(INITIAL_LEARNING_RATE)
LEARNING_DECAY_RATE = 0.9

LOSS = "mse"   # "binary_crossentropy"
MERGE = "multiply"

def make_model(output_names):
    mhc = SequenceInput(
        length=34,
        name="mhc",
        encoding="index",
        variable_length=True,
        embedding_dim=32,
        embedding_mask_zero=False,
        dense_layer_sizes=[32],
        dense_activation="tanh",
        dense_batch_normalization=True,
        dense_dropout=0.15)

    peptide = SequenceInput(
        length=45,
        name="peptide",
        encoding="index",
        add_start_tokens=True,
        add_stop_tokens=True,
        embedding_dim=32,
        embedding_mask_zero=True,
        variable_length=True,
        conv_filter_sizes=[9],
        conv_activation="relu",
        conv_output_dim=32,
        conv_dropout=0.25,
        conv_batch_normalization=True,
        n_conv_layers=2,
        # conv_weight_source=mhc,
        global_pooling=True,
        global_pooling_batch_normalization=True,
        global_pooling_dropout=0.25,
        dense_layer_sizes=[32],
        dense_activation="sigmoid",
        dense_batch_normalization=True,
        dense_dropout=0.15)

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
            activation=activation,
            loss=LOSS)
        print(output)
        outputs.append(output)
    return Predictor(
        inputs=[mhc, peptide],
        outputs=outputs,
        merge_mode=MERGE,
        training_metrics=["accuracy"])

def plot_aucs(test_name, train_aucs, test_aucs):
    fig = plt.figure(figsize=(8, 8))
    axes = fig.gca()
    axes.plot(
        np.arange(N_EPOCHS), train_aucs)
    axes.plot(
        np.arange(N_EPOCHS), test_aucs)
    plt.xlabel("epoch")
    plt.ylabel("AUC")
    plt.xlim(0, 15)
    plt.ylim(0.5, 1.0)
    plt.legend(["train", "test (%s)" % test_name])
    fig.savefig("auc_%s.png" % test_name)

def augment_with_decoys(
        hit_peptides,
        hit_mhc_alleles,
        hit_weights,
        decoy_multiple=TRAINING_DECOY_FACTOR):
    n_hits = len(hit_peptides)
    decoy_peptides = generate_negatives_from_proteome(
        hit_peptides, factor=decoy_multiple)

    n_decoys = len(decoy_peptides)
    assert n_decoys == int(n_hits * decoy_multiple)
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
    hits_to_decoys = hit_weights.sum() / float(n_decoys)
    weights[:n_hits] = hit_weights
    weights[n_hits:] = min(1.0, hits_to_decoys)
    return mass_spec_peptides, mass_spec_mhc_alleles, Y_mass_spec, weights

def shuffle_data(peptides, alleles, Y, weights):
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
    return peptides, alleles, Y, weights

def learning_rate_schedule(epoch):
    lr = INITIAL_LEARNING_RATE * LEARNING_DECAY_RATE ** epoch
    print("-- setting learning rate for epoch %d to %f" % (epoch, lr))
    return lr

def main():
    mhc_pseudosequences_dict = load_pseudosequences()

    hits_dict = load_mass_spec_hits()
    hit_peptides = []
    hit_mhc_alleles = []
    hit_weights = []
    for (allele, peptides) in hits_dict.items():
        shuffled_peptides, _, weights = group_similar_sequences(peptides)
        assert len(shuffled_peptides) == len(peptides), \
            "Exepcted %d peptides but got back %d" % (
                len(peptides),
                len(shuffled_peptides))
        hit_peptides.extend(shuffled_peptides)
        hit_mhc_alleles.extend([allele] * len(peptides))
        hit_weights.extend(weights)
    n_hits = len(hit_peptides)
    assert set(hit_mhc_alleles) == set(hits_dict.keys())
    hit_weights = np.array(hit_weights)

    mass_spec_peptides, mass_spec_mhc_alleles, Y_mass_spec, weights = \
        augment_with_decoys(
            hit_peptides=hit_peptides,
            hit_mhc_alleles=hit_mhc_alleles,
            hit_weights=hit_weights)

    n_mass_spec = len(mass_spec_peptides)

    mass_spec_peptides, mass_spec_mhc_alleles, Y_mass_spec, weights = \
        shuffle_data(
            peptides=mass_spec_peptides,
            alleles=mass_spec_mhc_alleles,
            Y=Y_mass_spec,
            weights=weights)

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

        train_auc_callback = CallbackAUC(
            name="train",
            peptides=train_peptides,
            mhc_seqs=train_mhc_seqs,
            weights=train_weights,
            labels=Y_train,
            predictor=predictor)
        test_auc_callback = CallbackAUC(
            name="test",
            peptides=test_peptides,
            mhc_seqs=test_mhc_seqs,
            weights=test_weights,
            labels=Y_test,
            predictor=predictor)

        callbacks = [
            train_auc_callback,
            test_auc_callback,
            LearningRateScheduler(learning_rate_schedule)
        ]

        predictor.fit(
            {
                "peptide": train_peptides,
                "mhc": train_mhc_seqs
            },
            Y_train,
            epochs=N_EPOCHS,
            sample_weight=train_weights,
            callbacks=callbacks,
            batch_size=BATCH_SIZE)

        plot_aucs(
            test_name=left_out_allele,
            train_aucs=train_auc_callback.aucs[MASS_SPEC_OUTPUT_NAME],
            test_aucs=test_auc_callback.aucs[MASS_SPEC_OUTPUT_NAME])

        combined_peptides_for_ppv = test_peptides + extra_decoy_peptides_for_ppv
        combined_mhc_seqs_for_ppv = test_mhc_seqs + extra_decoy_mhc_seqs_for_ppv
        Y_combined_for_ppv = np.zeros(len(combined_peptides_for_ppv))
        Y_combined_for_ppv[:len(Y_test)] = Y_test
        Y_pred_for_ppv_dict = predictor.predict_scores({
            "peptide": combined_peptides_for_ppv,
            "mhc": combined_mhc_seqs_for_ppv})

        for output_name in predictor.output_names:
            print("-- %s" % output_name)
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
