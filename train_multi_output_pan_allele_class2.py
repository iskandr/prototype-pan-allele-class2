from pepnet import SequenceInput, Output, Predictor
import numpy as np
import pandas as pd
import os


def load_pseudosequences(filename="pseudosequences.dat"):
    mhc_pseudosequences_dict = {}
    with open(filename) as f:
        for line in f:
            mhc, seq = line.split()
            mhc_pseudosequences_dict[mhc] = seq
    return mhc_pseudosequences_dict

def load_binding_data(filename="class2_data.csv"):
    df = pd.read_csv(filename)
    print("Loaded %d samples" % len(df))
    bad_mhc = ~df.mhc.str.contains("\*")
    print("Dropping %d rows without full alleles" % bad_mhc.sum())
    df = df[~bad_mhc]
    # no_units = df.units.isnull()
    # print("Dropping %d rows without units" % no_units.sum())
    # df = df[~no_units]
    return df

def from_ic50(ic50):
    x = 1.0 - (np.log(ic50) / np.log(50000))
    return np.minimum(
        1.0,
        np.maximum(0.0, x))

def to_ic50(x):
    return 50000.0 ** (1.0 - x)

assert np.allclose(to_ic50(from_ic50(40)), 40)


def load_hits(filename=None):
    if not filename:
        filename = os.environ["CLASS_II_DATA"]
    df = pd.read_excel(filename)
    hits = {}
    for allele in df.columns:
        hits[allele] = [
            s.upper() for s in df[allele]
            if isinstance(s, str) and len(s) > 0 and "X" not in s]
        print("Loaded %d hits for %s (max length %d)" % (
            len(hits[allele]),
            allele,
            max(len(p) for p in hits[allele])))
    return hits

def main():
    mhc = SequenceInput(
        length=34, name="mhc", encoding="index", variable_length=True,
        dense_layer_sizes=[32],
        dense_activation="sigmoid",
        dense_dropout=0.1)

    peptide = SequenceInput(
        length=50, name="peptide", encoding="index", variable_length=True,
        conv_filter_sizes=[9],
        conv_output_dim=8,
        n_conv_layers=2,
        global_pooling=True,
        dense_layer_sizes=[32],
        dense_activation="tanh",
        dense_dropout=0.1)

    mhc_pseudosequences_dict = load_pseudosequences()

    df = load_binding_data()

    # Load mass spec hits
    hits = load_hits()
    print(hits.keys())
    hit_peptides = []
    hit_mhc_seqs = []
    for (allele, peptides) in hits.items():
        allele = "DRB1*%s:%s" % (allele[3:5], allele[5:7])
        print(allele)
        hit_peptides.extend(peptides)
        allele_seq = mhc_pseudosequences_dict[allele]
        hit_mhc_seqs.extend([allele_seq] * len(peptides))

    # Restrict binding data to alleles for which we have pseudosequences
    valid_mhc_set = set(mhc_pseudosequences_dict.keys())
    df.mhc = df.mhc.str.replace("/", "-").str.replace("HLA-DRB", "DRB")
    valid_mhc_mask = df.mhc.isin(valid_mhc_set)
    print("Dropping %d rows without pseudosequences" % (
        len(df) - valid_mhc_mask.sum()))
    df = df[valid_mhc_mask]
    print("Kept %d rows" % (len(df)))
    df["output_name"] = df["assay_group"] + ":" + df["assay_method"]
    output_counts = df.output_name.value_counts()

    sufficiently_large_output_counts = output_counts[output_counts >= 200]

    sufficiently_large_output_names = set(sufficiently_large_output_counts.index)
    df_subset = df[df.output_name.isin(sufficiently_large_output_names)]

    outputs = []

    for output_name in sufficiently_large_output_names:
        if "IC50" in output_name or "EC50" in output_name:
            transform = from_ic50
            inverse = to_ic50
            activation = "sigmoid"
        elif "half life" in output_name:
            transform = np.log10
            inverse = (lambda x: 10.0 ** x)
            activation = "linear"
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
    predictor = Predictor(
        inputs=[mhc, peptide],
        outputs=outputs,
        merge_mode="multiply")

    predictor.save_diagram()

    unique_pmhcs = set(zip(df_subset["peptide"], df_subset["mhc"]))
    pmhc_list = sorted(unique_pmhcs)

    peptides = [p for (p, _) in pmhc_list]
    mhc_names = [m for (_, m) in pmhc_list]

    n_unique_pmhc = len(pmhc_list)
    pmhc_index_dict = {key: i for (i, key) in enumerate(pmhc_list)}

    output_name_list = [o.name for o in outputs]
    output_name_index_dict = {output_name: i for i, output_name in enumerate(output_name_list)}
    n_outputs = len(output_name_list)

    output_is_quantitative_dict = {
        output_name: any([(substr in output_name) for substr in ("IC50", "EC50", "half life")])
        for output_name in output_name_list
    }

    print(output_is_quantitative_dict)

    sums = np.zeros((n_unique_pmhc, n_outputs), dtype="float64")
    counts = np.zeros_like(sums, dtype="float64")

    for (output_name, peptide, mhc, qual, meas) in zip(
            df_subset.output_name, df_subset.peptide, df_subset.mhc,
            df_subset.qual, df_subset.meas):
        row_idx = pmhc_index_dict[(peptide, mhc)]
        col_idx = output_name_index_dict[output_name]
        counts[row_idx, col_idx] += 1
        if output_is_quantitative_dict[output_name]:
            sums[row_idx, col_idx] += np.log(1 + meas)
        else:
            sums[row_idx, col_idx] += qual.startswith("Positive")

    averages = sums / counts

    for name, col_idx in output_name_index_dict.items():
        if output_is_quantitative_dict[name]:
            averages[:, col_idx] = np.exp(averages[:, col_idx]) - 1
        else:
            averages[:, col_idx] = averages[:, col_idx] > 0.5

    averages[counts == 0] = np.nan

    normalized_mhc_names = [
        mhc_name.replace("/", "-").replace("HLA-DRB", "DRB")
        for mhc_name in mhc_names
    ]
    mhc_inputs = [
        mhc_pseudosequences_dict[mhc_name] for mhc_name in normalized_mhc_names]

    predictor.fit({"peptide": peptides, "mhc": mhc_inputs}, averages, epochs=5)
    y = predictor.predict({"peptide": peptides, "mhc": mhc_inputs})

    hit_peptides.append("SFYQQQLML")
    hit_mhc_seqs.append(hit_mhc_seqs[-1])

    y_dict = predictor.predict({"peptide": hit_peptides, "mhc": hit_mhc_seqs})

    y = y_dict[outputs[0].name]

    print(y)

    print("min=%0.4f, %0.4f/%0.4f/%0.4f, max=%0.4f" % (
        y.min(),
        np.percentile(y, 25),
        np.percentile(y, 50),
        np.percentile(y, 75),
        y.max()))

    np.save("pred.npy", y)

if __name__ == "__main__":
    main()
