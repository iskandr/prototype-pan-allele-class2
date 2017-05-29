from collections import Counter
import os

import pandas as pd
import numpy as np
from pyensembl import ensembl_grch38

from mhc_names import normalize_mhc_name
from pepnet.sequence_helpers import group_similar_sequences

from dataset import Dataset

def load_pseudosequences(filename="pseudosequences.dat"):
    mhc_pseudosequences_dict = {}
    with open(filename) as f:
        for line in f:
            mhc, seq = line.split()
            mhc_pseudosequences_dict[normalize_mhc_name(mhc)] = seq
    return mhc_pseudosequences_dict

def load_iedb_binding_data(
        filename="class2_data.csv", require_units=False, restrict_allele=None):
    df = pd.read_csv(filename)
    print("Loaded %d samples" % len(df))
    print("-- Tail:\n%s" % (df.tail(),))
    bad_mhc = df.mhc.str.len() < 7
    print("Dropping %d rows with short allele names" % bad_mhc.sum())
    df = df[~bad_mhc]
    df.mhc = df.mhc.map(normalize_mhc_name)
    if restrict_allele:
        df = df[df.mhc == restrict_allele]
        print("Keeping %d rows for allele %s" % (len(df), restrict_allele))
    if require_units:
        no_units = df.units.isnull()
        print("Dropping %d rows without units" % no_units.sum())
        df = df[~no_units]
    print("-- Tail after filtering and transform:\n%s" % (df.tail(),))
    return df


def load_mass_spec_hits_excel(filename=None, restrict_allele=None):
    if not filename:
        filename = os.environ["CLASS_II_DATA"]
    df = pd.read_excel(filename)
    hits = {}
    for allele in df.columns:
        column = df[allele]
        print(allele, normalize_mhc_name(allele))
        allele = normalize_mhc_name(allele)
        if restrict_allele and allele != restrict_allele:
            print("-- Skipping %s" % allele)
        hits[allele] = [
            s.upper() for s in column
            if isinstance(s, str) and len(s) > 0 and "X" not in s]
        print("Loaded %d hits for %s (max length %d)" % (
            len(hits[allele]),
            allele,
            max(len(p) for p in hits[allele])))
    return hits

def load_mass_spec_hits(directory=None, pattern="_uIP_"):
    if directory is None:
        directory = os.path.split(os.environ["CLASS_II_DATA"])[0]
    allele_to_path = {}
    for filename in os.listdir(directory):
        if pattern in filename:
            parts = filename.split(pattern)
            if len(parts) == 1:
                rest_of_filename = parts[0]
            else:
                rest_of_filename = parts[1]
            allele = rest_of_filename.split("_")[0]
            allele = normalize_mhc_name(allele)
            allele_to_path[allele] = os.path.join(directory, filename)
    allele_to_hits = {}
    for allele, path in allele_to_path.items():
        with open(path, "r") as f:
            original_hits = set([])
            normalized_hits = set([])
            for l in f:
                if not l:
                    continue
                if l.startswith("#") or l.startswith("sequence"):
                    continue
                parts = l.split()
                peptide = parts[0]
                if peptide in original_hits:
                    raise ValueError("Duplicate hit '%s' for allele %s" % (
                        peptide,
                        allele))
                original_hits.add(peptide)
                normalized_peptide = peptide.upper()
                normalized_hits.add(normalized_peptide)
            print("Loaded %d hits for %s (max length %d)" % (
                len(normalized_hits),
                allele,
                max(len(p) for p in normalized_hits)))
            allele_to_hits[allele] = normalized_hits
    return allele_to_hits

full_proteome_sequence = None

def generate_negatives_from_proteome(
        peptides,
        factor=1,
        match_length_distribution=False,
        min_length=None,
        max_length=None):
    negative_peptides = []
    original_set = set(peptides)
    global full_proteome_sequence
    if full_proteome_sequence is None:
        # make a single string out of all the proteins
        full_proteome_sequence = "".join(
            ensembl_grch38.protein_sequences.fasta_dictionary.values())
    n_desired = int(factor * len(peptides))
    unique_lengths = sorted({len(p) for p in peptides})
    if match_length_distribution:
        length_counts = Counter()
        for p in peptides:
            length_counts[len(p)] += 1
        unique_length_counts = [
            length_counts[l] for l in unique_lengths
        ]
        unique_length_probs = (
            unique_length_counts.astype("float32") / unique_length_counts.sum())
        random_lengths = np.random.choice(
            unique_lengths,
            size=2 * n_desired,
            replace=True,
            p=unique_length_probs)
    else:
        if min_length is None:
            min_length = min(unique_lengths)
        if max_length is None:
            max_length = max(unique_lengths)
        length_range = np.arange(min_length, max_length + 1)
        random_lengths = np.random.choice(
            length_range,
            size=2 * n_desired,
            replace=True)
    random_positions = np.random.randint(
        0, len(full_proteome_sequence) - 1, 2 * n_desired)

    for l, pos in zip(random_lengths, random_positions):
        if len(negative_peptides) >= n_desired:
            break
        s = full_proteome_sequence[pos:pos + l]
        if len(s) != l:
            continue
        if s in original_set:
            continue
        if "X" in s or "*" in s or "U" in s:
            continue
        negative_peptides.append(s)
    if len(negative_peptides) < n_desired:
        raise ValueError("Unable to make sufficient number of decoys (%d < %d)" % (
            len(negative_peptides), n_desired))
    return negative_peptides

def augment_with_decoys(
        hit_dataset,
        decoy_multiple,
        min_decoy_length=None,
        max_decoy_length=None):
    n_hits = len(hit_dataset)
    decoy_peptides = generate_negatives_from_proteome(
        hit_dataset.peptides,
        factor=decoy_multiple,
        min_length=min_decoy_length,
        max_length=max_decoy_length)

    n_decoys = len(decoy_peptides)
    assert n_decoys == int(n_hits * decoy_multiple)
    decoy_mhc_alleles = list(np.random.choice(hit_dataset.alleles, size=n_decoys))

    sum_hit_weights = hit_dataset.weights.sum()
    hits_to_decoys = sum_hit_weights / float(n_decoys)
    decoy_weight = min(1.0, hits_to_decoys)
    decoy_dataset = Dataset(
        peptides=decoy_peptides,
        alleles=decoy_mhc_alleles,
        labels=0,
        weights=decoy_weight)
    return hit_dataset.combine(decoy_dataset, preserve_group_ids=False)

def load_mass_spec(
        mhc_pseudosequences_dict, restrict_allele=None, decoy_factor=10):

    # Load mass spec hits
    hits = load_mass_spec_hits(restrict_allele=restrict_allele)
    hit_peptides = []
    hit_mhc_seqs = []
    for (allele, peptides) in hits.items():
        hit_peptides.extend(peptides)
        allele_seq = mhc_pseudosequences_dict[allele]
        hit_mhc_seqs.extend([allele_seq] * len(peptides))

    decoy_peptides = generate_negatives_from_proteome(
        hit_peptides, factor=decoy_factor)
    print("Generated %d decoys" % len(decoy_peptides))
    decoy_mhc_sequences = list(np.random.choice(
        hit_mhc_seqs,
        size=len(decoy_peptides)))
    print("Generated %d decoy MHC sequences" % len(decoy_mhc_sequences))
    return hit_peptides, hit_mhc_seqs, decoy_peptides, decoy_mhc_sequences

def load_mass_spec_hits_and_group_nested_sets(
        hits_directory=None):
    hits_dict = load_mass_spec_hits(directory=hits_directory)
    hit_peptides = []
    hit_mhc_alleles = []
    hit_weights = []
    hit_group_ids = []
    for (allele, peptides) in hits_dict.items():
        shuffled_peptides, group_ids, weights = group_similar_sequences(peptides)
        print("-- Distinct loci for %s: %d/%d" % (
            allele,
            len(set(group_ids)),
            len(group_ids)))
        assert len(shuffled_peptides) == len(peptides), \
            "Exepcted %d peptides but got back %d" % (
                len(peptides),
                len(shuffled_peptides))
        hit_peptides.extend(shuffled_peptides)
        hit_mhc_alleles.extend([allele] * len(peptides))
        hit_weights.extend(weights)
        # make group IDs unique for each allele
        hit_group_ids.extend((max(hit_group_ids) + 1 if len(hit_group_ids) > 0 else 0) + group_ids)
    assert set(hit_mhc_alleles) == set(hits_dict.keys())
    hit_weights = np.array(hit_weights)
    return Dataset(
        alleles=hit_mhc_alleles,
        peptides=hit_peptides,
        weights=hit_weights,
        group_ids=hit_group_ids)

def load_mass_spec_hits_and_decoys_grouped_by_nested_sets(
        decoy_multiple,
        hits_directory=None,
        min_decoy_length=None,
        max_decoy_length=None):
    hit_dataset = \
        load_mass_spec_hits_and_group_nested_sets(hits_directory=hits_directory)
    combined_dataset = augment_with_decoys(
            hit_dataset=hit_dataset,
            decoy_multiple=decoy_multiple,
            min_decoy_length=min_decoy_length,
            max_decoy_length=max_decoy_length)
    return combined_dataset.shuffle()
