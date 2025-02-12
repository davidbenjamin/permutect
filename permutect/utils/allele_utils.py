from __future__ import annotations

import cyvcf2
import numpy as np

from permutect.utils.math_utils import find_factors
from permutect.utils.enums import Variation


def trim_alleles_on_right(ref: str, alt: str):
    # if alt and ref alleles are not in minimal representation ie have redundant matching bases at the end, trim them
    trimmed_ref, trimmed_alt = ref, alt
    while len(trimmed_ref) > 1 and len(trimmed_alt) > 1 and trimmed_alt[-1] == trimmed_ref[-1]:
        trimmed_ref, trimmed_alt = trimmed_ref[:-1], trimmed_alt[:-1]
    return trimmed_ref, trimmed_alt


def find_variant_type(v: cyvcf2.Variant):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    ref = v.REF
    return Variation.get_type(ref, alt)


def get_variant_type(alt_allele, ref_allele):
    variant_size = len(alt_allele) - len(ref_allele)
    if variant_size == 0:
        return Variation.SNV
    else:
        return Variation.INSERTION if variant_size > 0 else Variation.DELETION


def count_leading_repeats(sequence: str, unit: str):
    # count how many times a unit string is repeated at the beginning of a larger string
    # eg 'ATATGGG', 'AT' -> 1; 'AGGGGG', 'G' -> 0; 'TTATTATTAGTTA', 'TTA' -> 3
    result = 0
    idx = 0
    unit_length = len(unit)
    while (idx + unit_length - 1 < len(sequence)) and sequence[idx:idx + unit_length] == unit:
        result += 1
        idx += unit_length
    return result


def count_trailing_repeats(sequence: str, unit: str):
    # same, but at the end of a sequence
    # eg 'ATATGGG', 'G' -> 3; 'AGGGGG', 'G' -> 5; 'TTATTATTAGTTA', 'TTA' -> 1
    result = 0
    unit_length = len(unit)
    idx = len(sequence) - unit_length   # index at beginning of comparison eg 'GGATC', 'TC' starts at index 5 - 2 = 3, the 'T'
    while idx >= 0 and sequence[idx:idx + unit_length] == unit:
        result += 1
        idx -= unit_length
    return result


MAX_NUM_BASES_FOR_ENCODING = 13


def truncate_bases_if_necessary(bases: str):
    return bases if len(bases) <= MAX_NUM_BASES_FOR_ENCODING else bases[:MAX_NUM_BASES_FOR_ENCODING]


def bases_as_base5_int(bases: str) -> int:
    # here we just butcher variants longer than 13 bases and chop!!!
    power_of_5 = 1
    bases_to_use = truncate_bases_if_necessary(bases)
    result = 0
    for nuc in bases_to_use:
        coeff = 1 if nuc == 'A' else (2 if nuc == 'C' else (3 if nuc == 'G' else 4))
        result += power_of_5 * coeff
        power_of_5 *= 5
    return result


def bases5_as_base_string(base5: int) -> str:
    result = ""
    remaining = base5
    while remaining > 0:
        digit = remaining % 5
        nuc = 'A' if digit == 1 else ('C' if digit == 2 else ('G' if digit == 3 else 'T'))
        result += nuc
        remaining = (remaining - digit) // 5
    return result


def is_repeat(bases: str, unit: str):
    # eg ACGACGACG, ACG -> True; TTATTA, TA -> False
    unit_length = len(unit)
    if len(bases) % unit_length == 0:
        num_repeats = len(bases) // len(unit)
        for repeat_idx in range(num_repeats):
            start = repeat_idx * unit_length
            if bases[start: start + unit_length] != unit:
                return False
        return True
    else:
        return False


def decompose_str_unit(indel_bases: str):
    # decompose an indel into its most basic repeated unit
    # examples: "ATATAT" -> ("AT", 3); "AAAAA" -> ("A", 5); "TTGTTG" -> ("TTG", 2); "ATGTG" -> "ATGTG", 1
    for unit_length in find_factors(len(indel_bases)):  # note: these are sorted ascending
        unit = indel_bases[:unit_length]
        if is_repeat(indel_bases, unit):
            return unit, (len(indel_bases) // unit_length)
    return indel_bases, 1


def get_str_info_array(ref_sequence_string: str, ref_allele: str, alt_allele: str):
    assert len(ref_sequence_string) % 2 == 1, "must be odd length to have well-defined middle"
    middle_idx = (len(ref_sequence_string) - 1) // 2

    ref, alt = ref_allele, alt_allele

    insertion_length = max(len(alt) - len(ref), 0)
    deletion_length = max(len(ref) - len(alt), 0)

    if len(ref) == len(alt):
        unit, num_units = alt, 1
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(ref):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx], unit)
    elif insertion_length > 0:
        unit, num_units = decompose_str_unit(alt[1:])  # the inserted sequence is everything after the anchor base that matches ref
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(ref):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx + 1], unit)   # +1 accounts for the anchor base
    else:
        unit, num_units = decompose_str_unit(ref[1:])  # the deleted sequence is everything after the anchor base
        # it's pretty arbitrary whether we include the deleted bases themselves as 'after' or not
        repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + len(alt):], unit)
        repeats_before = count_trailing_repeats(ref_sequence_string[:middle_idx + 1], unit)   # likewise, account for the anchor base
    # note that if indels are left-aligned (as they should be from the GATK) repeats_before really ought to be zero!!
    return np.array([insertion_length, deletion_length, len(unit), num_units, repeats_before, repeats_after])


def make_1d_sequence_tensor(sequence_string: str) -> np.ndarray:
    """
    convert string of form ACCGTA into tensor [ 0, 1, 1, 2, 3, 0]
    """
    result = np.zeros(len(sequence_string), dtype=np.uint8)
    for n, char in enumerate(sequence_string):
        integer = 0 if char == 'A' else (1 if char == 'C' else (2 if char == 'G' else 3))
        result[n] = integer
    return result


# returns two length-L 1D arrays of ref stacked on top of alt, with '4' in alt(ref) for deletions(insertions)
def get_ref_and_alt_sequences(ref_seq_1d, ref_allele: str, alt_allele: str):
    """
    :param ref_seq_1d: 1D numpy integer array in form eg ATTTCGG -> [0,3,3,3,1,2,2]
    :return:
    """
    assert len(ref_seq_1d) % 2 == 1, "ref sequence length should be odd"
    middle_idx = (len(ref_seq_1d) - 1) // 2
    max_allele_length = middle_idx  # just kind of a coincidence
    ref, alt = ref_allele[:max_allele_length], alt_allele[:max_allele_length]

    if len(ref) >= len(alt):    # substitution or deletion
        ref_array = ref_seq_1d
        alt_array = np.copy(ref_array)
        deletion_length = len(ref) - len(alt)
        # add the deletion value '4' to make the alt allele array as long as the ref allele
        alt_allele_array = make_1d_sequence_tensor(alt) if deletion_length == 0 else np.hstack((make_1d_sequence_tensor(alt), np.full(shape=deletion_length, fill_value=4)))
        alt_array[middle_idx: middle_idx + len(alt_allele_array)] = alt_allele_array
    else:   # insertion
        insertion_length = len(alt) - len(ref)
        before = ref_seq_1d[:middle_idx]
        after = ref_seq_1d[middle_idx + len(ref):-insertion_length]

        alt_allele_array = make_1d_sequence_tensor(alt)
        ref_allele_array = np.hstack((make_1d_sequence_tensor(ref), np.full(shape=insertion_length, fill_value=4)))

        ref_array = np.hstack((before, ref_allele_array, after))
        alt_array = np.hstack((before, alt_allele_array, after))

    assert len(ref_array) == len(alt_array)
    if len(ref) == len(alt): # SNV -- ref and alt ought to be different
        assert alt_array[middle_idx] != ref_array[middle_idx]
    else:   # indel -- ref and alt are the same at the anchor base, then are different
        assert alt_array[middle_idx + 1] != ref_array[middle_idx + 1]
    return ref_array[:len(ref_seq_1d)], alt_array[:len(ref_seq_1d)] # this clipping may be redundant
