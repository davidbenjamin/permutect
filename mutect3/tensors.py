import torch
import random
import numpy as np
import pickle
from typing import Set, List

class SiteInfo:
    def __init__(self, chromosome, position, ref, alt, popaf):
        self._locus = chromosome + ":" + str(position)
        self._ref = ref
        self._alt = alt
        self._popaf = popaf

    def locus(self):
        return self._locus

    def ref(self):
        return self._ref

    def alt(self):
        return self._alt

    def popaf(self):
        return self._popaf


class MutectInfo:
    def __init__(self, tlod: float, tumor_dp: int, filters: Set[str]):
        self._tlod = tlod
        self._tumor_dp = tumor_dp
        self._filters = filters

    def tlod(self) -> float:
        return self._tlod

    def tumor_depth(self) -> int:
        return self._tumor_dp

    def filters(self) -> Set[str]:
        return self._filters


class VariantInfo:
    # hec is abbreviation for haplotype equivalence counts
    def __init__(self, hec, haplotype_complexity, haplotype_dominance, ref_bases):
        # take integer haplotype equivalence counts (already in order from greatest to least from Mutect)
        # and calculate the fractional share of the 2nd and 3rd, or 0 if none exist
        total = 0
        for n in hec:
            total += n

        self._info = [0.0 if len(hec) < 2 else hec[1] / total, 0.0 if len(hec) < 3 else hec[2] / total]

        # now append the haplotype complexity and haplotype dominance
        self._info.append(haplotype_complexity)
        self._info.append(haplotype_dominance)
        self._info.extend([self.num_repeats(ref_bases, k) for k in range(1, 6)])
        self._info = torch.FloatTensor(self._info)

    def info_tensor(self) -> torch.Tensor:
        return self._info

    # count how many repeats of length k surround the middle base
    # example: num_repeats(GACTACTACTG,3) = 3
    @staticmethod
    def num_repeats(ref_bases, k):
        N = len(ref_bases)
        n = int((N - 1) / 2)  # note k > n will cause errors

        # extend a repeat forward, to front and then to back (both exclusive)
        # note that this only extends backward if they match the bases going forward
        # that is AAAAAGTGTCC (first G in the middle) will get extended forward through
        # both GTs but won't be extended back
        front = n + k
        while front < N and ref_bases[front] == ref_bases[front - k]:
            front = front + 1
        back = n - 1
        while back >= 0 and ref_bases[back] == ref_bases[back + k]:
            back = back - 1
        forward_repeats = (front - back - 1) / k

        # same idea but extending backwards first (now back is exclusive and front is inclusive)
        back = n - k
        while back >= 0 and ref_bases[back] == ref_bases[back + k]:
            back = back - 1
        front = n + 1
        while front < N and ref_bases[front] == ref_bases[front - k]:
            front = front + 1
        backward_repeats = (front - back - 1) / k

        return max(forward_repeats, backward_repeats)


class Datum:
    def __init__(self, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor, info_tensor: torch.Tensor, site_info: SiteInfo, mutect2_data: MutectInfo, artifact_label):
        self._site_info = site_info
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._mutect2_data = mutect2_data
        self._artifact_label = artifact_label

    def ref_tensor(self) -> torch.Tensor:
        return self._ref_tensor

    def alt_tensor(self) -> torch.Tensor:
        return self._alt_tensor

    def info_tensor(self) -> torch.Tensor:
        return self._info_tensor

    def site_info(self) -> SiteInfo:
        return self._site_info

    def mutect_info(self) -> MutectInfo:
        return self._mutect2_data

    def artifact_label(self):
        return self._artifact_label

# pickle and unpickle a Python list of Datum objects.  Convenient to have here because unpickling needs to have all
# the constituent classes of Datum explicitly imported.
def make_pickle(file, datum_list):
    with open(file, 'wb') as f:
        pickle.dump(datum_list, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def downsample(tensor: torch.Tensor, downsample) -> torch.Tensor:
    if downsample is None or downsample >= len(tensor):
        return tensor
    else:
        return tensor[torch.randperm(len(tensor))[:downsample]]


NUM_READ_FEATURES = 11  #size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9   # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
                        # and 5 for ref bases STR info

ARTIFACT_POPAF_THRESHOLD = 5.9  # only let things absent from gnomAD be artifacts out of caution
GERMLINE_POPAF_THRESHOLD = 1  # also very cautious.  There are so many germline variants we can be wasteful!

REF_DOWNSAMPLE = 20  # choose this many ref reads randomly
MIN_REF = 5

TLOD_THRESHOLD = 6  # we are classified artifacts other than sequencing errors described by base qualities
NON_ARTIFACT_PER_ARTIFACT = 20  # ratio of non-artifact to artifact in unsupervised training data


class TableReader:
    # separator between different alleles in the FRS (featurized read set) INFO field from Mutect output
    ALLELE_SEPARATOR = '|'

    def __init__(self, header_tokens, tumor_sample, normal_sample=None):
        self.site_info_indices = self._get_indices(header_tokens, ["CHROM", "POS", "REF", "ALT", "POPAF"])
        self.mutect_indices = self._get_indices(header_tokens, ["FILTER", "TLOD", tumor_sample + ".DP"])
        self.variant_info_indices = self._get_indices(header_tokens, ["HEC", "HAPCOMP", "HAPDOM", "REF_BASES"])
        self.status_idx = TableReader._index_if_exists(header_tokens, "STATUS")
        self.tumor_idx = TableReader._index_if_exists(header_tokens, tumor_sample + ".FRS")

        # optional normal data
        if normal_sample is not None:
            self.normal_idx, self.normal_dp_idx = self._get_indices(header_tokens,[normal_sample + ".FRS", normal_sample + ".DP"])

    def variant_info(self, tokens) -> VariantInfo:
        hec, hapcomp, hapdom, ref_bases = (tokens[idx] for idx in self.variant_info_indices)
        return VariantInfo([int(n) for n in hec.split(",")], int(hapcomp), float(hapdom), ref_bases)

    def site_info(self, tokens) -> SiteInfo:
        chrom, pos, ref, alt, popaf = (tokens[idx] for idx in self.site_info_indices)
        return SiteInfo(chrom, int(pos), ref, alt, float(popaf))

    def mutect_info(self, tokens) -> MutectInfo:
        filters, tlod, tumor_dp = (tokens[idx] for idx in self.mutect_indices)
        return MutectInfo(float(tlod), int(tumor_dp), set(filters.split(",")))

    def status(self, tokens):
        return tokens[self.status_idx]

    def tumor_ref_and_alt(self, tokens, ref_downsample=None):
        return TableReader._get_read_tensors(tokens[self.tumor_idx], ref_downsample)

    def normal_dp(self, tokens):
        return int(tokens[self.normal_dp_idx])

    def normal_ref_and_alt(self, tokens, ref_downsample=None):
        return TableReader._get_read_tensors(tokens[self.normal_idx], ref_downsample)

    @staticmethod
    def _index_if_exists(lis, element):
        return lis.index(element) if element in lis else None

    @staticmethod
    def _get_indices(lis, elements):
        return tuple(TableReader._index_if_exists(lis, element) for element in elements)

    # get the ref and alt tensors from the VariantsToTable raw output, convert to torch tensors, downsample ref if needed
    @staticmethod
    def _get_read_tensors(token, ref_downsample=None):
        tokens = token.split(TableReader.ALLELE_SEPARATOR)
        ref, alt = tuple(np.fromstring(x, dtype=int, sep=',').reshape((-1, NUM_READ_FEATURES)) for x in tokens)
        ref, alt = torch.from_numpy(ref).float(), torch.from_numpy(alt).float()
        return downsample(ref, ref_downsample), alt


# this takes a table from VariantsToTable and produces a Python list of Datum objects
def make_tensors(raw_file, is_training, sample_name, normal_sample_name=None, shuffle=True) -> List[Datum]:
    data = []

    # simple online method for balanced data set where for each k-alt-read artifact there are
    # NON_ARTIFACT_PER_ARTIFACT (downsampled) k-alt-read non-artifacts.  That is, alt count is not an informative
    # feature.
    unmatched_snv_counts = []
    unmatched_deletion_counts = []
    unmatched_insertion_counts = []

    with open(raw_file) as fp:
        reader = TableReader(fp.readline().split(), sample_name, normal_sample_name)

        for n, line in enumerate(fp):
            if n % 100000 == 0:
                print("Processing line " + str(n))

            tokens = line.split()
            site_info = reader.site_info(tokens)
            m2_data = reader.mutect_info(tokens)

            # Contamination and weak evidence / low log odds have low AFs but are not artifacts.  We exclude them
            # from training.  As for testing, M3 will continue to rely on them
            if "contamination" in m2_data.filters() or "weak_evidence" in m2_data.filters() or m2_data.tlod() < TLOD_THRESHOLD:
                continue

            # for testing, M3 relies on the existing germline filter.  For training, we keep germline variants
            # and downsample to simulate true non-artifacts with varying allele fraction
            if "germline" in m2_data.filters() and not is_training:
                continue

            # in training, we want low popaf for true artifacts, high AF for true variants
            # in order to have more confident weak labels.  In between, discard.
            if is_training and (GERMLINE_POPAF_THRESHOLD < site_info.popaf() < ARTIFACT_POPAF_THRESHOLD):
                continue

            ref_tensor, alt_tensor = reader.tumor_ref_and_alt(tokens, REF_DOWNSAMPLE)
            alt_count = len(alt_tensor)
            if alt_count == 0 or len(ref_tensor) < MIN_REF:
                continue

            af = alt_count / m2_data.tumor_depth()
            is_artifact = False
            if is_training:
                normal_af = 0
                has_normal = normal_sample_name is not None
                if has_normal:
                    normal_dp = reader.normal_dp(tokens)
                    normal_ref, normal_alt = reader.normal_ref_and_alt(tokens, REF_DOWNSAMPLE)
                    normal_af = 0 if normal_dp == 0 else len(normal_alt) / normal_dp
                    # TODO: output normal alt reads in dataset

                variant_size = len(site_info.alt()) - len(site_info.ref())
                unmatched_artifact_counts = unmatched_snv_counts if variant_size == 0 else \
                    (unmatched_insertion_counts if variant_size > 0 else unmatched_deletion_counts)

                # low AF in tumor and normal, rare in population implies artifact
                if (not has_normal or normal_af < 0.2) and af < 0.2 and site_info.popaf() > ARTIFACT_POPAF_THRESHOLD:
                    unmatched_artifact_counts.extend([alt_count] * NON_ARTIFACT_PER_ARTIFACT)
                    is_artifact = True
                # high AF in tumor and normal, common in population implies germline, which we downsample
                elif (not has_normal or normal_af > 0.35) and af > 0.35 and site_info.popaf() < GERMLINE_POPAF_THRESHOLD and unmatched_artifact_counts:
                    downsample_count = min(alt_count, unmatched_artifact_counts.pop())
                    alt_tensor = alt_tensor[torch.randperm(alt_count)[:downsample_count]]
                # inconclusive -- don't use for training data
                else:
                    continue
            else:
                # use Concordance STATUS field for test labels
                status = reader.status(tokens)
                is_artifact = (status == "FP" or status == "FTN")

            # assembly complexity site-level annotations
            info_tensor = reader.variant_info(tokens).info_tensor()

            data.append(Datum(ref_tensor, alt_tensor, info_tensor, site_info, m2_data, 1 if is_artifact else 0))
        if shuffle:
            random.shuffle(data)
        return data
