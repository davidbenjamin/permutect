import torch
import random
import numpy as np
import pickle
from torch.distributions.beta import Beta
from typing import Set, List
from tqdm.autonotebook import tqdm, trange
from collections import defaultdict
from mutect3 import utils

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

    def variant_type(self) -> utils.VariantType:
        diff = len(self.alt()) - len(self.ref())
        return utils.VariantType.SNV if diff == 0 else (utils.VariantType.INSERTION if diff > 0 else utils.VariantType.DELETION)

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
    def __init__(self, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor, info_tensor: torch.Tensor, site_info: SiteInfo,
                 mutect2_data: MutectInfo, artifact_label, normal_depth: int, normal_alt_count: int):
        self._site_info = site_info
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._mutect2_data = mutect2_data
        self._artifact_label = artifact_label
        self._normal_depth = normal_depth
        self._normal_alt_count = normal_alt_count

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

    def normal_depth(self) -> int:
        return self._normal_depth

    def normal_alt_count(self) -> int:
        return self._normal_alt_count



    # beta is distribution of downsampling fractions
    def downsampled_copy(self, beta: Beta):
        ref_frac = beta.sample().item()
        alt_frac = beta.sample().item()

        ref_length = max(1, round(ref_frac * len(self._ref_tensor)))
        alt_length = max(1, round(alt_frac * len(self._alt_tensor)))
        ref = downsample(self._ref_tensor, ref_length)
        alt = downsample(self._alt_tensor, alt_length)
        return Datum(ref, alt, self.info_tensor(), self.site_info(), self.mutect_info(), self.artifact_label(), self.normal_depth(), self.normal_alt_count())

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

RARE_POPAF = 5.9  # only let things absent from gnomAD be artifacts out of caution
COMMON_POPAF = 1  # also very cautious.  There are so many germline variants we can be wasteful!

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
    trusted_m2_filters = {"contamination", "germline", "weak_evidence"}
    data = []

    # simple method to balance data: for each k-alt-read artifact there are
    # NON_ARTIFACT_PER_ARTIFACT (downsampled) k-alt-read non-artifacts.
    unmatched_counts_by_type = defaultdict(list)

    with open(raw_file) as fp:
        reader = TableReader(fp.readline().split(), sample_name, normal_sample_name)
        pbar = tqdm(enumerate(fp))
        for n, line in pbar:

            tokens = line.split()
            site_info = reader.site_info(tokens)
            m2_data = reader.mutect_info(tokens)
            filters = m2_data.filters()
            popaf = site_info.popaf()

            # For testing we still rely on the M2 contamination, germline, and evidence filters.
            # For training, contamination has low-AF but is not an artifact, which we don't want
            if not is_training and filters.intersection(trusted_m2_filters):
                continue
            elif "contamination" in filters:
                continue

            ref_tensor, alt_tensor = reader.tumor_ref_and_alt(tokens, REF_DOWNSAMPLE)
            alt_count = len(alt_tensor)
            if alt_count == 0 or len(ref_tensor) < MIN_REF:
                continue

            tumor_af = alt_count / m2_data.tumor_depth()
            is_artifact = None

            has_normal = normal_sample_name is not None
            normal_dp = reader.normal_dp(tokens) if has_normal else 0
            normal_ref, normal_alt = reader.normal_ref_and_alt(tokens, REF_DOWNSAMPLE) if has_normal else (None, None)
            normal_alt_count = len(normal_alt) if has_normal else 0
            normal_af = 0 if normal_dp == 0 else normal_alt_count / normal_dp

            likely_seq_error = "weak_evidence" in filters or m2_data.tlod() < TLOD_THRESHOLD
            likely_germline = has_normal and normal_af > 0.2

            # extremely strict criteria because there are so many germline variants we can afford to waste a lot
            definite_germline = not likely_seq_error and popaf < COMMON_POPAF and (tumor_af > 0.35 and popaf < COMMON_POPAF) and (normal_af > 0.35 if has_normal else True)

            if is_training:
                unmatched_artifact_counts = unmatched_counts_by_type[utils.get_variant_type(site_info.alt(), site_info.ref())]

                # low AF in tumor and normal, rare in population implies artifact
                if not (likely_seq_error or likely_germline) and tumor_af < 0.2 and popaf > RARE_POPAF:
                    unmatched_artifact_counts.extend([alt_count] * NON_ARTIFACT_PER_ARTIFACT)
                    is_artifact = 1
                # high AF in tumor and normal, common in population implies germline, which we downsample
                elif definite_germline and unmatched_artifact_counts:
                    downsample_count = min(alt_count, unmatched_artifact_counts.pop())
                    alt_tensor = alt_tensor[torch.randperm(alt_count)[:downsample_count]]
                    is_artifact = 0
                # inconclusive -- unlabeled datum
                # to avoid too large a dataset, we try to bias toward possible artifacts and keep out obvious sequencing errors
                elif m2_data.tlod() > 4.0 and tumor_af < 0.3:
                    is_artifact = None
                    downsample_count = min(alt_count, 10)
                    alt_tensor = alt_tensor[torch.randperm(alt_count)[:downsample_count]]
                else:
                    continue
            else:
                # use Concordance STATUS field for test labels
                status = reader.status(tokens)
                is_artifact = (status == "FP" or status == "FTN")
                print(normal_dp, normal_alt_count)

            # assembly complexity site-level annotations
            info_tensor = reader.variant_info(tokens).info_tensor()

            data.append(Datum(ref_tensor, alt_tensor, info_tensor, site_info, m2_data, is_artifact, normal_dp, normal_alt_count))
        if shuffle:
            random.shuffle(data)
        return data

def generate_pickles(tumor_table, normal_table, tumor_sample, normal_sample, pickle_dir, pickle_prefix):

    pair_train_pickle, small_pair_train_pickle, tumor_train_pickle, normal_train_pickle, test_pickle, small_test_pickle = \
        (pickle_dir + pickle_prefix + suffix for suffix in ('-pair-train.pickle', '-small-pair-train.pickle', '-tumor-train.pickle', \
                                                            '-normal-train.pickle', '-test.pickle', '-small-test.pickle'))

    # we form a few kinds of training data: tumor data using the normal
    # (the normal doesn't change the format but helps make better truth guesses)
    print("Generating and pickling tumor tensors for training using tumor and normal")
    pair_train_data = make_tensors(tumor_table, True, tumor_sample, normal_sample)
    make_pickle(pair_train_pickle, pair_train_data)

    print("Generating and pickling small (by 10x) tumor tensors for training using tumor and normal")
    make_pickle(small_pair_train_pickle, pair_train_data[:int(len(pair_train_data)/10)])

    print("Generating and pickling tumor tensors for training using only tumor")
    tumor_train_data = make_tensors(tumor_table, True, tumor_sample)
    make_pickle(tumor_train_pickle, tumor_train_data)

    print("Generating and pickling normal tensors for training using only normal")
    normal_train_data = make_tensors(normal_table, True, normal_sample)
    make_pickle(normal_train_pickle, normal_train_data)

    print("Generating and pickling tumor tensors for testing using STATUS labels")
    test_data = make_tensors(tumor_table, False, tumor_sample, normal_sample)
    make_pickle(test_pickle, test_data)

    print("Generating and pickling small (by 10x) tumor tensors for for testing using STATUS labels")
    make_pickle(small_test_pickle, test_data[:int(len(test_data)/10)])
