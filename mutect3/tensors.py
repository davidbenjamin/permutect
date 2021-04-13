import torch
import random
import numpy as np
import pickle

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
    def __init__(self, tlod, tumor_dp, filters):
        self._tlod = tlod
        self._tumor_dp = tumor_dp
        self._filters = filters

    def tlod(self):
        return self._tlod

    def tumor_depth(self):
        return self._tumor_dp

    def filters(self):
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

    def info_tensor(self):
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
    def __init__(self, ref_tensor, alt_tensor, info_tensor, metadata, mutect2_data, artifact_label):
        self._ref_tensor = ref_tensor
        self._alt_tensor = alt_tensor
        self._info_tensor = info_tensor
        self._metadata = metadata
        self._mutect2_data = mutect2_data
        self._artifact_label = artifact_label

    def ref_tensor(self):
        return self._ref_tensor

    def alt_tensor(self):
        return self._alt_tensor

    def info_tensor(self):
        return self._info_tensor

    def metadata(self):
        return self._metadata

    def mutect_info(self):
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


NUM_READ_FEATURES = 11  #size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9   # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
                        # and 5 for ref bases STR info

ARTIFACT_POPAF_THRESHOLD = 5.9  # only let things absent from gnomAD be artifacts out of caution
GERMLINE_POPAF_THRESHOLD = 1  # also very cautious.  There are so many germline variants we can be wasteful!

REF_DOWNSAMPLE = 10  # choose this many ref reads randomly
MIN_REF = 5

TLOD_THRESHOLD = 6  # we are classified artifacts other than sequencing errors described by base qualities
NON_ARTIFACT_PER_ARTIFACT = 50  # ratio of non-artifact to artifact in unsupervised training data


class TableReader:
    # separator between different alleles in the FRS (featurized read set) INFO field from Mutect output
    ALLELE_SEPARATOR = '|'

    def __init__(self, header_tokens, tumor_sample, normal_sample=None):
        # site metadata
        self.chrom_idx, self.pos_idx, self.ref_allele_idx, self.alt_allele_idx, self.popaf_idx = \
            TableReader._get_indices(header_tokens, ["CHROM", "POS", "REF", "ALT", "POPAF"])

        # Mutect2 data
        self.filter_idx, self.tlod_idx, self.tumor_dp_idx, = \
            TableReader._get_indices(header_tokens, ["FILTER", "TLOD", tumor_sample + ".DP"])

        # variant info features
        self.hec_idx, self.hapcomp_idx, self.hapdom_idx, self.ref_bases_idx = \
            self._get_indices(header_tokens, ["HEC", "HAPCOMP", "HAPDOM", "REF_BASES"])

        self.status_idx = TableReader._index_if_exists(header_tokens, "STATUS")
        self.tumor_idx = TableReader._index_if_exists(header_tokens, tumor_sample + ".FRS")

        # optional normal data
        if normal_sample is not None:
            self.normal_idx, self.normal_dp_idx = TableReader._get_indices(header_tokens,
                                                                           [normal_sample + ".FRS",
                                                                            normal_sample + ".DP"])

    def variant_info(self, tokens):
        haplotype_equivalence_counts = [int(n) for n in tokens[self.hec_idx].split(",")]
        haplotype_complexity = int(tokens[self.hapcomp_idx])
        haplotype_dominance = float(tokens[self.hapdom_idx])
        ref_bases = tokens[self.ref_bases_idx]
        return VariantInfo(haplotype_equivalence_counts, haplotype_complexity, haplotype_dominance, ref_bases)

    def site_info(self, tokens):
        chromosome = tokens[self.chrom_idx]
        position = int(tokens[self.pos_idx])
        ref = tokens[self.ref_allele_idx]
        alt = tokens[self.alt_allele_idx]
        popaf = float(tokens[self.popaf_idx])
        return SiteInfo(chromosome, position, ref, alt, popaf)

    def mutect_info(self, tokens):
        tlod = float(tokens[self.tlod_idx])
        tumor_dp = int(tokens[self.tumor_dp_idx])
        filters = set(tokens[self.filter_idx].split(","))
        return MutectInfo(tlod, tumor_dp, filters)

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
        if ref_downsample is not None and len(ref) > ref_downsample:
            ref = ref[torch.randperm(len(ref))[:ref_downsample]]
        return ref, alt


# this takes a table from VariantsToTable and produces a Python list of Datum objects
def make_tensors(raw_file, is_training, sample_name, normal_sample_name=None, shuffle=True):
    data = []

    # simple online method for balanced data set where for each k-alt-read artifact there are
    # NON_ARTIFACT_PER_ARTIFACT (downsampled) k-alt-read non-artifacts.  That is, alt count is not an informative
    # feature.
    unmatched_artifact_counts = []

    with open(raw_file) as fp:
        reader = TableReader(fp.readline().split(), sample_name, normal_sample_name)

        for n, line in enumerate(fp):
            if n % 100000 == 0:
                print("Processing line " + str(n))

            tokens = line.split()
            metadata = reader.site_info(tokens)
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
            if is_training and (GERMLINE_POPAF_THRESHOLD < metadata.popaf() < ARTIFACT_POPAF_THRESHOLD):
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

                # low AF in tumor and normal, rare in population implies artifact
                if (not has_normal or normal_af < 0.2) and af < 0.2 and metadata.popaf() > ARTIFACT_POPAF_THRESHOLD:
                    unmatched_artifact_counts.extend([alt_count] * NON_ARTIFACT_PER_ARTIFACT)
                    is_artifact = True
                # high AF in tumor and normal, common in population implies germline, which we downsample
                elif (
                        not has_normal or normal_af > 0.35) and af > 0.35 and metadata.popaf() < GERMLINE_POPAF_THRESHOLD and unmatched_artifact_counts:
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

            data.append(Datum(ref_tensor, alt_tensor, info_tensor, metadata, m2_data, 1 if is_artifact else 0))
        if shuffle:
            random.shuffle(data)
        return data
