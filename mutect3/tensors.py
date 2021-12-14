import torch
import random
import numpy as np
import pickle
import pysam
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
        return utils.VariantType.SNV if diff == 0 else (
            utils.VariantType.INSERTION if diff > 0 else utils.VariantType.DELETION)


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
    def __init__(self, ref_tensor: torch.Tensor, alt_tensor: torch.Tensor, info_tensor: torch.Tensor,
                 site_info: SiteInfo,
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

    def set_label(self, label):
        self._artifact_label = label

    # beta is distribution of downsampling fractions
    def downsampled_copy(self, beta: Beta):
        ref_frac = beta.sample().item()
        alt_frac = beta.sample().item()

        ref_length = max(1, round(ref_frac * len(self._ref_tensor)))
        alt_length = max(1, round(alt_frac * len(self._alt_tensor)))
        ref = downsample(self._ref_tensor, ref_length)
        alt = downsample(self._alt_tensor, alt_length)
        return Datum(ref, alt, self.info_tensor(), self.site_info(), self.mutect_info(), self.artifact_label(),
                     self.normal_depth(), self.normal_alt_count())


def unlabeled_datum_from_vcf(rec: pysam.VariantRecord, tumor, normal, ref_downsample):
    #print(rec.contig + ':' + str(rec.pos))
    has_normal = normal is not None

    tumor_dp = rec.samples[tumor]["DP"]
    normal_dp = rec.samples[normal]["DP"] if has_normal else 0

    tumor_frs_cnt = rec.samples[tumor]["FRSCNT"]
    normal_alt_count = rec.samples[normal]["FRSCNT"][1] if has_normal else 0

    if sum(tumor_frs_cnt) > 0:
        tumor_frs = rec.samples[tumor]["FRS"]
        ref_tensor = torch.tensor(tumor_frs[:tumor_frs_cnt[0] * NUM_READ_FEATURES]).float().reshape((-1, NUM_READ_FEATURES))
        alt_tensor = torch.tensor(tumor_frs[tumor_frs_cnt[0] * NUM_READ_FEATURES:]).float().reshape((-1, NUM_READ_FEATURES))
        ref_tensor = downsample(ref_tensor, ref_downsample)
    else:
        ref_tensor = torch.tensor([]).float().reshape((-1, NUM_READ_FEATURES))
        alt_tensor = torch.tensor([]).float().reshape((-1, NUM_READ_FEATURES))

    info = rec.info
    site_info = SiteInfo(rec.contig, rec.pos, rec.alleles[0], rec.alleles[1], info['POPAF'][0])
    variant_info = VariantInfo(info["HEC"], info["HAPCOMP"][0], info["HAPDOM"][0], info["REF_BASES"])
    mutect_info = MutectInfo(info["TLOD"][0], tumor_dp, set(rec.filter))

    return Datum(ref_tensor, alt_tensor, variant_info.info_tensor(), site_info, mutect_info, None, normal_dp,
                 normal_alt_count)


# pickle and unpickle a Python list of Datum objects.  Convenient to have here because unpickling needs to have all
# the constituent classes of Datum explicitly imported.
def make_pickle(file, datum_list):
    with open(file, 'wb') as f:
        pickle.dump(datum_list, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


class NormalArtifactDatum:
    def __init__(self, normal_alt_count: int, normal_depth: int, tumor_alt_count: int, tumor_depth: int,
                 downsampling: float, variant_type: str):
        self._normal_alt_count = normal_alt_count
        self._normal_depth = normal_depth
        self._tumor_alt_count = tumor_alt_count
        self._tumor_depth = tumor_depth
        self._downsampling = downsampling
        self._variant_type = variant_type

    def normal_alt_count(self) -> int:
        return self._normal_alt_count

    def normal_depth(self) -> int:
        return self._normal_depth

    def tumor_alt_count(self) -> int:
        return self._tumor_alt_count

    def tumor_depth(self) -> int:
        return self._tumor_depth

    def downsampling(self) -> float:
        return self._downsampling

    def variant_type(self) -> str:
        return self._variant_type


class NormalArtifactTableReader:
    def __init__(self, header_tokens):
        self.normal_alt_idx = header_tokens.index("normal_alt")
        self.normal_dp_idx = header_tokens.index("normal_dp")
        self.tumor_alt_idx = header_tokens.index("tumor_alt")
        self.tumor_dp_idx = header_tokens.index("tumor_dp")
        self.downsampling_idx = header_tokens.index("downsampling")
        self.type_idx = header_tokens.index("type")

    def normal_alt_count(self, tokens):
        return int(tokens[self.normal_alt_idx])

    def normal_depth(self, tokens):
        return int(tokens[self.normal_dp_idx])

    def tumor_alt_count(self, tokens):
        return int(tokens[self.tumor_alt_idx])

    def tumor_depth(self, tokens):
        return int(tokens[self.tumor_dp_idx])

    def downsampling(self, tokens):
        return float(tokens[self.downsampling_idx])

    def variant_type(self, tokens):
        return tokens[self.type_idx]


def read_normal_artifact_data(table_file, shuffle=True) -> List[NormalArtifactDatum]:
    data = []

    with open(table_file) as fp:
        reader = NormalArtifactTableReader(fp.readline().split())

        pbar = tqdm(enumerate(fp))
        for n, line in pbar:
            tokens = line.split()

            normal_alt_count = reader.normal_alt_count(tokens)
            normal_depth = reader.normal_depth(tokens)
            tumor_alt_count = reader.tumor_alt_count(tokens)
            tumor_depth = reader.tumor_depth(tokens)
            downsampling = reader.downsampling(tokens)
            variant_type = reader.variant_type(tokens)

            data.append(NormalArtifactDatum(normal_alt_count, normal_depth, tumor_alt_count, tumor_depth, downsampling,
                                            variant_type))

    if shuffle:
        random.shuffle(data)
    print("Done")
    return data


def generate_normal_artifact_pickle(table_file, pickle_file):
    data = read_normal_artifact_data(table_file)
    make_pickle(pickle_file, data)


def downsample(tensor: torch.Tensor, downsample_fraction) -> torch.Tensor:
    if downsample_fraction is None or downsample_fraction >= len(tensor):
        return tensor
    else:
        return tensor[torch.randperm(len(tensor))[:downsample_fraction]]


NUM_READ_FEATURES = 11  # size of each read's feature vector from M2 annotation
NUM_INFO_FEATURES = 9  # size of each variant's info field tensor (3 components for HEC, one each for HAPDOM, HAPCOMP)
# and 5 for ref bases STR info

RARE_POPAF = 5.9  # only let things absent from gnomAD be artifacts out of caution
COMMON_POPAF = 1  # also very cautious.  There are so many germline variants we can be wasteful!

REF_DOWNSAMPLE = 20  # choose this many ref reads randomly
MIN_REF = 5

TLOD_THRESHOLD = 6  # we are classified artifacts other than sequencing errors described by base qualities
NON_ARTIFACT_PER_ARTIFACT = 20  # ratio of non-artifact to artifact in unsupervised training data


def make_training_tensors_from_vcf(vcf, tumor, normal=None, shuffle=True) -> List[Datum]:
    data = []

    # simple method to balance data: for each k-alt-read artifact there are
    # NON_ARTIFACT_PER_ARTIFACT (downsampled) k-alt-read non-artifacts.
    unmatched_counts_by_type = defaultdict(list)
    pbar = tqdm()
    for n, rec in enumerate(pysam.VariantFile(vcf)):
        if n % 10000 == 0:
            print(rec.contig + ':' + str(rec.pos))
        datum = unlabeled_datum_from_vcf(rec, tumor, normal, REF_DOWNSAMPLE)
        alt_count = len(datum.alt_tensor())
        if alt_count == 0 or len(datum.ref_tensor()) < MIN_REF:
            continue

        tumor_af = alt_count / datum.mutect_info().tumor_depth()
        has_normal = normal is not None
        normal_dp = datum.normal_depth()
        normal_alt_count = datum.normal_alt_count()
        normal_af = 0 if normal_dp == 0 else normal_alt_count / normal_dp

        likely_seq_error = "weak_evidence" in datum.mutect_info().filters() or datum.mutect_info().tlod() < TLOD_THRESHOLD
        likely_germline = has_normal and normal_af > 0.2

        popaf = datum.site_info().popaf()
        # extremely strict criteria because there are so many germline variants we can afford to waste a lot
        definite_germline = not likely_seq_error and popaf < COMMON_POPAF and (
                tumor_af > 0.35 and popaf < COMMON_POPAF) and (normal_af > 0.35 if has_normal else True)

        unmatched_artifact_counts = unmatched_counts_by_type[
            utils.get_variant_type(datum.site_info().alt(), datum.site_info().ref())]

        # low AF in tumor and normal, rare in population implies artifact
        if not (likely_seq_error or likely_germline) and tumor_af < 0.2 and popaf > RARE_POPAF:
            unmatched_artifact_counts.extend([alt_count] * NON_ARTIFACT_PER_ARTIFACT)
            datum.set_label(1)
        # high AF in tumor and normal, common in population implies germline, which we downsample
        elif definite_germline and unmatched_artifact_counts:
            downsample_count = min(alt_count, unmatched_artifact_counts.pop())

            # TODO: make this a set method or something!!!!!
            datum._alt_tensor = datum.alt_tensor()[torch.randperm(alt_count)[:downsample_count]]
            datum.set_label(0)
        # inconclusive -- unlabeled datum
        # to avoid too large a dataset, we try to bias toward possible artifacts and keep out obvious sequencing errors
        elif datum.mutect_info().tlod() > 4.0 and tumor_af < 0.3:
            downsample_count = min(alt_count, 10)
            # TODO: a setter would be better!!
            datum._alt_tensor = datum.alt_tensor()[torch.randperm(alt_count)[:downsample_count]]
        else:
            continue

        data.append(datum)
    return data
