import torch
import random
from mutect3.data.read_set_datum import NUM_READ_FEATURES, NUM_INFO_FEATURES, ReadSetDatum
from mutect3.utils import VariantType

BASES = ['A', 'C', 'G', 'T']


# random isotropic Gaussian tensor, dilated by different amounts in each dimension
def make_random_tensor(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    assert mean.size() == std.size()

    # TODO: random normal needs same length as mean
    return mean + std * torch.randn(len(mean))


class RandomInfoGenerator:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        assert len(mean) == NUM_INFO_FEATURES
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def generate(self) -> torch.Tensor:
        return make_random_tensor(self.mean, self.std)


class RandomReadGenerator:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        assert len(mean) == NUM_READ_FEATURES
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def generate(self, num_reads: int) -> torch.Tensor:
        return torch.vstack([make_random_tensor(self.mean, self.std) for _ in range(num_reads)])


# return tuple of ref allele, alt_allele
def make_random_ref_and_alt_alleles(variant_type: VariantType):
    ref_index = random.randint(0, len(BASES) - 1)
    alt_index = (ref_index + random.randint(1, len(BASES) - 1)) % len(BASES)
    if variant_type == VariantType.SNV:
        return BASES[ref_index], BASES[alt_index]
    # make a 1-base insertion
    elif variant_type == VariantType.INSERTION:
        return BASES[ref_index], BASES[ref_index] + BASES[alt_index]
    elif variant_type == VariantType.DELETION:
        return BASES[alt_index] + BASES[ref_index] , BASES[alt_index]


def make_random_data(art_info_gen: RandomInfoGenerator, var_info_gen: RandomInfoGenerator, art_read_gen: RandomReadGenerator,
                     var_read_gen: RandomReadGenerator, num_data: int, artifact_fraction=0.5, unlabeled_fraction=0.1,
                     indel_fraction=0.2, ref_downsampling=10):






    data = []
    for _ in range(0, num_data):
        position = random.randint(1, 1000000)

        # generate label
        artifact = random.uniform(0,1) < artifact_fraction
        unlabeled = random.uniform(0,1) < unlabeled_fraction
        label = "UNLABELED" if unlabeled else ("ARTIFACT" if artifact else "VARIANT")

        # generate variant type
        indel = random.uniform(0,1) < indel_fraction
        variant_type = (VariantType.DELETION if random.uniform(0,1) < 0.5 else VariantType.INSERTION) if indel else VariantType.SNV

        ref_count = ref_downsampling
        alt_count = random.randint(3, 10)

        # we assume artifact used the original alts but variant was downsampled from a het
        # TODO: model alt allele fraction for test data
        pd_tumor_depth, pd_alt_count = (100, alt_count if artifact else 50)

        # TODO: eventually model normal artifact, but not yet
        normal_depth, normal_alt_count = 50, 0

        ref, alt = make_random_ref_and_alt_alleles(variant_type)

        info_tensor = (art_info_gen if artifact else var_info_gen).generate()

        ref_tensor = var_read_gen.generate(ref_count)
        alt_tensor = (art_read_gen if artifact else var_read_gen).generate(alt_count)

        data.append(ReadSetDatum("CONTIG", position, ref, alt, ref_tensor, alt_tensor, info_tensor, label,
                                 pd_tumor_depth, pd_alt_count, normal_depth, normal_alt_count))

    return data


# good and bad data are generated by distinct gaussians
def make_two_gaussian_data(num_data):
    var_info_mean = torch.tensor([-1]*9)
    var_info_std = torch.tensor([1]*9)
    art_info_mean = torch.tensor([1] * 9)
    art_info_std = torch.tensor([1] * 9)

    var_info_gen = RandomInfoGenerator(var_info_mean, var_info_std)
    art_info_gen = RandomInfoGenerator(art_info_mean, art_info_std)

    var_read_mean = torch.tensor([-1] * 11)
    var_read_std = torch.tensor([1] * 11)
    art_read_mean = torch.tensor([1] * 11)
    art_read_std = torch.tensor([1] * 11)

    var_read_gen = RandomReadGenerator(var_read_mean, var_read_std)
    art_read_gen = RandomReadGenerator(art_read_mean, art_read_std)

    return make_random_data(art_info_gen=art_info_gen, var_info_gen=var_info_gen, art_read_gen=art_read_gen,
                            var_read_gen=var_read_gen, num_data=num_data)


# good and bad data are generated by gaussians with same mean (0) but artifacts are much more spread out
def make_wide_and_narrow_gaussian_data(num_data):
    var_info_mean = torch.tensor([0]*9)
    var_info_std = torch.tensor([1]*9)
    art_info_mean = torch.tensor([0] * 9)
    art_info_std = torch.tensor([2] * 9)

    var_info_gen = RandomInfoGenerator(var_info_mean, var_info_std)
    art_info_gen = RandomInfoGenerator(art_info_mean, art_info_std)

    var_read_mean = torch.tensor([0] * 11)
    var_read_std = torch.tensor([1] * 11)
    art_read_mean = torch.tensor([0] * 11)
    art_read_std = torch.tensor([2] * 11)

    var_read_gen = RandomReadGenerator(var_read_mean, var_read_std)
    art_read_gen = RandomReadGenerator(art_read_mean, art_read_std)

    return make_random_data(art_info_gen=art_info_gen, var_info_gen=var_info_gen, art_read_gen=art_read_gen,
                            var_read_gen=var_read_gen, num_data=num_data)


