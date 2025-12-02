import math

import torch
from matplotlib import pyplot as plt
from torch import nn, IntTensor
from torch.distributions import Beta

from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate
from permutect.utils.array_utils import index_tensor
from permutect.utils.stats_utils import uniform_binomial_log_lk, beta_binomial_log_lk
from permutect.utils.enums import Variation


DEPTH_CUTOFFS = [10, 20]
NUM_DEPTH_BINS = len(DEPTH_CUTOFFS) + 1


# the bin is the number of cutoffs that are met or exceeded
def depths_to_depth_bins(depths_b: IntTensor):
    assert len(DEPTH_CUTOFFS) == 2
    return (depths_b >= DEPTH_CUTOFFS[0]).long() + (depths_b >= DEPTH_CUTOFFS[1]).long()


class ArtifactSpectra(nn.Module):
    """
    a beta-binomial distributions for each variant type
    """

    def __init__(self):
        super(ArtifactSpectra, self).__init__()
        self.V = len(Variation)
        self.D = NUM_DEPTH_BINS

        self.alpha_pre_exp_dv = torch.nn.Parameter(torch.log(2*torch.ones(self.D, self.V)))
        self.beta_pre_exp_dv = torch.nn.Parameter(torch.log(30 * torch.ones(self.D, self.V)))


    '''
    here x is a 2D tensor, 1st dimension batch, 2nd dimension being features that determine which Beta mixture to use
    n and k are 1D tensors, the only dimension being batch.
    '''
    def forward(self, variant_types_b: IntTensor, depths_b: IntTensor, alt_counts_b):
        var_types_b = variant_types_b.long()
        depth_bins_b = depths_to_depth_bins(depths_b)

        alpha_dv, beta_dv = torch.exp(self.alpha_pre_exp_dv), torch.exp(self.beta_pre_exp_dv)
        alpha_b = index_tensor(alpha_dv, (depth_bins_b, var_types_b))
        beta_b = index_tensor(beta_dv, (depth_bins_b, var_types_b))
        result_b = beta_binomial_log_lk(n=depths_b, k=alt_counts_b, alpha=alpha_b, beta=beta_b)
        return result_b

    # TODO: utter code duplication with somatic spectrum
    def fit(self, num_epochs: int, types_b: IntTensor, depths_b: IntTensor, alt_counts_b: IntTensor, batch_size: int=64):
        optimizer = torch.optim.Adam(self.parameters())
        num_batches = math.ceil(len(alt_counts_b) / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_start = batch * batch_size
                batch_end = min(batch_start + batch_size, len(alt_counts_b))
                batch_slice = slice(batch_start, batch_end)
                loss = -torch.mean(self.forward(types_b[batch_slice], depths_b[batch_slice], alt_counts_b[batch_slice]))
                backpropagate(optimizer, loss)

    '''
    get raw data for a spectrum plot of probability density vs allele fraction for a particular variant type
    '''
    def spectrum_density_vs_fraction(self, variant_type: Variation, depth: int):
        depth_bin = depths_to_depth_bins(torch.tensor(depth)).item()
        fractions_f = torch.arange(0.01, 0.99, 0.001)  # 1D tensor

        # scalar tensors
        alpha, beta = torch.exp(self.alpha_pre_exp_dv[depth_bin, variant_type]), torch.exp(self.beta_pre_exp_dv[depth_bin, variant_type])
        alpha_1, beta_1 = alpha.view(1).to(device=fractions_f.device), beta.view(1).to(device=fractions_f.device)
        densities_f = torch.exp(Beta(alpha_1, beta_1).log_prob(fractions_f))
        return fractions_f, densities_f

    # this works for ArtifactSpectra and OverdispersedBinomialMixture
    def plot_artifact_spectra(self, depth: int = None):
        # plot AF spectra in two-column grid with as many rows as needed
        art_spectra_fig, art_spectra_axs = plt.subplots(math.ceil(len(Variation) / 2), 2, sharex='all', sharey='all')
        for variant_type in Variation:
            n = variant_type
            row, col = int(n / 2), n % 2
            frac, dens = self.spectrum_density_vs_fraction(variant_type, depth)
            art_spectra_axs[row, col].plot(frac.detach().numpy(), dens.detach().numpy(), label=variant_type.name)
            art_spectra_axs[row, col].set_title(variant_type.name + " artifact AF spectrum")
        for ax in art_spectra_fig.get_axes():
            ax.label_outer()
        return art_spectra_fig, art_spectra_axs

    '''
    here x is a 1D tensor, a single datum/row of the 2D tensors as above
    '''
    def plot_spectrum(self, variant_type: Variation, title, depth: int):
        fractions, densities = self.spectrum_density_vs_fraction(variant_type, depth)
        return simple_plot([(fractions.numpy(), densities.numpy(), " ")], "AF", "density", title)
