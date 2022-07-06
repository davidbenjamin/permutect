import torch
from torch.distributions.binomial import Binomial
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture


# compare learned beta binomial mixture to the data it was trained on
def compare_model_to_data(model: BetaBinomialMixture, inputs_2d_tensor, depths_1d_tensor, alt_counts_1d_tensor):
    generated_alt_counts = model.sample(inputs_2d_tensor, depths_1d_tensor)

    means = torch.mean(
    )


# given a discrete distribution of allele fractions between 0 and 1, and desired depths, generate alt counts,
# fit a BetaBinomialMixture, and compare moments of the underlying Beta mixture (without the binomial part) to
# those of the empirical allele fractions
def test_on_discrete_af_distribution(fractions_1d: torch.Tensor, weights_1d: torch.Tensor, training_depths_1d: torch.Tensor,
                                num_components: int = 10, num_epochs=1000):

    idx = weights_1d.multinomial(num_samples=len(training_depths_1d), replacement=True)
    empirical_fractions_1d = fractions_1d[idx]

    empirical_counts = Binomial(training_depths_1d, empirical_fractions_1d).sample().squeeze()
    dummy_input = torch.ones((len(empirical_counts), 1))

    model = BetaBinomialMixture(input_size=1, num_components=num_components)
    model.fit(num_epochs=num_epochs, inputs_2d_tensor=dummy_input, depths_1d_tensor=training_depths_1d,
              alt_counts_1d_tensor=empirical_counts)

    # moments E[x], E[ln(x)], E[x ln(x)]
    model_mean, model_log_mean, model_log_linear_mean = model.moments_of_underlying_beta_mixture(torch.Tensor([1]))

    given_mean = torch.sum(weights_1d * fractions_1d)
    given_log_mean = torch.sum(weights_1d * torch.log(fractions_1d))
    given_log_linear_mean = torch.sum(weights_1d * fractions_1d * torch.log(fractions_1d))

    assert torch.abs(model_mean - given_mean).item() < 0.02
    assert torch.abs(model_log_mean - given_log_mean).item() < 0.1
    assert torch.abs(model_log_linear_mean - given_log_linear_mean).item() < 0.02


def test_single_component():
    num_samples = 1000
    for fraction in [0.05, 0.1, 0.5, 0.75]:
        for depth in [10, 100, 1000]:
            depths = depth*torch.ones(num_samples).int()
            test_on_discrete_af_distribution(fractions_1d=torch.Tensor([fraction]), weights_1d=torch.Tensor([1.0]), training_depths_1d=depths)


def test_two_components():
    num_samples = 1000
    depth = 100
    depths = depth * torch.ones(num_samples).int()
    for fraction_pair in [(0.1, 0.9), (0.2, 0.4), (0.4, 0.8)]:
        for weight_pair in [(0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]:
            test_on_discrete_af_distribution(fractions_1d=torch.Tensor(fraction_pair), weights_1d=torch.Tensor(weight_pair), training_depths_1d=depths)


def test_make_one_component_dominate():
    means = torch.Tensor([0.1, 0.3, 0.5, 0.7])
    model = BetaBinomialMixture(input_size=1, num_components=len(means))
    model.set_means(means)
    model.set_weights(torch.Tensor([10, 2, 2, 5]))
    model.plot_spectrum(torch.Tensor([1]), "T") # spot check spectrum