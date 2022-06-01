import torch
from torch.distributions.binomial import Binomial
from mutect3.architecture.beta_binomial_mixture import BetaBinomialMixture


def test_make_one_component_dominate():
    component_to_dominate = 2
    model = BetaBinomialMixture(input_size=1, num_components=5)

    with torch.no_grad():
        model.weights_pre_softmax.weight[component_to_dominate] += 10   # boost this row/component

    r = 9




# test with artificial data where there is a single AF, hence count data is binomial
def test_single_af():
    af = 0.2
    num_samples = 1000

    # we're only having one variant type, so the one-hot input is trivial -- just a bunch of 1s
    dummy_input = torch.ones((num_samples, 1))
    depths = torch.randint(low=10, high=50, size=(num_samples, 1))
    alt_counts = Binomial(depths, torch.tensor([af])).sample().squeeze()

    model = BetaBinomialMixture(input_size=1, num_components=5) # input_size=1 means one spectrum eg for SNVs
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 10000
    for epoch in range(num_epochs):
        loss = -torch.mean(model.forward(dummy_input, alt_counts, depths.squeeze()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    N = 100
    depth = 1000
    depths = torch.Tensor([depth for _ in range(N)])
    dummy_input = torch.ones((N, 1))
    counts = model.sample(dummy_input, depths)

    model.plot_spectrum(torch.Tensor([1]), "TITLE")
    assert torch.abs(torch.mean(counts.float()) - af*depth).item() < depth/50
    assert torch.sum(counts > depth*(af+0.1)) < 5
    torch.sum(counts < depth * (af - 0.1)) < 5
