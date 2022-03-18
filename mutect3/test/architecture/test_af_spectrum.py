import torch
from torch.distributions.binomial import Binomial
from mutect3.architecture.af_spectrum import AFSpectrum
from mutect3.metrics.training_metrics import TrainingMetrics


# test with artificial data where there is a single AF, hence count data is binomial
def test_single_af():
    af = 0.2
    num_samples = 1000

    depths = torch.randint(low=10, high=50, size=(num_samples, 1))
    alt_counts = Binomial(depths, torch.tensor([af])).sample()

    model = AFSpectrum()
    optimizer = torch.optim.Adam(model.parameters())

    num_epochs = 10000
    for epoch in range(num_epochs):
        loss = -torch.sum(model.log_likelihood(torch.squeeze(alt_counts), torch.squeeze(depths)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    depth = 1000
    depths = [depth for _ in range(100)]
    counts = model.sample(depths)

    assert torch.abs(torch.mean(counts.float()) - af*depth).item() < depth/50
    assert torch.sum(counts > depth*(af+0.1)) < 5
    torch.sum(counts < depth * (af - 0.1)) < 5
