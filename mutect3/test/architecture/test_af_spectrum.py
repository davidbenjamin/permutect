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
    metrics = TrainingMetrics()

    num_epochs = 10000
    for epoch in range(num_epochs):
        loss = -torch.sum(model.log_likelihood(torch.squeeze(alt_counts), torch.squeeze(depths)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metrics.add("NLL", "TRAINING", loss.item())

    loss_list = metrics.metrics.get("NLL").get("TRAINING")
    assert loss_list[-1] < 0.01
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]

    #pred = torch.sign(torch.sigmoid(model.forward(x))-0.5)
    #lab = torch.sign(y-0.5)

    #errors = torch.sum(torch.abs((pred - lab)/2)).item()

    # we should have perfect accuracy
    #assert errors < 1