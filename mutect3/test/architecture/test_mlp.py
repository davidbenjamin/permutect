import torch
from mutect3.architecture.mlp import MLP


# test with artificial data where a*x = 0 is a perfect linear separator
def test_linearly_separable_data():
    input_dim = 3
    num_samples = 1000
    a = torch.rand(input_dim, 1)
    x = torch.rand(num_samples, input_dim)
    y = (torch.sign(torch.matmul(x,a)) + 1)/2   # labels are 0 / 1

    layer_sizes = [input_dim, 1]
    model = MLP(layer_sizes)

    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []

    num_epochs = 10000
    for epoch in range(num_epochs):
        prediction = model.forward(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    assert loss_list[-1] < 0.01
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]

    pred = torch.sign(torch.sigmoid(model.forward(x))-0.5)
    lab = torch.sign(y-0.5)

    errors = torch.sum(torch.abs((pred - lab)/2)).item()

    # we should have perfect accuracy
    assert errors < 1


# test with annular data where y = 1 when  1/3 < norm(x) < 2/3
def test_annular_data():
    input_dim = 3
    num_samples = 1000
    x = torch.rand(num_samples, input_dim)/torch.sqrt(torch.tensor([input_dim]))

    norms = torch.norm(x, dim=1)
    y = (torch.sign(norms - 0.33) * torch.sign(0.66 - norms) + 1)/2

    layer_sizes = [input_dim, 5, 5, 5, 5, 1]
    model = MLP(layer_sizes)

    loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []

    num_epochs = 10000
    for epoch in range(num_epochs):
        prediction = model.forward(x)
        loss = loss_func(torch.squeeze(prediction), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    assert loss_list[-1] < 0.2

    pred = torch.squeeze(torch.sign(torch.sigmoid(model.forward(x)) - 0.5))
    lab = torch.sign(y - 0.5)

    errors = torch.sum(torch.abs((pred - lab) / 2)).item()

    # we should have perfect accuracy
    assert errors < 100
