import torch


if __name__ == "__main__":
    w = torch.tensor([[-0.1]], requires_grad=True)
    x = torch.tensor([[0.4, 0.5, -0.2]])
    y = torch.tensor([[0.4, 0.5, -0.2]])


    y_hat = x * w
    loss = torch.nn.functional.mse_loss(y_hat, y, reduction='mean')
    loss.backward()

    print(w.grad)
    print(loss.item())
    print(y_hat)