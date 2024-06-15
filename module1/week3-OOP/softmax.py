import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x / torch.sum(exp_x)

class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x - torch.max(x))
        return exp_x / torch.sum(exp_x)

class SoftmaxStable2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0, keepdims=True)
        x_exp = torch.exp(x - x_max.values)
        partition = x_exp.sum(0, keepdim=True)
        return x_exp / partition


if __name__ == '__main__':
    data = torch.Tensor([1, 2, 3])
    softmax = Softmax()
    output = softmax(data)
    print(output)

    softmax_stable = SoftmaxStable()
    output = softmax_stable(data)
    print(output)

    softmax_function = nn.Softmax(dim=0)
    output = softmax_function(data)
    assert round(output[0].item(), 2) == 0.09
    print(output)

    data2 = torch.Tensor([5, 2, 4])
    output = softmax(data2)
    assert round(output[-1].item(), 2) == 0.26
    print(output)

    data3 = torch.Tensor([1, 2, 3e8])
    output = softmax(data3)
    assert round(output[0].item(), 2) == 0
    print(output)

    softmax_stable = SoftmaxStable2()
    output = softmax_stable(data)
    assert round(output[-1].item(), 2) == 0.67
    print(output)
