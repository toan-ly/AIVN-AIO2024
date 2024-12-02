import torch.nn as nn

class FineTuningMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FineTuningMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.flatten = nn.Flatten()

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = nn.Sigmoid(x)
        x = self.layer2(x)
        x = nn.Sigmoid(x)
        x = self.output(x)
        return x