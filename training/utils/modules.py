import torch


class LinearBlock(torch.nn.Module):
    def __init__(self, input_dim, linear_dim, drop_p=0.):
        super(LinearBlock, self).__init__()
        self.linear_layer = torch.nn.Linear(input_dim, linear_dim)
        self.dropout = torch.nn.Dropout(p=drop_p)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.linear_layer(x)
        out = self.dropout(out)
        out = self.relu(out)
        return out