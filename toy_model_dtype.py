import torch
from torch import nn

x = torch.randn((10, 10), dtype=torch.float32)


class ToyModel(nn.Module):
    def __int__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


model = ToyModel(10, 10)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    print(model(x).dtype())
