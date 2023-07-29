import torch
import torch.nn as nn
import torch.optim as optim


class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 100)
        self.linear2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x

model = NetA()
output = torch.randn((32, 30))
output = model(output)

labels = torch.ones_like(output)
print(labels)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
print(loss)
loss.backward()

for name, param in model.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")
print()
for name, param in model.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")

