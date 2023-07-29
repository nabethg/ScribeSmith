import torch
import torch.nn as nn
import torch.optim as optim

class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(30, 100)
        self.linear2 = nn.Linear(100, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x


class NetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 10)
        self.linear2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x

out = torch.randn((32, 30))
neta = NetA()
netb = NetB()

print(out.shape)
out = neta(out)
print(out.shape)
out = netb(out)
print(out.shape)

# optima = optim.SGD(neta.parameters(), lr=0.01)
# optimb = optim.SGD(netb.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for name, param in neta.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")
print()
for name, param in netb.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")

loss = out.sum()
loss.backward()

# print(loss)

# loss.backward
print("NETWORK")

for name, param in neta.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")
print()
for name, param in netb.named_parameters():
    # if param.requires_grad and param.grad is not None:
    print(f"Parameter: {name}, Gradient: {param.grad}")

