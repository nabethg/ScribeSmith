from pprint import pprint
from itertools import permutations as perm, combinations as comb
import torch
import torch.nn as nn

# Dictionary to map labels to indices and vectors
label_to_index = {"0": 0, "a": 1, "b": 2, "c": 3}
label_to_vector = {
    "0": [1.0, -1.0, -1.0, -1.0],
    "a": [-1.0, 1.0, -1.0, -1.0],
    "b": [-1.0, -1.0, 1.0, -1.0],
    "c": [-1.0, -1.0, -1.0, 1.0],
}

# Given sequence
sequence = "0a0b0a0c0a0b0b0a0a0"

# Generate the tensor based on the sequence
tensor_list = [label_to_vector[label] for label in sequence]
tensor = torch.tensor(tensor_list, dtype=torch.float32).unsqueeze(1)

print(tensor.shape)

seq = "aaaaabbbc"
unique_perms = list(set(list(perm(seq))))

ctc_loss = nn.CTCLoss()

loss_and_seq = []
for i in unique_perms:
    loss = ctc_loss(
        tensor.log_softmax(2),
        torch.tensor([{"a":1,"b":2,"c":3}[j] for j in i]),
        torch.tensor([19]),
        torch.tensor([9])
    )
    loss_and_seq.append((loss.item(), i))
loss_and_seq.sort()

pprint(loss_and_seq)

# input_lengths