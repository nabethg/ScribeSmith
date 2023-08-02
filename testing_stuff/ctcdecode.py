import torch
import torch.nn
import torch.nn as nn

# class GreedyCTCDecoder(torch.nn.Module):
#     def __init__(self, labels, blank=0):
#         super().__init__()
#         self.labels = labels
#         self.blank = blank

#     def forward(self, emission: torch.Tensor) -> List[str]:
#         """Given a sequence emission over labels, get the best path
#         Args:
#           emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

#         Returns:
#           List[str]: The resulting transcript
#         """
#         indices = torch.argmax(emission, dim=-1)  # [num_seq,]
#         indices = torch.unique_consecutive(indices, dim=-1)
#         indices = [i for i in indices if i != self.blank]
#         joined = "".join([self.labels[i] for i in indices])
#         return joined.replace("|", " ").strip().split()
import editdistance
# out = torch.tensor([9, 9, 9, 8, 8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 22, 4, 5, 5])
# print(editdistance.eval("among", "self.labels for i in indicedss")/len("among"))

from torchmetrics.text import CharErrorRate
preds = ["this is the prediction", "there is an other sample"]
target = ["this is the reference", "there is another one"]
# 0.3810
# 0.3000
preds = ["there is an other sample"]
target = ["there is another one"]

cer = CharErrorRate()
print(cer(preds, target))



# print(torch.unique_consecutive(out))

