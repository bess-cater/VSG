import torch
import torch.nn.functional as F
from torch import nn

#here attention just as in transformer!!!
class MHCattention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.coarse = nn.Conv1d(in_channels=300,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)
        self.m_pool = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, text, text_masks):
        attn = self.sa(text)
        coarse = self.coarse(attn)
        out = self.m_pool(coarse)

        return out