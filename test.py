import torch
from torch import nn
import torch.nn.functional as F
text = torch.load("text.pt")
#torch.Size([5, 300)
video = torch.load("video.pt")
import numpy as np
#torch.Size([607, 4096])

# num_sample_clips = 512
# #video_1 = torch.randint(1,5, (210, 4096), dtype=torch.float)
# num_clips = video.shape[0]
# idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips

# idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
# new_visual_input = []
# for i in range(num_sample_clips):
#         s_idx, e_idx = idxs[i].item(), idxs[i+1].item() #0, 1
#         #print(video[s_idx: e_idx].shape) --> 2,4096 OR 1,4096!
#         if s_idx < e_idx:
#             new_visual_input.append(torch.mean(video[s_idx:e_idx],dim=0))

#         else:
#             new_visual_input.append(video[s_idx])
# new_visual_input = torch.stack(new_visual_input, dim=0) #512x4096

video = torch.randint(1,5, (512, 4096), dtype=torch.float)
