import torch
import torch.nn.functional as F
from torch import nn
from config import config
from textual import Pyramid
from attention import MHCattention
from pred_head import Head



import numpy as np
 

class VSG(nn.Module):

    def __init__(self):

        super().__init__()
        self.conf = config()
        self.proj_d = self.conf.MODEL.proj
        self.visual_proj = nn.Linear(4096, self.proj_d)
        self.text = Pyramid()
        self.coarse_video = nn.Conv2d(in_channels=n,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)
        self.finer_video = nn.Conv2d(in_channels=n,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)
        self.fine_video = nn.Conv2d(in_channels=n,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)
        self.MHCA = MHCattention()
        self.coarse_pred = Head()
        self.finer_pred = Head()
        self.final_pred = Head()  
        self.loss_1=''     


    def forward(self, text, text_masks, clips, temp_bound):
        clip_p = self.visual_proj(clips)
        fine_t, coarser_t, coarse_t = self.Pyramid(text)
        hidden_video = self.MHCA(clip_p, fine_t)
        video = self.coarse_video(hidden_video)
        pred_1 = self.coarse_pred(video)
        #TODO crop by prediction: if Spred< Sgt AND Epred>Egt
        if pred_1[0]<temp_bound[0] and pred_1[1]>temp_bound[1]:
            video = video[:,:,pred_1[0]:pred_1[1]]
        coarser_text = self.finer(fine_text)
        hidden_video = self.MHCA(video, coarser_text)
        video = self.finer_video(hidden_video)
        pred_2 = self.finer_pred(video)
        #TODO crop by prediction: if Spred< Sgt AND Epred>Egt
        if pred_2[0]<temp_bound[0] and pred_2[1]>temp_bound[1]:
            video = video[:,:,pred_2[0]:pred_2[1]]
        coarse_text = self.coarse(coarser_text)
        hidden_video = self.MHCA(video, coarse_text)
        video = self.finer_video(hidden_video)
        pred_3 = self.finer_pred(video)
        out = [pred_1, pred_2, pred_3]

        return out