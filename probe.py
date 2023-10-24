#take a random tensor and do your shit with it.
import os
import json

import h5py
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import h5py
import numpy as np
import glob

filename = "data/tacos/tall_c3d_features.hdf5"
id = 's30-d52.avi'


vocab = torchtext.vocab.pretrained_aliases["glove.840B.300d"]() #! from 6B to 840B!
vocab.itos.extend(['<unk>'])
vocab.stoi['<unk>'] = vocab.vectors.shape[0]
vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
print("shapeeeeeeeeeeeee")
print(vocab.vectors.shape)
word_embedding = nn.Embedding.from_pretrained(vocab.vectors)
probe={}
probe["id"]=id
with open("data/tacos/anno/probe.json",'r') as f:
    annos = json.load(f)
    for k in annos.keys():
        if k==id:
            probe["time"] = annos[k]["timestamps"][1]
            sent = annos[k]["sentences"][1]
            print(sent.split())
            word_idxs = torch.tensor([vocab.stoi.get(w.lower(), 400000) for w in sent.split()], dtype=torch.long)
            print(word_idxs)
            word_vectors = word_embedding(word_idxs)
            print(word_vectors.shape)
            probe["sent"]=word_vectors
            torch.save(word_vectors, 'text.pt')
with h5py.File(filename, "r") as f:
    video = torch.tensor(f.get(id))
    torch.save(video, 'video.pt')
    print(video.shape)
    probe["video"]=video

    #s30-d52.avi
    