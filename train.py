import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#import dataset
#--> train and val!!!
#import 
def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        outputs = model(**model_inputs)
        losses = criterion(outputs, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    criterion.eval()
    submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)

def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    train_loader = DataLoader(
            train_dataset,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
    val_loader = DataLoader(
            val_dataset,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)
