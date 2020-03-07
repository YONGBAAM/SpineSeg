import numpy as np
import os
import torch
import torch.nn as nn
from dataset import SegDataset, get_loader_train_val
from line_dataset import get_lineloader_train_val
from model import SegmentNet
from train import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
###################################
# TRAINING           ##############
###################################

config = dict(num_epochs=1000, learning_rates=1e-4, save_every=50,
              all_model_save=0.99,
              is_lr_decay=True, lrdecay_thres=0.1, lrdecay_every=50, lrdecay_window = 20,
              model_save_dest="./model", dropout_prob=0.5
              )

batch_size = 24

config['model_name'] = 'Line'

####    DataLoader
lineloader_train, lineloader_val = get_lineloader_train_val(batch_size)

model = SegmentNet().to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train = lineloader_train, loader_val = lineloader_val, criterion = nn.MSELoss(), **config)

# trainer.load_model('Line_ep800_tL2.69e-07_vL2.28e-07.tar')

trainer.train()
trainer.test(test_loader=lineloader_val)



