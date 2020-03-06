import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


from label_io import read_data_names, read_labels, plot_image, chw, hwc
from dataset import SegDataset, get_loader_train_val
from model import SegmentNet
from train import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
#plt.rcParams["figure.figsize"] = (24, 16)

###################################
# TRAINING           ##############
###################################

config = dict(num_epochs=5000, learning_rates=1e-4, save_every=50,
              all_model_save=0.99,
              is_lr_decay=True, lrdecay_thres=0.1, lrdecay_every=250, lrdecay_window = 50,
              model_save_dest="./model", dropout_prob=0.5
              )
batch_size = 24
config['model_name'] = 'Seg'

####    DataLoader
loader_train, loader_val = get_loader_train_val(batch_size_tr=batch_size)

#check train data

# _td = next(iter(loader_train))
# _im = _td['image'][0]
# _lab = _td['label'][0]
#
# labsample = np.asarray(_lab)
# print(np.max(labsample))
# plt.figure()
# plt.subplot(211)
# plot_image(_im)
# plt.subplot(212)
# plot_image(_lab)
# plt.show()

model = SegmentNet().to(device)

trainer = Trainer(model=model,
                  optimizer=torch.optim.Adam(model.parameters(), lr=config['learning_rates']),
                  loader_train = loader_train, loader_val = loader_val, criterion = nn.MSELoss(), **config)
trainer.load_model('Seg_ep187_tL8.61e-03_vL2.67e-02.tar')
trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=config['learning_rates'])
trainer.train()
trainer.test(test_loader=loader_val)



