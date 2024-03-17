import sys
sys.path.insert(0,'/home/gayathri/Tanvi_Temp/for_vxm/lib/python3.8/site-packages/')
import os
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm


import numpy as np
from matplotlib import pyplot as plt
import torch

import time
from utils.common_utils import read_volseg, visualise_AE_ipop, intensity_normalize
from utils.dataloader import AEDataset_Elastic
from utils.losses import SiamRegQC_loss, SiamRegQC_multimodal_loss, Simonvosky_loss
from create_dataset import create_multimodal_data
from torchvision import transforms
import glob
from torch.utils.data import DataLoader
import random


# vol,_ = read_volseg('../reg_n4/IXI002-Guys-0828-T1.nii.gz')
# vol = intensity_normalize(vol)
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.CenterCrop(256),
#                                ])

# print('-----------Create Dataset------------')
# GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
# axis_to_extract = 1
# start_slice = 40
# end_slice = -40

# # true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
# #                           axis=axis_to_extract,is_true=True)
# distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
#                                axis=axis_to_extract,is_true=False)

# data = distorted_dataset
# train_files = random.sample(GT_files,10)
# test_files = random.sample(GT_files,5)
# test_files = [x for x in test_files if x not in train_files]

# for file in train_files:
#     vol, _ = read_volseg(file)
#     vol = intensity_normalize(vol)    

#     # true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
#     #                           axis=axis_to_extract, is_true=True)
#     distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
#                                    axis=axis_to_extract,is_true=False)
    
#     data +=  distorted_dataset

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])

GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
train_files = random.sample(GT_files,10)
test_files = random.sample(GT_files,5)
test_files = [x for x in test_files if x not in train_files]
print('-----------Create Dataset------------')
data, train_loader =create_multimodal_data(train_files, transform)
    
print('Num Train Sections:', len(data))
print('Num Test Vols:', len(test_files))

ndim = 1
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# vol_shape = torch_fxd.shape[1:-1]
vol_shape = (256,256)
batch_size=32
device = 'cuda:0'

vxm_model = vxm.networks.VxmDense(vol_shape,nb_features, int_steps=0).to(device)
# vxm_model.load_state_dict(torch.load('../model_weights/VXM_SiamRegQC.pt'))
train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
generator = iter(train_loader)
ss_loss = SiamRegQC_loss(device, '../model_weights/MRI_GuysT1_NL_20240309-223041_BS32_EP10.pt')
# ss_loss = Simonvosky_loss(device, '../model_weights/MRI_Guys_multimodal_Simon_NPNL_20240313-134110_BS32_EP10.pt')
# ss_loss = SiamRegQC_loss(device)
# ss_loss = SiamRegQC_multimodal_loss(device) 
# ss_loss = SiamRegQC_multimodal_loss(device,'../model_weights/MRI_Guys_multimodal_NL_20240313-153739_BS32_EP10.pt') 
# vxm.losses.NCC().loss
losses = [ss_loss, vxm.losses.Grad('l2').loss]
weights = [0.5, 0.5]
optimizer = torch.optim.Adam(vxm_model.parameters(), lr=1e-3)

best_loss = 1000000
# training loops
num_ep = 60
print('-----------Training Begins!---------------')
for epoch in range(num_ep):


    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
  
    for inputs1, inputs2, _ in train_loader:
        step_start_time = time.time()
        # run inputs through the model to produce a warped image and flow field
        inputs1 = inputs1.type(torch.FloatTensor).to(device)
        inputs2 = inputs2.type(torch.FloatTensor).to(device)

        out = vxm_model(inputs1, inputs2)
        
        zeros = torch.zeros_like(out[1].unsqueeze(1)).to(device)
        ytrue = [inputs2, zeros]
        ypred = [out[0], out[1].unsqueeze(1)]

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(ytrue[n], ypred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    if (epoch+1)%2==0:
                # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, num_ep)
        time_info = '%.4f sec/step' % np.mean(epoch_step_time)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
        if loss.item()<best_loss:
            best_loss = loss.item()
            print('-----Saving Model---------')
            print('Best Loss Achieved!:', best_loss)
            torch.save(vxm_model.state_dict(), '../model_weights/VXM_multimodal_DeepSim.pt') 


# visualise_AE_ipop(out[0], 32)

# test_data = distorted_dataset
# for file in test_files:
#     vol, _ = read_volseg(file)
#     vol = intensity_normalize(vol)    

#     true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
#                             axis=axis_to_extract, is_true=True)
#     distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
#                                 axis=axis_to_extract,is_true=False)
    
#     test_data +=  distorted_dataset

# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
tdata, test_dataloader =create_multimodal_data(test_files, transform)
t_losses = 0.
for inputs1, inputs2, labels in test_dataloader:

    inputs1 = inputs1.type(torch.FloatTensor).to(device)
    inputs2 = inputs2.type(torch.FloatTensor).to(device)

    out = vxm_model(inputs1, inputs2)
    t_losses += losses[0](inputs2, out[0])
print(f'Average Test Loss: {t_losses/len(test_dataloader):.3f}')