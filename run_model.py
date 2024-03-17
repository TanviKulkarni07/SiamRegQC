import glob
import random

import torch
from torchvision import transforms

from utils.common_utils import read_volseg, visualise_AE_ipop, intensity_normalize
from utils.dataloader import AEDataset_Elastic

from models.model import Autoencoder, SiameseNetwork_AE, Autoencoder_Simon
from models.train_cv import train_cv
from config.config import *
from infer import infer_test

vol,_ = read_volseg('../reg_n4/IXI002-Guys-0828-T1.nii.gz')
vol = intensity_normalize(vol)
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.CenterCrop(256),
                               ])

print('-----------Create Dataset------------')
GT_files = list(glob.glob('../reg_n4/*Guys*T1*'))
axis_to_extract = 1
start_slice = 40
end_slice = -40

true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
                          axis=axis_to_extract,is_true=True)
distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
                               axis=axis_to_extract,is_true=False)

data = true_dataset + distorted_dataset
train_files = random.sample(GT_files,10)
test_files = random.sample(GT_files,5)
test_files = [x for x in test_files if x not in train_files]

for file in train_files:
    vol, _ = read_volseg(file)
    vol = intensity_normalize(vol)    

    true_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform, 
                              axis=axis_to_extract, is_true=True)
    distorted_dataset = AEDataset_Elastic(vol[:,start_slice:end_slice,:], transform,
                                   axis=axis_to_extract,is_true=False)
    
    data += true_dataset + distorted_dataset
    
print('Num Train Sections:', len(data))
print('Num Test Vols:', len(test_files))
    
print('-------------Instantiate Model---------------')
device = 'cuda:0'

# Instantiate the model
autoencoder = Autoencoder()
autoencoder = autoencoder.to(device)
if add_pretrain:
    autoencoder.load_state_dict(torch.load('../model_weights/MRI_AE.pt'))

ae_siam = SiameseNetwork_AE(autoencoder.encoder).to(device)

# ae_siam = Autoencoder_Simon().to(device)

optimizer = torch.optim.Adam(ae_siam.parameters(), lr=learning_rate)

print('------------Training Begins!------------')
train_cv(ae_siam, data, optimizer, device, add_cc_loss, num_folds, num_epochs, batch_size, model_filename)

print('------------Inference Begins!----------------')
infer_test(test_files, ae_siam, device)